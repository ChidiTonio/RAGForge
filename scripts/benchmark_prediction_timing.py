"""Opt-in host-clock instrumentation for CrossEncoder prediction.

Wrappers themselves add no CUDA calls. An optional pre-transfer callback
may deliberately perform and synchronize GPU work in a labeled intervention.
Intervals measure host elapsed time, not isolated GPU execution time.
"""

from contextlib import contextmanager
from functools import wraps
from inspect import unwrap
from time import perf_counter


@contextmanager
def prediction_timing(
    backend, *, clock=perf_counter, detailed=False, before_input_transfer=None,
):
    """Temporarily wrap predict, to, and preprocess on one backend instance.

    The caller must not use this instance concurrently. Device placement and
    preprocessing totals include only calls made inside predict. All methods
    are restored on normal exit and on exceptions.
    """
    # Detailed mode temporarily patches the prediction module's transfer
    # reference. No concurrent use of that module is supported.
    if before_input_transfer is not None and not detailed:
        raise ValueError("A pre-transfer probe requires detailed timing.")
    names = ("predict", "to", "preprocess")
    if detailed:
        names += ("forward",)
    transfer_globals = None
    original_transfer = None
    if detailed:
        predict_function = unwrap(backend.predict)
        transfer_globals = getattr(predict_function, "__globals__", None)
        if transfer_globals is None:
            raise RuntimeError("Cannot inspect prediction function globals.")
        original_transfer = transfer_globals.get("batch_to_device")
        if not callable(original_transfer):
            raise RuntimeError("Prediction has no callable batch_to_device.")
    originals = {name: getattr(backend, name) for name in names}
    instance_values = {
        name: (name in vars(backend), vars(backend).get(name))
        for name in names
    }
    totals = {
        "predict_ms": 0.0,
        "device_placement_ms": 0.0,
        "preprocess_ms": 0.0,
        "predict_calls": 0,
        "device_placement_calls": 0,
        "preprocess_calls": 0,
    }
    if detailed:
        originals["input_transfer"] = original_transfer
        totals.update({
            "input_transfer_ms": 0.0,
            "input_transfer_calls": 0,
            "forward_ms": 0.0,
            "forward_calls": 0,
        })
    if before_input_transfer is not None:
        totals.update({"cuda_probe_ms": 0.0, "cuda_probe_calls": 0})
    probe_pending = False
    predict_depth = 0

    def stage_wrapper(name, key):
        original = originals[name]

        @wraps(original)
        def measured(*args, **kwargs):
            nonlocal probe_pending
            if not predict_depth:
                return original(*args, **kwargs)
            if name == "input_transfer" and probe_pending:
                probe_pending = False
                probe_started = clock()
                try:
                    before_input_transfer()
                finally:
                    totals["cuda_probe_ms"] += (clock() - probe_started) * 1000
                    totals["cuda_probe_calls"] += 1
            started = clock()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = (clock() - started) * 1000
                totals[f"{key}_ms"] += elapsed
                totals[f"{key}_calls"] += 1

        return measured

    @wraps(originals["predict"])
    def measured_predict(*args, **kwargs):
        nonlocal predict_depth, probe_pending
        if predict_depth:
            raise RuntimeError("Nested predict calls are unsupported.")
        started = clock()
        predict_depth += 1
        probe_pending = before_input_transfer is not None
        try:
            return originals["predict"](*args, **kwargs)
        finally:
            predict_depth -= 1
            totals["predict_ms"] += (clock() - started) * 1000
            totals["predict_calls"] += 1

    installed = []
    transfer_installed = False
    try:
        for name, wrapper in (
            ("to", stage_wrapper("to", "device_placement")),
            ("preprocess", stage_wrapper("preprocess", "preprocess")),
            ("predict", measured_predict),
        ):
            setattr(backend, name, wrapper)
            installed.append(name)
        if detailed:
            setattr(backend, "forward", stage_wrapper("forward", "forward"))
            installed.append("forward")
            transfer_globals["batch_to_device"] = stage_wrapper(
                "input_transfer", "input_transfer"
            )
            transfer_installed = True
        yield totals
    finally:
        if transfer_installed:
            transfer_globals["batch_to_device"] = original_transfer
        for name in reversed(installed):
            was_present, value = instance_values[name]
            if was_present:
                setattr(backend, name, value)
            else:
                delattr(backend, name)
        # The remainder includes output conversion, pending GPU waits,
        # sorting, activation, score collection, and wrapper overhead.
        totals["predict_other_ms"] = (
            totals["predict_ms"]
            - totals["device_placement_ms"]
            - totals["preprocess_ms"]
            - totals.get("input_transfer_ms", 0.0)
            - totals.get("forward_ms", 0.0)
            - totals.get("cuda_probe_ms", 0.0)
        )
