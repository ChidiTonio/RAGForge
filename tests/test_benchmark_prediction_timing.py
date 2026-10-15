"""Download-free tests for temporary backend timing wrappers."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def timing(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("benchmark_prediction_timing")


class Backend:
    def __init__(self, now):
        self.now = now
        self.fail = False

    def to(self, device):
        self.now[0] += 0.003
        return self

    def preprocess(self, batch):
        self.now[0] += 0.005
        return batch

    def predict(self, pairs, *, device="cuda"):
        self.to(device)
        for batch in (pairs[:1], pairs[1:]):
            self.preprocess(batch)
        self.now[0] += 0.020
        if self.fail:
            raise ValueError("prediction failed")
        return [1.0] * len(pairs)


def test_totals_arguments_and_restoration(timing):
    now = [0.0]
    backend = Backend(now)
    before = dict(vars(backend))
    with timing.prediction_timing(backend, clock=lambda: now[0]) as totals:
        # Placement outside predict must not enter the prediction total.
        assert backend.to("cpu") is backend
        assert backend.predict(["a", "b"], device="cpu") == [1.0, 1.0]
        assert backend.predict(["c", "d"]) == [1.0, 1.0]
    assert vars(backend) == before
    assert totals["predict_ms"] == pytest.approx(66.0)
    assert totals["device_placement_ms"] == pytest.approx(6.0)
    assert totals["preprocess_ms"] == pytest.approx(20.0)
    assert totals["predict_other_ms"] == pytest.approx(40.0)
    assert totals["predict_calls"] == 2
    assert totals["device_placement_calls"] == 2
    assert totals["preprocess_calls"] == 4


def test_prediction_failure_restores_methods(timing):
    now = [0.0]
    backend = Backend(now)
    backend.fail = True
    before = dict(vars(backend))
    with (
        pytest.raises(ValueError, match="prediction failed"),
        timing.prediction_timing(backend, clock=lambda: now[0]) as totals,
    ):
        backend.predict(["a", "b"])
    assert vars(backend) == before
    assert totals["predict_ms"] == pytest.approx(33.0)
    assert totals["predict_other_ms"] == pytest.approx(20.0)


def test_existing_instance_method_is_restored(timing):
    now = [0.0]
    backend = Backend(now)

    def custom_to(device):
        now[0] += 0.007
        return backend

    backend.to = custom_to
    with timing.prediction_timing(backend, clock=lambda: now[0]) as totals:
        backend.predict(["a", "b"])
    assert backend.to is custom_to
    assert "predict" not in vars(backend)
    assert "preprocess" not in vars(backend)
    assert totals["device_placement_ms"] == pytest.approx(7.0)
    assert totals["predict_ms"] == pytest.approx(37.0)


def test_context_failure_without_prediction_restores_methods(timing):
    backend = Backend([0.0])
    before = dict(vars(backend))
    with (
        pytest.raises(RuntimeError, match="caller failed"),
        timing.prediction_timing(backend) as totals,
    ):
        raise RuntimeError("caller failed")
    assert vars(backend) == before
    assert totals["predict_calls"] == 0
    assert totals["predict_other_ms"] == 0.0


def batch_to_device(now, device):
    """Fake transfer used by DetailedBackend.predict."""
    now[0] += 0.002
    return now


class DetailedBackend(Backend):
    def __init__(self, now):
        super().__init__(now)
        self.fail_stage = None

    def forward(self, batch):
        self.now[0] += 0.004
        if self.fail_stage == "forward":
            raise RuntimeError("forward failed")
        return batch

    def predict(self, pairs, *, device="cuda"):
        self.to(device)
        self.preprocess(pairs)
        for _ in range(2):
            batch = batch_to_device(self.now, device)
            self.forward(batch)
        self.now[0] += 0.006
        return [1.0] * len(pairs)


@pytest.mark.parametrize("use_probe", [False, True])
def test_detailed_totals_probe_and_restoration(timing, use_probe):
    now = [0.0]
    backend = DetailedBackend(now)
    before = dict(vars(backend))
    original_transfer = batch_to_device
    probe_calls = []

    def probe():
        probe_calls.append(True)
        now[0] += 0.007

    with timing.prediction_timing(
        backend,
        clock=lambda: now[0],
        detailed=True,
        before_input_transfer=probe if use_probe else None,
    ) as totals:
        assert backend.predict(["a", "b"]) == [1.0, 1.0]
        assert backend.predict(["c"]) == [1.0]

    assert vars(backend) == before
    assert batch_to_device is original_transfer
    assert totals["predict_calls"] == 2
    assert totals["predict_ms"] == pytest.approx(66 if use_probe else 52)
    assert totals["device_placement_ms"] == pytest.approx(6)
    assert totals["preprocess_ms"] == pytest.approx(10)
    assert totals["input_transfer_ms"] == pytest.approx(8)
    assert totals["input_transfer_calls"] == 4
    assert totals["forward_ms"] == pytest.approx(16)
    assert totals["forward_calls"] == 4
    assert totals["predict_other_ms"] == pytest.approx(12)
    if use_probe:
        assert len(probe_calls) == 2
        assert totals["cuda_probe_calls"] == 2
        assert totals["cuda_probe_ms"] == pytest.approx(14)
    else:
        assert not probe_calls
        assert "cuda_probe_ms" not in totals


@pytest.mark.parametrize("failure", ["transfer", "forward", "probe"])
def test_detailed_failure_restores_all_patches(timing, monkeypatch, failure):
    now = [0.0]
    backend = DetailedBackend(now)
    if failure == "forward":
        backend.fail_stage = "forward"

    prediction_globals = DetailedBackend.predict.__globals__
    if failure == "transfer":
        def failing_transfer(now, device):
            now[0] += 0.002
            raise RuntimeError("transfer failed")

        monkeypatch.setitem(
            prediction_globals, "batch_to_device", failing_transfer,
        )

    original_transfer = prediction_globals["batch_to_device"]
    before = dict(vars(backend))

    def failing_probe():
        now[0] += 0.007
        raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError, match=f"{failure} failed"), timing.prediction_timing(
        backend,
        clock=lambda: now[0],
        detailed=True,
        before_input_transfer=failing_probe if failure == "probe" else None,
    ) as totals:
        backend.predict(["a", "b"])

    assert vars(backend) == before
    assert prediction_globals["batch_to_device"] is original_transfer
    assert totals["predict_calls"] == 1
    if failure == "probe":
        assert totals["cuda_probe_calls"] == 1
        assert totals["cuda_probe_ms"] == pytest.approx(7)
        assert totals["input_transfer_calls"] == 0
    elif failure == "transfer":
        assert totals["input_transfer_calls"] == 1
        assert totals["input_transfer_ms"] == pytest.approx(2)
    else:
        assert totals["forward_calls"] == 1
        assert totals["forward_ms"] == pytest.approx(4)


def test_probe_requires_detailed_timing(timing):
    backend = Backend([0.0])
    before = dict(vars(backend))
    with pytest.raises(ValueError, match="requires detailed timing"), timing.prediction_timing(
        backend, before_input_transfer=lambda: None,
    ):
        pass
    assert vars(backend) == before
