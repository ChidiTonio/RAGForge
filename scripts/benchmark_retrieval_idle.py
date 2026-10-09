"""Measure retrieval latency after controlled application-idle intervals.

Uses a synthetic query, not labeled evaluation questions. Sleep, validation,
setup, and serialization are excluded from request timings. Application
inactivity does not guarantee GPU inactivity or a particular power state.
"""

import argparse
import hashlib
import json
import math
import platform
import random
from dataclasses import fields
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, sleep

import benchmark_retrieval_e2e as baseline
from benchmark_prediction_timing import prediction_timing
from evaluate_retrieval import load_json

from ragforge.ingestion.chunking import TextChunk

DEFAULT_QUERY = "Explain how students can investigate a scientific hypothesis."


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Must be finite and nonnegative.")
    return number


def load_corpus(path):
    """Validate corpus records without loading any evaluation questions."""
    manifest, corpus_hash = load_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported corpus schema_version.")
    records = manifest.get("chunks")
    if not isinstance(records, list) or not records:
        raise ValueError("Corpus must contain a nonempty chunks list.")
    if manifest.get("chunk_count") != len(records):
        raise ValueError("Manifest chunk_count does not match chunks.")

    names = {field.name for field in fields(TextChunk)}
    chunks = []
    chunk_to_id = {}
    known_ids = set()
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Chunk record {position} must be an object.")
        if set(record) != names | {"id"}:
            raise ValueError(f"Chunk record {position} has unexpected fields.")
        chunk_id = record["id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Chunk ID must be nonblank text.")
        if chunk_id in known_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")
        values = {name: record[name] for name in names}
        for name in ("source", "text"):
            if not isinstance(values[name], str) or not values[name].strip():
                raise ValueError(f"{chunk_id}: {name} must be nonblank text.")
        for name in ("page_number", "chunk_number", "start_char", "end_char"):
            if type(values[name]) is not int:
                raise ValueError(f"{chunk_id}: {name} must be an integer.")
        if values["page_number"] <= 0 or values["chunk_number"] <= 0:
            raise ValueError(f"{chunk_id}: invalid page or chunk number.")
        if not 0 <= values["start_char"] < values["end_char"]:
            raise ValueError(f"{chunk_id}: invalid character offsets.")
        if len(values["text"]) != values["end_char"] - values["start_char"]:
            raise ValueError(f"{chunk_id}: text length does not match offsets.")
        expected = f"p{values['page_number']:03d}-c{values['chunk_number']:03d}"
        if chunk_id != expected:
            raise ValueError(f"{chunk_id}: ID does not match chunk metadata.")
        chunk = TextChunk(**values)
        if chunk in chunk_to_id:
            raise ValueError(f"{chunk_id}: duplicate TextChunk value.")
        chunks.append(chunk)
        chunk_to_id[chunk] = chunk_id
        known_ids.add(chunk_id)
    return chunks, chunk_to_id, corpus_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("evaluation/stem_education/corpus.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reranker-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--idle-seconds", nargs="+", type=nonnegative_float,
        default=[0.0, 0.1, 1.0, 5.0, 15.0],
    )
    parser.add_argument("--trials", type=baseline.positive_int, default=5)
    parser.add_argument("--warmup-queries", type=baseline.positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=baseline.positive_int)
    parser.add_argument(
        "--stage-timing", action="store_true",
        help="Add a host timestamp after CPU retrieval; no intermediate CUDA sync.",
    )
    parser.add_argument(
        "--prediction-timing", action="store_true",
        help="Time backend predict/to/preprocess; requires --stage-timing.",
    )
    parser.add_argument(
        "--prediction-detail-timing", action="store_true",
        help="Also time input transfer and forward; requires --prediction-timing.",
    )
    parser.add_argument(
        "--cuda-pre-transfer-probe", action="store_true",
        help="Time tiny GPU work and sync before input transfer; CUDA/detail only.",
    )
    args = parser.parse_args()
    if args.cuda_pre_transfer_probe and (
        not args.prediction_detail_timing or args.reranker_device != "cuda"
    ):
        parser.error(
            "--cuda-pre-transfer-probe requires "
            "--prediction-detail-timing and --reranker-device cuda."
        )

    if args.prediction_detail_timing and not args.prediction_timing:
        parser.error("--prediction-detail-timing requires --prediction-timing.")

    if args.prediction_timing and not args.stage_timing:
        parser.error("--prediction-timing requires --stage-timing.")

    if args.output.exists():
        parser.error("Output exists. Choose a new --output path.")
    if args.output.resolve() == args.corpus.resolve():
        parser.error("Output must not overwrite the corpus.")
    if not args.query.strip():
        parser.error("Query must be nonblank.")
    if len(set(args.idle_seconds)) != len(args.idle_seconds):
        parser.error("Idle intervals must be unique.")
    try:
        chunks, chunk_to_id, corpus_hash = load_corpus(args.corpus)
        synchronize = baseline.device_synchronizer(args.reranker_device)
    except (OSError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    torch = baseline.torch
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    schedule = []
    rng = random.Random(args.seed)
    for trial in range(1, args.trials + 1):
        gaps = list(args.idle_seconds)
        rng.shuffle(gaps)
        schedule.extend((trial, gap) for gap in gaps)

    print(f"Validated {len(chunks)} chunks; no labeled questions loaded.", flush=True)
    print("Loading models and building CPU hybrid index...", flush=True)
    started = perf_counter()
    embedder = baseline.SentenceTransformerEmbedder(
        baseline.EMBEDDING_MODEL, revision=baseline.EMBEDDING_REVISION,
        device="cpu", batch_size=baseline.BATCH_SIZE,
    )
    hybrid = baseline.HybridRetriever(
        chunks, embedder=embedder,
        candidate_k=baseline.BACKEND_CANDIDATE_K, rrf_k=baseline.RRF_K,
    )
    scorer = baseline.CrossEncoderScorer(
        baseline.RERANKER_MODEL, revision=baseline.RERANKER_REVISION,
        device=args.reranker_device, batch_size=baseline.BATCH_SIZE,
        max_length=baseline.MAX_LENGTH,
    )
    reranker = baseline.Reranker(scorer=scorer)
    synchronize()
    setup_ms = (perf_counter() - started) * 1000
    model = scorer._model.model
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    devices = sorted({str(parameter.device) for parameter in model.parameters()})

    # Allocate outside request timings, before warmup or idle trials.
    probe_tensor = None
    if args.cuda_pre_transfer_probe:
        probe_tensor = torch.zeros(1, device=scorer._model.device)
        synchronize()

    def cuda_probe():
        probe_tensor.add_(1)
        synchronize()

    def measured_request():
        # Deliberately no CUDA call before the start clock.
        started = perf_counter()
        pool = hybrid.search(args.query, top_k=baseline.FINAL_POOL_K)
        retrieval_end = perf_counter() if args.stage_timing else None
        results = reranker.rerank(args.query, pool, top_k=baseline.TOP_K)
        synchronize()
        ended = perf_counter()
        # Validation happens only after the end clock.
        if not pool or len(pool) > baseline.FINAL_POOL_K:
            raise ValueError("Unexpected candidate-pool size.")
        baseline.candidate_ranking(pool, chunk_to_id, set())
        if len(results) != min(baseline.TOP_K, len(pool)):
            raise ValueError("Unexpected result count.")
        if not {result.chunk for result in results} <= {item.chunk for item in pool}:
            raise ValueError("Reranking introduced a new chunk.")
        ranking = baseline.candidate_ranking(results, chunk_to_id, set())
        measurement = {
            "end_to_end_ms": (ended - started) * 1000,
            "candidate_count": len(pool),
            "top_k_chunk_ids": [item["chunk_id"] for item in ranking],
        }
        if args.stage_timing:
            measurement.update({
                "retrieval_ms": (retrieval_end - started) * 1000,
                "reranking_ms": (ended - retrieval_end) * 1000,
            })
        return measurement, started, ended

    def request():
        if not args.prediction_timing:
            return measured_request()
        # Install host-only wrappers before the E2E start clock.
        with prediction_timing(
            scorer._model, clock=perf_counter,
            detailed=args.prediction_detail_timing,
            before_input_transfer=(
                cuda_probe if args.cuda_pre_transfer_probe else None
            ),
        ) as totals:
            measurement, started, ended = measured_request()
        measurement["prediction"] = dict(totals)
        return measurement, started, ended

    first, _, _ = request()
    for _ in range(args.warmup_queries):
        request()
    reference, _, _ = request()

    print(f"Measuring {len(schedule)} idle trials...", flush=True)
    samples = []
    for sequence, (trial, gap) in enumerate(schedule, start=1):
        # Conditioning completes and synchronizes before the idle interval.
        conditioning, _, conditioning_end = request()
        sleep_start = perf_counter()
        if gap:
            sleep(gap)
        sample, request_start, _ = request()
        sample.update({
            "sequence": sequence,
            "trial": trial,
            "requested_idle_seconds": gap,
            "sleep_window_seconds": request_start - sleep_start,
            "application_idle_seconds": request_start - conditioning_end,
            "conditioning_end_to_end_ms": conditioning["end_to_end_ms"],
            "matches_warm_reference": (
                sample["top_k_chunk_ids"] == reference["top_k_chunk_ids"]
            ),
        })
        if args.stage_timing:
            sample.update({
                "conditioning_retrieval_ms": conditioning["retrieval_ms"],
                "conditioning_reranking_ms": conditioning["reranking_ms"],
            })
        if args.prediction_timing:
            sample["conditioning_prediction"] = conditioning["prediction"]
        samples.append(sample)
        print(
            f"{sequence}/{len(schedule)} | gap {gap:g}s | "
            f"request {sample['end_to_end_ms']:.2f} ms",
            flush=True,
        )

    summaries = []
    for gap in args.idle_seconds:
        rows = [s for s in samples if s["requested_idle_seconds"] == gap]
        summaries.append({
            "requested_idle_seconds": gap,
            "sample_count": len(rows),
            "end_to_end": baseline.latency_summary(
                [s["end_to_end_ms"] for s in rows]
            ),
            "mean_application_idle_seconds": sum(
                s["application_idle_seconds"] for s in rows
            ) / len(rows),
            "warm_reference_order_agreement": sum(
                s["matches_warm_reference"] for s in rows
            ) / len(rows),
        })

    if args.stage_timing:
        for summary in summaries:
            rows = [
                s for s in samples
                if s["requested_idle_seconds"] == summary["requested_idle_seconds"]
            ]
            for stage in ("retrieval", "reranking"):
                summary[stage] = baseline.latency_summary(
                    [s[f"{stage}_ms"] for s in rows]
                )

    if args.prediction_timing:
        for summary in summaries:
            rows = [
                s for s in samples
                if s["requested_idle_seconds"] == summary["requested_idle_seconds"]
            ]
            summary["prediction"] = {
                key: baseline.latency_summary(
                    [s["prediction"][key] for s in rows]
                )
                for key in (
                    "predict_ms", "device_placement_ms",
                    "preprocess_ms", "predict_other_ms",
                ) + (
                    ("input_transfer_ms", "forward_ms")
                    if args.prediction_detail_timing else ()
                ) + (
                    ("cuda_probe_ms",) if args.cuda_pre_transfer_probe else ()
                )
            }

    report = {
        "report_schema_version": (
            5 if args.cuda_pre_transfer_probe
            else 4 if args.prediction_detail_timing
            else 3 if args.prediction_timing
            else 2 if args.stage_timing else 1
        ),
        "experiment": (
            "controlled_application_idle_hybrid_reranking_cuda_pre_transfer_probe"
            if args.cuda_pre_transfer_probe
            else "controlled_application_idle_hybrid_reranking_prediction_detail_timing"
            if args.prediction_detail_timing
            else "controlled_application_idle_hybrid_reranking_prediction_timing"
            if args.prediction_timing
            else "controlled_application_idle_hybrid_reranking_stage_timing"
            if args.stage_timing
            else "controlled_application_idle_hybrid_reranking"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpus_path": str(args.corpus),
        "corpus_sha256": corpus_hash,
        "chunk_count": len(chunks),
        "query": args.query,
        "query_role": "synthetic latency probe; no quality evaluation",
        "git": {
            "commit": baseline.git_output("rev-parse", "HEAD"),
            "status_porcelain": baseline.git_output("status", "--porcelain"),
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__), Path(baseline.__file__),
                Path(__file__).with_name("benchmark_prediction_timing.py"),
                Path(__file__).with_name("evaluate_retrieval.py"),
                Path(__file__).with_name("evaluate_reranking.py"),
            )
        },
        "settings": {
            "embedding_model": baseline.EMBEDDING_MODEL,
            "embedding_revision": baseline.EMBEDDING_REVISION,
            "embedding_device": "cpu",
            "reranker_model": baseline.RERANKER_MODEL,
            "reranker_revision": baseline.RERANKER_REVISION,
            "reranker_device": args.reranker_device,
            "reranker_parameter_dtypes": dtypes,
            "reranker_parameter_devices": devices,
            "batch_size": baseline.BATCH_SIZE,
            "max_length": baseline.MAX_LENGTH,
            "top_k": baseline.TOP_K,
            "backend_candidate_k": baseline.BACKEND_CANDIDATE_K,
            "final_pool_k": baseline.FINAL_POOL_K,
            "rrf_k": baseline.RRF_K,
            "warmup_queries": args.warmup_queries,
            "trials_per_gap": args.trials,
            "idle_seconds": args.idle_seconds,
            "seed": args.seed,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "stage_timing": args.stage_timing,
            "prediction_timing": args.prediction_timing,
            "prediction_detail_timing": args.prediction_detail_timing,
            "cuda_pre_transfer_probe": args.cuda_pre_transfer_probe,
            "cuda_probe_definition": (
                "Once per predict, before first batch_to_device: increment "
                "one preallocated CUDA value and synchronize. Host elapsed "
                "time is included in predict and E2E, reported separately "
                "as cuda_probe_ms, and excluded from predict_other_ms."
                if args.cuda_pre_transfer_probe else None
            ),
            "prediction_detail_timing_definition": (
                "Host elapsed batch_to_device and forward calls; no added CUDA "
                "synchronization. predict_other_ms excludes these intervals "
                "and includes output conversion and any pending GPU waits."
                if args.prediction_detail_timing else None
            ),
            "prediction_timing_definition": (
                "Host elapsed predict time; nested to and preprocess totals; "
                "predict_other_ms is the remainder, not GPU execution time."
                if args.prediction_timing else None
            ),
            "timing": (
                "No pre-request sync; timed CUDA probe and intermediate sync "
                "before first input transfer; post-request sync before end clock."
                if args.cuda_pre_transfer_probe else
                "No pre-request sync; post-request sync before end clock."
            ),
            "stage_timing_definition": (
                "Host timestamp after CPU hybrid search; reranking includes "
                "host overhead, transfers, and final synchronization; "
                "no intermediate CUDA synchronization."
                if args.stage_timing else None
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name()
                if args.reranker_device == "cuda" else None
            ),
            "package_versions": {
                name: version(name)
                for name in ("sentence-transformers", "transformers", "torch", "numpy")
            },
        },
        "setup_ms": setup_ms,
        "first_query_after_setup": first,
        "warm_reference": reference,
        "per_gap": summaries,
        "samples": samples,
        "limitations": [
            "One synthetic query; no retrieval-quality or generalization claim.",
            "Application inactivity does not establish GPU power state.",
            "No power, clock, or thermal telemetry captured by this script.",
            "Small per-gap samples; tail estimates are exploratory.",
            "No HTTP, answer generation, concurrent load, or process-startup timing.",
            "Conditioning is a full pipeline request, not a GPU-only intervention.",
            "Sleep-window timing includes host scheduling and request-call overhead.",
        ],
    }
    if args.stage_timing:
        report["limitations"].append(
            "Stage timing adds a host timestamp; reranking_ms is not "
            "GPU-only execution time."
        )
    if args.prediction_timing:
        report["limitations"].extend([
            "Prediction wrappers add host-clock and Python overhead.",
            (
                "Wrappers are installed before the E2E start clock and restored "
                "after validation; surrounding application-idle timing includes "
                "this host overhead."
            ),
            (
                "Nested stage intervals may include asynchronous GPU waiting; "
                "predict_other_ms is not GPU-only execution time."
            ),
        ])
    if args.prediction_detail_timing:
        report["limitations"].extend([
            "Detailed timing temporarily wraps a module-level transfer "
            "reference; concurrent prediction in this process is unsupported.",
            "Forward host time is not completed GPU execution time; "
            "the remainder is not isolated output-transfer time.",
        ])
    if args.cuda_pre_transfer_probe:
        report["settings"]["stage_timing_definition"] = (
            "Host timestamp after CPU retrieval; reranking includes a timed "
            "pre-transfer GPU probe and intermediate synchronization."
        )
        report["settings"]["prediction_detail_timing_definition"] = (
            "Host elapsed input-transfer and forward calls after a timed "
            "GPU probe with synchronization. The remainder excludes the probe."
        )
        report["limitations"].append(
            "The pre-transfer probe deliberately changes GPU scheduling "
            "and synchronization; this is an intervention, not baseline latency."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    for summary in summaries:
        latency = summary["end_to_end"]
        print(
            f"Gap {summary['requested_idle_seconds']:g}s | "
            f"mean {latency['mean_ms']:.2f} ms | "
            f"median {latency['median_ms']:.2f} ms"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
