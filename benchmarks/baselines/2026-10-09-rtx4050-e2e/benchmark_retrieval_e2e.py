"""Measure fresh hybrid retrieval plus reranking in one wall-clock interval.

Run from the repository root. Embedding and hybrid retrieval remain on CPU.
Model loading, index construction, validation, and serialization are excluded
from request timings. This does not measure answer generation or HTTP latency.
"""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from evaluate_reranking import (
    BACKEND_CANDIDATE_K,
    BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    FINAL_POOL_K,
    MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_REVISION,
    RRF_K,
    FrozenRetriever,
    candidate_ranking,
    device_synchronizer,
    latency_summary,
)
from evaluate_retrieval import TOP_K, evaluate, load_inputs

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.retrieval.cross_encoder import CrossEncoderScorer
from ragforge.retrieval.hybrid import HybridRetriever
from ragforge.retrieval.reranking import Reranker


def git_output(*arguments):
    """Record repository provenance without requiring Git to be available."""
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("evaluation/stem_education/corpus.json"),
    )
    parser.add_argument(
        "--questions", type=Path,
        default=Path("evaluation/stem_education/questions_heldout.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reranker-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--passes", type=positive_int, default=5)
    parser.add_argument("--warmup-passes", type=positive_int, default=2)
    parser.add_argument(
        "--torch-threads", type=positive_int,
        help="Optional intra-op thread override; otherwise retain PyTorch defaults.",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output exists. Choose a new --output path.")
    if args.output.resolve() in {args.corpus.resolve(), args.questions.resolve()}:
        parser.error("Output must not overwrite an input.")
    try:
        _, dataset, chunks, chunk_to_id, corpus_hash, questions_hash = load_inputs(
            args.corpus, args.questions
        )
        synchronize = device_synchronizer(args.reranker_device)
    except (OSError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    questions = dataset["questions"]
    texts = [question["question"] for question in questions]
    if len(set(texts)) != len(texts):
        parser.error("Unique question texts are required.")

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    # Use explicit TF32 settings; do not claim equivalence to older reports.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Validated: {len(chunks)} chunks | {len(questions)} questions", flush=True)
    print("Loading CPU embedder and building hybrid index...", flush=True)
    started = perf_counter()
    embedder = SentenceTransformerEmbedder(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
        device="cpu", batch_size=BATCH_SIZE,
    )
    embedder_load_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    hybrid = HybridRetriever(
        chunks, embedder=embedder, candidate_k=BACKEND_CANDIDATE_K, rrf_k=RRF_K,
    )
    index_build_ms = (perf_counter() - started) * 1000

    print(f"Loading reranker on {args.reranker_device}...", flush=True)
    synchronize()
    started = perf_counter()
    scorer = CrossEncoderScorer(
        RERANKER_MODEL, revision=RERANKER_REVISION,
        device=args.reranker_device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )
    reranker = Reranker(scorer=scorer)
    synchronize()
    reranker_load_ms = (perf_counter() - started) * 1000

    # Introspection only: no dtype conversion or adapter behavior changes.
    model = scorer._model.model
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})

    def timed_query(question):
        query = question["question"]
        synchronize()
        started = perf_counter()
        pool = hybrid.search(query, top_k=FINAL_POOL_K)
        results = reranker.rerank(query, pool, top_k=TOP_K)
        synchronize()
        elapsed_ms = (perf_counter() - started) * 1000

        # All benchmark validation is outside the measured interval.
        if not pool or len(pool) > FINAL_POOL_K:
            raise ValueError("Unexpected candidate-pool size.")
        candidate_ranking(pool, chunk_to_id, set(question["relevant_chunk_ids"]))
        if len(results) != min(TOP_K, len(pool)):
            raise ValueError("Unexpected result count.")
        if not {result.chunk for result in results} <= {item.chunk for item in pool}:
            raise ValueError("Reranking introduced a new chunk.")
        ranking = candidate_ranking(
            results, chunk_to_id, set(question["relevant_chunk_ids"])
        )
        return elapsed_ms, pool, results, [item["chunk_id"] for item in ranking]

    print("Measuring first query after setup...", flush=True)
    first_ms, first_pool, _, first_ids = timed_query(questions[0])

    print(f"Running {args.warmup_passes} full warm-up passes...", flush=True)
    for _ in range(args.warmup_passes):
        for question in questions:
            timed_query(question)

    samples = []
    reference_ids = {}
    first_rankings = {}
    first_pools = {}
    print(f"Running {args.passes} measured passes...", flush=True)
    for pass_index in range(args.passes):
        for question in questions:
            elapsed_ms, pool, results, ids = timed_query(question)
            question_id = question["id"]
            if pass_index == 0:
                reference_ids[question_id] = ids
                first_rankings[question["question"]] = results
                first_pools[question["question"]] = pool
            samples.append({
                "pass": pass_index + 1,
                "question_id": question_id,
                "candidate_count": len(pool),
                "end_to_end_ms": elapsed_ms,
                "top_k_chunk_ids": ids,
                "matches_first_measured_pass": ids == reference_ids[question_id],
            })
        print(f"Completed pass {pass_index + 1}/{args.passes}", flush=True)

    quality = {
        "hybrid": evaluate(FrozenRetriever(first_pools), questions, chunk_to_id),
        "hybrid_reranked": evaluate(
            FrozenRetriever(first_rankings), questions, chunk_to_id
        ),
    }
    summary = latency_summary([sample["end_to_end_ms"] for sample in samples])
    repeated_samples = [sample for sample in samples if sample["pass"] > 1]
    agreement = (
        sum(sample["matches_first_measured_pass"] for sample in repeated_samples)
        / len(repeated_samples)
        if repeated_samples else None
    )
    report = {
        "report_schema_version": 1,
        "experiment": "direct_end_to_end_hybrid_reranking",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Passage retrieval only; not answer evaluation.",
        "dataset_evaluation_role": dataset.get("evaluation_role"),
        "dataset_id": dataset.get("dataset_id"),
        "label_status": dataset.get("label_status"),
        "corpus_path": str(args.corpus),
        "corpus_sha256": corpus_hash,
        "questions_path": str(args.questions),
        "questions_sha256": questions_hash,
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "timed_query_count": len(samples),
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "status_porcelain": git_output("status", "--porcelain"),
        },
        "settings": {
            "top_k": TOP_K,
            "backend_candidate_k": BACKEND_CANDIDATE_K,
            "final_pool_k": FINAL_POOL_K,
            "rrf_k": RRF_K,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "embedding_device": "cpu",
            "reranker_model": RERANKER_MODEL,
            "reranker_revision": RERANKER_REVISION,
            "reranker_device": args.reranker_device,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "reranker_activation": "identity",
            "reranker_parameter_dtypes": parameter_dtypes,
            "reranker_parameter_devices": parameter_devices,
            "warmup_passes": args.warmup_passes,
            "warmup_queries": args.warmup_passes * len(questions),
            "timing_passes": args.passes,
            "query_order": "dataset order on every pass",
            "cuda_synchronized": args.reranker_device == "cuda",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
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
        "setup_latency": {
            "embedder_load_ms": embedder_load_ms,
            "hybrid_index_build_ms": index_build_ms,
            "reranker_load_ms": reranker_load_ms,
        },
        "first_query_after_setup": {
            "question_id": questions[0]["id"],
            "end_to_end_ms": first_ms,
            "candidate_count": len(first_pool),
            "top_k_chunk_ids": first_ids,
            "note": "Not process startup or a controlled idle-to-active measurement.",
        },
        "latency": {
            "end_to_end": summary,
            "sequential_service_rate_qps": 1000.0 / summary["mean_ms"],
            "note": (
                "Direct wall-clock timing of fresh query embedding, BM25, dense "
                "search, fusion, and reranking through top-k results. Excludes "
                "setup, validation, quality scoring, logging, and serialization. "
                "Service rate is the reciprocal of mean request latency, not "
                "concurrent throughput or whole-benchmark elapsed throughput."
            ),
        },
        "quality_first_measured_pass": quality,
        "repeat_top_k_order_agreement": agreement,
        "repeat_comparison_count": len(repeated_samples),
        "samples": samples,
        "limitations": [
            "Repeated timings do not increase the number of unique labeled questions.",
            "Small fixed corpus; not a production-scale or concurrent-load benchmark.",
            "Fixed query order; no randomized ordering or idle-gap experiment.",
            "No answer generation, network, HTTP, or process-startup timing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        f"Mean {summary['mean_ms']:.2f} ms | "
        f"median {summary['median_ms']:.2f} ms | p95 {summary['p95_ms']:.2f} ms"
    )
    metrics = quality["hybrid_reranked"]
    print(f"Hit@5 {metrics['hit_rate_at_k']:.3f} | MRR@5 {metrics['mrr_at_k']:.3f}")
    print(f"Repeat top-five order agreement: {agreement}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
