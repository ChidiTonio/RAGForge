"""Compare hybrid ordering and cross-encoder ordering on fixed candidates.

Run from the repository root. Model construction may download model files.
This evaluates passage retrieval, not generated answers or factual truth.
Use --evaluation-role heldout for a frozen held-out evaluation.
The embedding model revision is pinned; the reranker revision is not.
"""

import argparse
import json
import math
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from evaluate_retrieval import TOP_K, evaluate, load_inputs

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.retrieval.cross_encoder import CrossEncoderScorer
from ragforge.retrieval.hybrid import HybridRetriever
from ragforge.retrieval.reranking import Reranker

BACKEND_CANDIDATE_K = 20
FINAL_POOL_K = 20
RRF_K = 60
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 8
MAX_LENGTH = 512


class FrozenRetriever:
    """Expose previously computed rankings through the evaluator interface."""

    def __init__(self, rankings):
        self._rankings = rankings

    def search(self, query, *, top_k=5):
        return list(self._rankings[query][:top_k])


def latency_summary(values):
    """Summarize milliseconds from one measured pass over the questions."""
    return {
        "total_ms": sum(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def candidate_ranking(pool, chunk_to_id, relevant_ids):
    """Validate and serialize the complete frozen candidate pool."""
    seen = set()
    ranking = []
    for rank, result in enumerate(pool, start=1):
        if result.chunk not in chunk_to_id:
            raise ValueError("Candidate chunk is outside the corpus.")
        chunk_id = chunk_to_id[result.chunk]
        if chunk_id in seen:
            raise ValueError("Candidate pool contains duplicate chunks.")
        seen.add(chunk_id)
        score = float(result.score)
        if not math.isfinite(score):
            raise ValueError("Candidate score must be finite.")
        ranking.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "score": score,
            "is_relevant": chunk_id in relevant_ids,
        })
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("evaluation/equity_research/corpus.json"),
    )
    parser.add_argument(
        "--questions", type=Path,
        default=Path("evaluation/equity_research/questions_expanded.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/equity_research/reranking_experiment_v1.json"),
    )
    parser.add_argument(
        "--evaluation-role",
        choices=("development", "heldout"),
        default="development",
        help=(
            "Report metadata only; does not change retrieval or scoring. "
            "Use heldout for a frozen held-out dataset."
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists. Choose a new --output path.")
    if args.output.resolve() in {
        args.corpus.resolve(), args.questions.resolve()
    }:
        parser.error("Output must not overwrite an input file.")

    try:
        (
            manifest, dataset, chunks, chunk_to_id,
            corpus_hash, questions_hash,
        ) = load_inputs(args.corpus, args.questions)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    questions = dataset["questions"]
    query_texts = [question["question"] for question in questions]
    if len(set(query_texts)) != len(query_texts):
        parser.error("This experiment requires unique question texts.")

    evaluation_scope = (
        "held-out passage retrieval evaluation; not answer evaluation"
        if args.evaluation_role == "heldout"
        else "development diagnostic; not held-out"
    )

    print(
        f"Validated: {len(chunks)} chunks | {len(questions)} questions",
        flush=True,
    )
    print(f"Evaluation scope: {evaluation_scope}", flush=True)
    print("Loading models and building the hybrid index on CPU...", flush=True)
    started = perf_counter()
    embedder = SentenceTransformerEmbedder(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        device="cpu",
        batch_size=BATCH_SIZE,
    )
    embedder_load_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    hybrid = HybridRetriever(
        chunks, embedder=embedder,
        candidate_k=BACKEND_CANDIDATE_K, rrf_k=RRF_K,
    )
    index_build_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    scorer = CrossEncoderScorer(
        RERANKER_MODEL, device="cpu",
        batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )
    reranker = Reranker(scorer=scorer)
    reranker_load_ms = (perf_counter() - started) * 1000

    print("Running one unmeasured warm-up query...", flush=True)
    warm_pool = hybrid.search(query_texts[0], top_k=FINAL_POOL_K)
    reranker.rerank(query_texts[0], warm_pool, top_k=TOP_K)

    pools = {}
    retrieval_ms = {}
    print("Freezing all hybrid candidate pools...", flush=True)
    for question in questions:
        query = question["question"]
        started = perf_counter()
        pool = tuple(hybrid.search(query, top_k=FINAL_POOL_K))
        retrieval_ms[question["id"]] = (perf_counter() - started) * 1000
        if len(pool) > FINAL_POOL_K:
            raise ValueError("Hybrid returned more than the requested pool size.")
        candidate_ranking(pool, chunk_to_id, set(question["relevant_chunk_ids"]))
        pools[query] = pool

    reranked = {}
    reranking_ms = {}
    print("Reranking the frozen pools...", flush=True)
    for question in questions:
        query = question["question"]
        started = perf_counter()
        # Keep the complete reordered pool for diagnostics, then evaluate top five.
        results = reranker.rerank(query, pools[query], top_k=FINAL_POOL_K)
        reranking_ms[question["id"]] = (perf_counter() - started) * 1000
        if (
            len(results) != len(pools[query])
            or {result.chunk for result in results}
            != {result.chunk for result in pools[query]}
        ):
            raise ValueError("Reranking changed candidate membership.")
        reranked[query] = results

    methods = {
        "hybrid": evaluate(FrozenRetriever(pools), questions, chunk_to_id),
        "hybrid_reranked": evaluate(
            FrozenRetriever(reranked), questions, chunk_to_id
        ),
    }
    paired_rows = []
    for question, original, reordered in zip(
        questions,
        methods["hybrid"]["per_question"],
        methods["hybrid_reranked"]["per_question"],
        strict=True,
    ):
        query = question["question"]
        relevant_ids = set(question["relevant_chunk_ids"])
        ranking = candidate_ranking(pools[query], chunk_to_id, relevant_ids)
        relevant_count = sum(item["is_relevant"] for item in ranking)
        question_id = question["id"]
        paired_rows.append({
            "question_id": question_id,
            "candidate_count": len(ranking),
            "candidate_pool_hit": int(relevant_count > 0),
            "relevant_labels_in_pool": relevant_count,
            "relevant_label_coverage": relevant_count / len(relevant_ids),
            "hybrid_candidates": ranking,
            "reranked_candidates": candidate_ranking(
                reranked[query], chunk_to_id, relevant_ids
            ),
            "hit_at_k_delta": reordered["hit_at_k"] - original["hit_at_k"],
            "reciprocal_rank_at_k_delta": (
                reordered["reciprocal_rank_at_k"]
                - original["reciprocal_rank_at_k"]
            ),
            "retrieval_ms": retrieval_ms[question_id],
            "reranking_ms": reranking_ms[question_id],
        })

    report = {
        "report_schema_version": 1,
        "experiment": "fixed_hybrid_candidates_cross_encoder",
        "evaluation_scope": evaluation_scope,
        "evaluation_role": args.evaluation_role,
        "dataset_evaluation_role": dataset.get("evaluation_role"),
        "label_status": dataset.get("label_status"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.get("dataset_id"),
        "corpus_path": str(args.corpus),
        "corpus_sha256": corpus_hash,
        "questions_path": str(args.questions),
        "questions_sha256": questions_hash,
        "pdf_filename": manifest["pdf_filename"],
        "pdf_sha256": manifest["pdf_sha256"],
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "chunking": {
            "chunk_size": manifest["chunk_size"],
            "overlap": manifest["overlap"],
        },
        "settings": {
            "top_k": TOP_K,
            "backend_candidate_k": BACKEND_CANDIDATE_K,
            "final_pool_k": FINAL_POOL_K,
            "rrf_k": RRF_K,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "embedding_revision_pinned": True,
            "reranker_model": RERANKER_MODEL,
            "reranker_revision": None,
            "reranker_revision_pinned": False,
            "device": "cpu",
            "embedding_batch_size": BATCH_SIZE,
            "reranking_batch_size": BATCH_SIZE,
            "reranker_max_length": MAX_LENGTH,
            "reranker_activation": "identity",
            "warmup_queries": 1,
            "timing_passes": 1,
            "model_revision_pinned": False,
        },
        "package_versions": {
            name: version(name)
            for name in ("sentence-transformers", "transformers", "torch", "numpy")
        },
        "setup_latency": {
            "embedder_load_ms": embedder_load_ms,
            "hybrid_index_build_ms": index_build_ms,
            "reranker_load_ms": reranker_load_ms,
        },
        "latency": {
            "note": "Single warmed-up pass; descriptive, not a rigorous benchmark.",
            "retrieval": latency_summary(list(retrieval_ms.values())),
            "reranking": latency_summary(list(reranking_ms.values())),
            "combined": latency_summary([
                retrieval_ms[q["id"]] + reranking_ms[q["id"]] for q in questions
            ]),
        },
        "candidate_pool": {
            "hit_rate": sum(r["candidate_pool_hit"] for r in paired_rows)
            / len(paired_rows),
            "mean_relevant_label_coverage": sum(
                r["relevant_label_coverage"] for r in paired_rows
            ) / len(paired_rows),
        },
        "methods": methods,
        "paired_diagnostics": paired_rows,
    }

    print(f"\n{'Method':<20} {'Hit Rate@5':>12} {'MRR@5':>12}")
    for name, metrics in methods.items():
        print(
            f"{name:<20} {metrics['hit_rate_at_k']:>12.3f}"
            f" {metrics['mrr_at_k']:>12.3f}"
        )
    print(f"Candidate-pool hit rate: {report['candidate_pool']['hit_rate']:.3f}")
    print(
        "Mean added reranking latency: "
        f"{report['latency']['reranking']['mean_ms']:.1f} ms/query"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
    print(f"Saved report: {args.output}")
    print(
        "Passage retrieval evaluation only; embedding revision is pinned, "
        "reranker revision is not pinned."
    )


if __name__ == "__main__":
    main()