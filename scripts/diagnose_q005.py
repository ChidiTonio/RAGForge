"""Inspect q005 rankings without changing the fixed retrieval baseline."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from evaluate_retrieval import (
    CANDIDATE_K,
    RRF_K,
    TOP_K,
    HybridRetriever,
    SentenceTransformerEmbedder,
    load_inputs,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main() -> None:
    corpus_path = Path("evaluation/equity_research/corpus.json")
    questions_path = Path("evaluation/equity_research/questions.json")
    output = Path("evaluation/equity_research/q005_diagnostic.json")

    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    (
        _, dataset, chunks, chunk_to_id,
        corpus_hash, questions_hash,
    ) = load_inputs(corpus_path, questions_path)

    question = next(
        (item for item in dataset["questions"] if item["id"] == "q005"),
        None,
    )
    if question is None:
        raise SystemExit("Question q005 was not found.")

    relevant_ids = set(question["relevant_chunk_ids"])
    query = question["question"]

    print(f"Query: {query}", flush=True)
    print("Building the baseline-configured hybrid index...", flush=True)
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
        batch_size=8,
    )
    hybrid = HybridRetriever(
        chunks,
        embedder=embedder,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
    )

    def serialize(results):
        rows = []
        seen = set()
        for rank, result in enumerate(results, start=1):
            if result.chunk not in chunk_to_id:
                raise ValueError("Result outside the fixed corpus.")
            chunk_id = chunk_to_id[result.chunk]
            score = float(result.score)
            if chunk_id in seen or not math.isfinite(score):
                raise ValueError("Duplicate result or nonfinite score.")
            seen.add(chunk_id)
            rows.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "score": score,
                "is_relevant": chunk_id in relevant_ids,
            })
        return rows

    # Read-only diagnostic access to the exact backends used by hybrid.
    bm25 = serialize(hybrid._bm25.search(query, top_k=CANDIDATE_K))
    dense = serialize(hybrid._dense.search(query, top_k=CANDIDATE_K))

    # Candidate depth remains fixed even when output depth is increased.
    fused = serialize(hybrid.search(query, top_k=len(chunks)))
    baseline_top = serialize(hybrid.search(query, top_k=TOP_K))

    if fused[:TOP_K] != baseline_top:
        raise ValueError("Expanded output does not match the top-five prefix.")

    ranks = {
        "bm25": {row["chunk_id"]: row["rank"] for row in bm25},
        "dense": {row["chunk_id"]: row["rank"] for row in dense},
    }
    candidate_ids = set(ranks["bm25"]) | set(ranks["dense"])
    if {row["chunk_id"] for row in fused} != candidate_ids:
        raise ValueError("Fused output does not match the candidate union.")

    for row in fused:
        chunk_id = row["chunk_id"]
        for method in ("bm25", "dense"):
            rank = ranks[method].get(chunk_id)
            row[f"{method}_candidate_rank"] = rank
            row[f"{method}_rrf_contribution"] = (
                1.0 / (RRF_K + rank) if rank is not None else 0.0
            )
        reconstructed = (
            row["bm25_rrf_contribution"] + row["dense_rrf_contribution"]
        )
        if not math.isclose(
            row["score"], reconstructed, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"RRF reconstruction mismatch: {chunk_id}")

    fused_ranks = {row["chunk_id"]: row["rank"] for row in fused}
    relevant_positions = {
        chunk_id: {
            "bm25_candidate_rank": ranks["bm25"].get(chunk_id),
            "dense_candidate_rank": ranks["dense"].get(chunk_id),
            "hybrid_rank": fused_ranks.get(chunk_id),
        }
        for chunk_id in sorted(relevant_ids)
    }

    print("\nRank  Chunk       BM25  Dense  BM25 RRF   Dense RRF  Total RRF")
    for row in fused:
        marker = "*" if row["is_relevant"] else " "
        bm25_rank = row["bm25_candidate_rank"]
        dense_rank = row["dense_candidate_rank"]
        print(
            f"{row['rank']:>4}  {row['chunk_id']}{marker} "
            f"{str(bm25_rank) if bm25_rank is not None else '-':>5} "
            f"{str(dense_rank) if dense_rank is not None else '-':>6} "
            f"{row['bm25_rrf_contribution']:>10.6f} "
            f"{row['dense_rrf_contribution']:>10.6f} "
            f"{row['score']:>10.6f}"
        )

    report = {
        "diagnostic_schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "corpus_sha256": corpus_hash,
        "questions_sha256": questions_hash,
        "question": question,
        "settings": {
            "baseline_top_k": TOP_K,
            "diagnostic_output_top_k": len(chunks),
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "device": "cpu",
            "batch_size": 8,
            "model_selection": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        "top_five_prefix_verified": True,
        "relevant_positions": relevant_positions,
        "bm25_candidates": bm25,
        "dense_candidates": dense,
        "hybrid_ranking": fused,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )

    print("\nRelevant chunk positions:")
    print(json.dumps(relevant_positions, indent=2))
    print(f"\nSaved: {output}")
    print("* marks labeled evidence; '-' means absent from that candidate list.")


if __name__ == "__main__":
    main()