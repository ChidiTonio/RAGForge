"""Evaluate BM25, dense, and hybrid retrieval on a fixed labeled corpus.

Run from the repository root:
    python scripts/evaluate_retrieval.py

This evaluates passage retrieval, not generated answers or factual truth.
The first run may download embedding model files.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.bm25 import BM25Retriever
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.hybrid import HybridRetriever

TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def collect_run_metadata():
    """Capture code/runtime identity without storing private status paths."""
    script_path = Path(__file__).resolve()
    provenance = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "evaluator_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "git_commit": None,
        "git_working_tree_dirty": None,
        "git_metadata_status": "unavailable",
        "git_metadata_error": None,
        "python_version": platform.python_version(),
    }

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=script_path.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError) as exc:
        # Avoid storing stderr or potentially private paths.
        provenance["git_metadata_error"] = type(exc).__name__
    else:
        provenance.update({
            "git_commit": commit,
            "git_working_tree_dirty": bool(status),
            "git_metadata_status": "available",
        })

    package_versions = {}
    for name in ("sentence-transformers", "transformers", "torch", "numpy"):
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = None

    return provenance, package_versions


def load_json(path: Path) -> tuple[dict, str]:
    """Load a JSON object and hash the exact input file bytes."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level JSON must be an object.")
    return data, hashlib.sha256(raw).hexdigest()


def load_inputs(corpus_path: Path, questions_path: Path):
    """Validate the fixed corpus and labels without invoking embeddings."""
    manifest, corpus_hash = load_json(corpus_path)
    dataset, questions_hash = load_json(questions_path)

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported corpus schema_version.")

    records = manifest.get("chunks")
    if not isinstance(records, list) or not records:
        raise ValueError("Corpus must contain a nonempty chunks list.")
    if manifest.get("chunk_count") != len(records):
        raise ValueError("Manifest chunk_count does not match chunks.")

    chunk_fields = {field.name for field in fields(TextChunk)}
    chunks = []
    chunk_to_id = {}
    known_ids = set()

    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Chunk record {position} must be an object.")
        if set(record) != chunk_fields | {"id"}:
            raise ValueError(f"Chunk record {position} has unexpected fields.")

        chunk_id = record["id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk record {position} has an invalid ID.")
        if chunk_id in known_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        values = {name: record[name] for name in chunk_fields}
        for name in ("source", "text"):
            if not isinstance(values[name], str) or not values[name].strip():
                raise ValueError(f"{chunk_id}: {name} must be nonblank text.")

        for name in ("page_number", "chunk_number", "start_char", "end_char"):
            if type(values[name]) is not int:
                raise ValueError(f"{chunk_id}: {name} must be an integer.")

        if values["page_number"] <= 0 or values["chunk_number"] <= 0:
            raise ValueError(f"{chunk_id}: page and chunk numbers must be positive.")
        if not 0 <= values["start_char"] < values["end_char"]:
            raise ValueError(f"{chunk_id}: invalid character offsets.")
        if len(values["text"]) != values["end_char"] - values["start_char"]:
            raise ValueError(f"{chunk_id}: text length does not match offsets.")

        expected_id = (
            f"p{values['page_number']:03d}-c{values['chunk_number']:03d}"
        )
        if chunk_id != expected_id:
            raise ValueError(f"{chunk_id}: ID does not match chunk metadata.")

        chunk = TextChunk(**values)
        if chunk in chunk_to_id:
            raise ValueError(f"{chunk_id}: duplicate TextChunk value.")

        chunks.append(chunk)
        chunk_to_id[chunk] = chunk_id
        known_ids.add(chunk_id)

    if dataset.get("source") != manifest.get("pdf_filename"):
        raise ValueError("Dataset source does not match corpus pdf_filename.")

    questions = dataset.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must contain a nonempty questions list.")

    question_ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise TypeError("Each question must be an object.")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Each question must have a nonblank string ID.")
        if question_id in question_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")
        question_ids.add(question_id)

        text = question.get("question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{question_id}: question must be nonblank text.")

        relevant_ids = question.get("relevant_chunk_ids")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(item, str) for item in relevant_ids)
        ):
            raise ValueError(f"{question_id}: invalid relevant_chunk_ids.")
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(f"{question_id}: duplicate relevance labels.")

        missing = set(relevant_ids) - known_ids
        if missing:
            raise ValueError(
                f"{question_id}: unknown relevant IDs: {sorted(missing)}"
            )

    return (
        manifest, dataset, chunks, chunk_to_id, corpus_hash, questions_hash
    )


def evaluate(retriever, questions, chunk_to_id):
    """Compute binary hit rate and first-relevant reciprocal rank at TOP_K."""
    rows = []
    for question in questions:
        results = retriever.search(question["question"], top_k=TOP_K)
        if len(results) > TOP_K:
            raise ValueError("Retriever returned more than top_k results.")

        relevant_ids = set(question["relevant_chunk_ids"])
        retrieved_ids = set()
        ranking = []
        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if result.chunk not in chunk_to_id:
                raise ValueError("Retriever returned a chunk outside the corpus.")
            chunk_id = chunk_to_id[result.chunk]
            if chunk_id in retrieved_ids:
                raise ValueError("Retriever returned duplicate chunks.")
            retrieved_ids.add(chunk_id)

            score = float(result.score)
            if not math.isfinite(score):
                raise ValueError("Retriever returned a nonfinite score.")

            is_relevant = chunk_id in relevant_ids
            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = rank

            ranking.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "score": score,
                "is_relevant": is_relevant,
            })

        rows.append({
            "question_id": question["id"],
            "question": question["question"],
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "hit_at_k": int(first_relevant_rank is not None),
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank_at_k": (
                1.0 / first_relevant_rank
                if first_relevant_rank is not None else 0.0
            ),
            "ranking": ranking,
        })

    return {
        "hit_rate_at_k": sum(row["hit_at_k"] for row in rows) / len(rows),
        "mrr_at_k": (
            sum(row["reciprocal_rank_at_k"] for row in rows) / len(rows)
        ),
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("evaluation/equity_research/corpus.json"),
    )
    parser.add_argument(
        "--questions", type=Path,
        default=Path("evaluation/equity_research/questions.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/equity_research/retrieval_baseline.json"),
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists. Choose a new --output path.")

    try:
        (
            manifest, dataset, chunks, chunk_to_id,
            corpus_hash, questions_hash,
        ) = load_inputs(args.corpus, args.questions)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    provenance, package_versions = collect_run_metadata()
    questions = dataset["questions"]
    print(
        f"Validated: {len(chunks)} chunks | {len(questions)} questions",
        flush=True,
    )
    print("Loading pinned all-MiniLM-L6-v2 on CPU...", flush=True)
    embedder = SentenceTransformerEmbedder(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
        batch_size=8,
    )

    # Use public constructors, including the actual hybrid implementation.
    # Dense and hybrid each build their own dense index.
    print("Building BM25, dense, and hybrid indexes...", flush=True)
    retrievers = {
        "bm25": BM25Retriever(chunks),
        "dense": DenseRetriever(chunks, embedder=embedder),
        "hybrid": HybridRetriever(
            chunks,
            embedder=embedder,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
        ),
    }

    report = {
        "report_schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "package_versions": package_versions,
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
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "embedder_class": "SentenceTransformerEmbedder",
            "model_selection": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": "cpu",
            "batch_size": 8,
        },
        "methods": {},
    }

    print(f"\n{'Method':<12} {'Hit Rate@5':>12} {'MRR@5':>12}")
    for name, retriever in retrievers.items():
        metrics = evaluate(retriever, questions, chunk_to_id)
        report["methods"][name] = metrics
        print(
            f"{name:<12} {metrics['hit_rate_at_k']:>12.3f}"
            f" {metrics['mrr_at_k']:>12.3f}"
        )
        for row in metrics["per_question"]:
            ranked_ids = ", ".join(
                f"{item['chunk_id']}{'*' if item['is_relevant'] else ''}"
                for item in row["ranking"]
            )
            print(f"  {row['question_id']}: {ranked_ids or '(no results)'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )

    print(f"\nSaved report: {args.output}")
    print("* marks a labeled relevant chunk.")
    print("This is a small development baseline, not a general benchmark.")


if __name__ == "__main__":
    main()