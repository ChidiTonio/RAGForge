"""Manually check PDF ingestion and real-model hybrid retrieval on CPU.

Run from the repository root:
    python scripts/smoke_hybrid.py data/example.pdf --query "Your question"

The first run may download model files. This is not part of pytest.
RRF scores are ranking values, not probabilities or cosine similarities.
"""

import argparse
import math
from pathlib import Path

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.ingestion.chunking import chunk_pages
from ragforge.ingestion.pdf import parse_pdf
from ragforge.retrieval.hybrid import HybridRetriever


def positive_integer(value: str) -> int:
    """Validate positive integer command-line arguments."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be an integer.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF.")
    parser.add_argument("--query", required=True, help="Research question.")
    parser.add_argument("--top-k", type=positive_integer, default=5)
    parser.add_argument("--candidate-k", type=positive_integer, default=20)
    parser.add_argument("--rrf-k", type=positive_integer, default=60)
    parser.add_argument("--chunk-size", type=positive_integer, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF file not found: {args.pdf}")
    if not args.query.strip():
        parser.error("--query must contain non-whitespace text.")
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("--overlap must satisfy 0 <= overlap < --chunk-size.")

    print(f"Reading PDF: {args.pdf}", flush=True)
    pages = parse_pdf(args.pdf)
    chunks = chunk_pages(
        pages,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    text_page_count = sum(page.has_text for page in pages)
    print(
        f"Pages: {len(pages)} | Pages with text: {text_page_count} "
        f"| Chunks: {len(chunks)}"
    )

    if not chunks:
        parser.error(
            "No text chunks were extracted. Scanned PDFs may require OCR; "
            "this ingestion pipeline does not perform OCR."
        )

    print("Loading all-MiniLM-L6-v2 on CPU...", flush=True)
    embedder = SentenceTransformerEmbedder(device="cpu", batch_size=8)

    print("Building BM25 and dense indexes...", flush=True)
    retriever = HybridRetriever(
        chunks,
        embedder=embedder,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )

    print(f"\nQuery: {args.query}", flush=True)
    results = retriever.search(args.query, top_k=args.top_k)

    # These are integration checks, not relevance assertions.
    assert results, "Expected results for a nonblank query."
    assert len(results) <= min(args.top_k, len(chunks))

    corpus = set(chunks)
    result_chunks = [result.chunk for result in results]
    assert len(set(result_chunks)) == len(result_chunks)
    assert all(chunk in corpus for chunk in result_chunks)

    scores = [result.score for result in results]
    assert all(math.isfinite(score) and score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)

    # Two backends can each contribute at most their rank-one RRF value.
    maximum_score = 2.0 / (args.rrf_k + 1)
    assert all(score <= maximum_score for score in scores)
    assert retriever.search(" \n\t ", top_k=args.top_k) == []

    print(
        f"Returned: {len(results)} | candidate_k={args.candidate_k} "
        f"| rrf_k={args.rrf_k}"
    )
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = " ".join(chunk.text.split())
        if len(preview) > 240:
            preview = preview[:240] + "..."

        print(
            f"\n{rank}. rrf_score={result.score:.6f} "
            f"| source={chunk.source} "
            f"| page={chunk.page_number} "
            f"| chunk={chunk.chunk_number} "
            f"| chars=[{chunk.start_char}, {chunk.end_char})"
        )
        print(f"   {preview}")

    print("\nPASS: PDF-to-hybrid retrieval integration checks.")
    print("Inspect passage relevance manually; this is not a quality evaluation.")


if __name__ == "__main__":
    main()