"""Manually check real-model embeddings and dense retrieval on CPU.

Run from the repository root:
    python scripts/smoke_dense.py

The first run may download model files. This is not part of pytest.
"""

import numpy as np

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.dense import DenseRetriever


def main() -> None:
    texts = [
        "Solar panels convert sunlight into electrical energy.",
        "Bread dough rises when yeast produces carbon dioxide.",
        "Penguins are flightless birds adapted to swimming.",
    ]
    chunks = [
        TextChunk(
            source="smoke-test",
            page_number=index + 1,
            chunk_number=1,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for index, text in enumerate(texts)
    ]

    print("Loading all-MiniLM-L6-v2 on CPU...", flush=True)
    embedder = SentenceTransformerEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        device="cpu",
        batch_size=8,
    )

    vectors = embedder.embed(texts)
    assert embedder.dimension == 384, "Unexpected dimension for this model."
    assert vectors.shape == (len(texts), embedder.dimension)
    assert vectors.dtype == np.dtype(np.float32)
    assert np.isfinite(vectors).all()
    assert np.all(np.any(vectors != 0, axis=1))

    empty_vectors = embedder.embed([])
    assert empty_vectors.shape == (0, embedder.dimension)
    assert empty_vectors.dtype == np.dtype(np.float32)

    print(f"Embeddings: shape={vectors.shape}, dtype={vectors.dtype}")

    # Re-encode the small corpus through the actual retriever construction path.
    retriever = DenseRetriever(chunks, embedder=embedder)
    query = "How can sunlight be used to generate electricity?"
    results = retriever.search(query, top_k=3)

    assert len(results) == len(chunks)
    assert all(-1.0 <= result.score <= 1.0 for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results),
        reverse=True,
    )
    assert all(any(result.chunk is chunk for chunk in chunks) for result in results)

    print(f"\nQuery: {query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result.score:.4f} "
            f"page={result.chunk.page_number} | {result.chunk.text}"
        )

    print("\nPASS: real-model embedding and retrieval checks.")
    print("Inspect the ranking above; this smoke test is not a quality evaluation.")


if __name__ == "__main__":
    main()