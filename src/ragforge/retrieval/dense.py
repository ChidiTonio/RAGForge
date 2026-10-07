"""Exact cosine-similarity retrieval over source-linked text chunks."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ragforge.embeddings.base import Embedder
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.types import SearchResult


def _normalize_embeddings(
    vectors: NDArray[np.float32],
    *,
    expected_rows: int,
    expected_dimension: int | None = None,
) -> NDArray[np.float64]:
    """Validate embedding output and return independently owned unit vectors."""
    if not isinstance(vectors, np.ndarray):
        raise TypeError("Embeddings must be a NumPy array.")



    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if vectors.shape[0] != expected_rows:
        raise ValueError("Embedding row count must match the number of texts.")

    if vectors.shape[1] == 0:
        raise ValueError("Embedding dimension must be greater than zero.")

    if expected_dimension is not None and vectors.shape[1] != expected_dimension:
        raise ValueError("Query embedding dimension must match the index.")

    if vectors.dtype != np.dtype(np.float32):
        raise ValueError("Embeddings must have dtype float32.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain only finite values.")

    # Float64 arithmetic avoids overflow or underflow when computing norms
    # of finite float32 vectors. Copy so we never modify embedder-owned data.
    normalized = vectors.astype(np.float64, copy=True)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Every embedding vector must have a nonzero norm.")

    normalized /= norms
    return normalized


class DenseRetriever:
    """An in-memory index using exact cosine similarity.

    Every nonblank query ranks all indexed chunks. No lexical overlap
    or positive score is required. Input order breaks score ties.

    The embedder must remain configured for the same vector space
    throughout the lifetime of the index.
    """

    def __init__(
        self,
        chunks: Iterable[TextChunk],
        *,
        embedder: Embedder,
    ) -> None:
        self._chunks = tuple(chunks)

        if not self._chunks:
            raise ValueError("At least one chunk is required.")

        if any(not chunk.text.strip() for chunk in self._chunks):
            raise ValueError("Every chunk must contain nonblank text.")

        self._embedder = embedder
        vectors = self._embedder.embed([chunk.text for chunk in self._chunks])
        self._vectors = _normalize_embeddings(
            vectors,
            expected_rows=len(self._chunks),
        )

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k chunks ordered by descending cosine similarity.

        Blank queries return no results without calling the embedder.
        Scores lie between -1 and 1 and are not relevance probabilities.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not query.strip():
            return []

        query_vectors = self._embedder.embed([query])
        normalized_query = _normalize_embeddings(
            query_vectors,
            expected_rows=1,
            expected_dimension=self._vectors.shape[1],
        )

        # Both sides are unit vectors, so their dot product is cosine
        # similarity. Clipping corrects small floating-point overshoots.
        scores = np.clip(self._vectors @ normalized_query[0], -1.0, 1.0)

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), index),
        )

        return [
            SearchResult(chunk=self._chunks[index], score=float(scores[index]))
            for index in ranked_indices[:top_k]
        ]