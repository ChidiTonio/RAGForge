"""Combine sparse and dense rankings using reciprocal rank fusion."""

from collections.abc import Iterable

from ragforge.embeddings.base import Embedder
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.bm25 import BM25Retriever
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.types import SearchResult


def _validate_positive_integer(value: int, *, name: str) -> None:
    """Reject booleans, non-integers, and nonpositive integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


class HybridRetriever:
    """Fuse BM25 and dense ranks over the same fixed corpus.

    Chunks must be unique by TextChunk value equality and contain at
    least one word token, as required by the BM25 index.

    Each backend contributes at most candidate_k results. Candidate
    depth is fixed at construction and does not grow with search top_k.
    Fusion uses one-based ranks, not the backends' raw scores.

    Input corpus order breaks fused-score ties. Scores are positive
    ranking values, not probabilities or cosine similarities.
    """

    def __init__(
        self,
        chunks: Iterable[TextChunk],
        *,
        embedder: Embedder,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        _validate_positive_integer(candidate_k, name="candidate_k")
        _validate_positive_integer(rrf_k, name="rrf_k")

        self._chunks = tuple(chunks)
        if not self._chunks:
            raise ValueError("At least one chunk is required.")

        if any(not isinstance(chunk, TextChunk) for chunk in self._chunks):
            raise TypeError("Every chunk must be a TextChunk.")

        self._positions = {
            chunk: position for position, chunk in enumerate(self._chunks)
        }
        if len(self._positions) != len(self._chunks):
            raise ValueError("Duplicate chunks are not allowed.")

        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

        # Validate the lexical corpus before invoking document embedding.
        self._bm25 = BM25Retriever(self._chunks)
        self._dense = DenseRetriever(self._chunks, embedder=embedder)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k candidates ordered by descending RRF score.

        Blank queries return no results without searching either backend.
        Candidates appearing in only one ranking remain eligible.
        Backend exceptions propagate unchanged.
        """
        _validate_positive_integer(top_k, name="top_k")
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            return []

        sparse_results = self._bm25.search(
            query, top_k=self._candidate_k
        )
        dense_results = self._dense.search(
            query, top_k=self._candidate_k
        )

        scores: dict[TextChunk, float] = {}
        for ranking in (sparse_results, dense_results):
            for rank, result in enumerate(ranking, start=1):
                contribution = 1.0 / (self._rrf_k + rank)
                scores[result.chunk] = (
                    scores.get(result.chunk, 0.0) + contribution
                )

        ranked_chunks = sorted(
            scores,
            key=lambda chunk: (-scores[chunk], self._positions[chunk]),
        )

        return [
            SearchResult(chunk=chunk, score=scores[chunk])
            for chunk in ranked_chunks[:top_k]
        ]