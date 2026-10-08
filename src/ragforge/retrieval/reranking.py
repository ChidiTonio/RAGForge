"""Rerank a fixed retrieval candidate list using an injectable scorer."""

import math
from collections.abc import Iterable, Sequence
from numbers import Real
from typing import Protocol

from ragforge.retrieval.types import SearchResult


class RelevanceScorer(Protocol):
    """Score query-passage pairs in passage order.

    Return one finite real score per passage. Higher means more relevant.
    Scores are ranking values, not necessarily probabilities.
    """

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        """Return one relevance score for each passage."""
        ...


class Reranker:
    """Reorder existing candidates without retrieving additional chunks.

    Equal relevance scores retain the original candidate order.
    Backend exceptions propagate unchanged.
    """

    def __init__(self, *, scorer: RelevanceScorer) -> None:
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Iterable[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return up to top_k candidates with new relevance scores.

        Blank queries and empty candidate lists return no results without
        invoking the scorer. Candidate inputs are never modified.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            return []

        batch = list(candidates)
        if any(not isinstance(result, SearchResult) for result in batch):
            raise TypeError("Every candidate must be a SearchResult.")
        if not batch:
            return []

        passages = [result.chunk.text for result in batch]
        raw_scores = list(self._scorer.score(query, passages))
        if len(raw_scores) != len(batch):
            raise ValueError("Score count must match candidate count.")

        scores: list[float] = []
        for value in raw_scores:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("Every relevance score must be a real number.")
            try:
                score = float(value)
            except OverflowError as exc:
                raise ValueError("Every relevance score must be finite.") from exc
            if not math.isfinite(score):
                raise ValueError("Every relevance score must be finite.")
            scores.append(score)

        ranked_indices = sorted(
            range(len(batch)),
            key=lambda index: (-scores[index], index),
        )
        return [
            SearchResult(chunk=batch[index].chunk, score=scores[index])
            for index in ranked_indices[:top_k]
        ]
