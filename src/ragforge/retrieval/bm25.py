"""BM25 retrieval over source-linked text chunks."""

import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.types import SearchResult


def tokenize(text: str) -> list[str]:
    """Case-fold text and extract Unicode word tokens.

    Punctuation separates tokens; underscores remain within tokens.
    This baseline does not perform stemming or remove stop words.
    """
    return re.findall(r"\w+", text.casefold())





class BM25Retriever:
    """An in-memory BM25 index with deterministic ranking.

    Input order breaks score ties. Every indexed chunk must contain at
    least one word token. Results require at least one query-token match,
    but their BM25 scores need not be positive.
    """

    def __init__(self, chunks: Iterable[TextChunk]) -> None:
        self._chunks = tuple(chunks)
        if not self._chunks:
            raise ValueError("At least one chunk is required.")

        tokenized_chunks = [tokenize(chunk.text) for chunk in self._chunks]
        if any(not tokens for tokens in tokenized_chunks):
            raise ValueError("Every chunk must contain at least one word token.")

        self._token_sets = tuple(frozenset(tokens) for tokens in tokenized_chunks)
        self._index = BM25Okapi(tokenized_chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k matching chunks, ordered by descending score.

        Blank, punctuation-only, and out-of-vocabulary queries return
        no results. Matching chunks are retained even at zero or negative
        scores because BM25 scores depend on corpus statistics.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        candidate_indices = [
            index
            for index, terms in enumerate(self._token_sets)
            if query_terms.intersection(terms)
        ]
        if not candidate_indices:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: (-float(scores[index]), index),
        )

        return [
            SearchResult(chunk=self._chunks[index], score=float(scores[index]))
            for index in ranked_indices[:top_k]
        ]