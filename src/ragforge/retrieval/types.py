"""Shared result types for retrieval implementations."""

from dataclasses import dataclass

from ragforge.ingestion.chunking import TextChunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its raw ranking score, not a probability."""

    chunk: TextChunk
    score: float
