"""Split extracted PDF pages into overlapping character-based chunks."""

from collections.abc import Iterable
from dataclasses import dataclass

from ragforge.ingestion.pdf import PDFPage


@dataclass(frozen=True)
class TextChunk:
    """A text passage with metadata linking it to its source page."""

    source: str
    page_number: int
    chunk_number: int
    start_char: int
    end_char: int
    text: str


def chunk_pages(
    pages: Iterable[PDFPage],
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Create chunks without crossing page boundaries.

    Sizes and offsets are measured in Python string characters, not tokens.
    Character offsets reference the original PDFPage.text; end_char is
    exclusive. Chunk numbers are one-based and restart on each page.
    Blank pages and whitespace-only chunks are omitted.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap

    for page in pages:
        if not page.has_text:
            continue

        start = 0
        chunk_number = 0

        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            text = page.text[start:end]

            if text.strip():
                chunk_number += 1
                chunks.append(
                    TextChunk(
                        source=page.source,
                        page_number=page.page_number,
                        chunk_number=chunk_number,
                        start_char=start,
                        end_char=end,
                        text=text,
                    )
                )

            if end == len(page.text):
                break

            start += step

    return chunks