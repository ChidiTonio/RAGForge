"""Conservative single-page abstract routing with exact chunk offsets."""

import re
from dataclasses import replace

from ragforge.retrieval.types import SearchResult

_START = re.compile(
    r"^[ \t]*(?:\d+[.)]?[ \t]+)?abstract[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_END = re.compile(
    r"^[ \t]*(?:key[ \t-]*words\b|"
    r"(?:\d+[.)]?[ \t]+)?introduction\b)",
    re.IGNORECASE | re.MULTILINE,
)


def _abstract_chunks(chunks):
    """Identify one bounded abstract on one of the first three pages."""
    chunks = list(chunks)
    by_page = {}
    for chunk in chunks:
        if 1 <= chunk.page_number <= 3:
            by_page.setdefault(
                (chunk.source, chunk.page_number), []
            ).append(chunk)

    sections = []
    for batch in by_page.values():
        chars = [None] * max(chunk.end_char for chunk in batch)
        for chunk in batch:
            if len(chunk.text) != chunk.end_char - chunk.start_char:
                raise ValueError("Unexpected chunk offsets.")
            for offset, character in enumerate(
                chunk.text, start=chunk.start_char
            ):
                if chars[offset] not in (None, character):
                    raise ValueError("Overlapping chunk text disagrees.")
                chars[offset] = character

        text = "".join(char if char is not None else " " for char in chars)
        for start in _START.finditer(text):
            end = _END.search(text, start.end())
            if end is None:
                continue

            left, right = start.end(), end.start()
            # Do not infer a section across missing reconstructed text.
            if any(char is None for char in chars[left:right]):
                continue

            selected = []
            for chunk in sorted(batch, key=lambda item: item.start_char):
                lo = max(left, chunk.start_char)
                hi = min(right, chunk.end_char)
                if lo >= hi:
                    continue
                passage = chunk.text[
                    lo - chunk.start_char:hi - chunk.start_char
                ]
                if re.search(r"\w", passage):
                    selected.append(
                        replace(
                            chunk,
                            text=passage,
                            start_char=lo,
                            end_char=hi,
                        )
                    )
            if selected:
                sections.append(selected)

    # Ambiguous or unbounded detection must not trigger whole-PDF search.
    return sections[0] if len(sections) == 1 else []


class AbstractSectionRetriever:
    """Route explicit abstract queries; otherwise use normal retrieval."""

    def __init__(self, fallback, reranker, chunks, *, report=None):
        self._fallback = fallback
        self._reranker = reranker
        self._abstract = _abstract_chunks(chunks)
        self._report = report if report is not None else lambda message: None
        self._report(
            f"Abstract scope: {len(self._abstract)} usable chunks "
            "(conservative single-page detection)."
        )

    def search(self, query: str, *, top_k: int = 5):
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            return []

        if re.search(
            r"\bsections?\s+\d+(?:\.\d+)*\b", query, re.IGNORECASE
        ):
            self._report(
                "Numbered-section scope unavailable; "
                "no unrelated evidence supplied."
            )
            return []

        if not re.search(r"\babstract\b", query, re.IGNORECASE):
            return self._fallback.search(query, top_k=top_k)
        if not self._abstract:
            self._report(
                "Abstract scope unavailable; no unrelated evidence supplied."
            )
            return []
        self._report("Retrieval scope: detected abstract")
        candidates = [
            SearchResult(chunk=chunk, score=0.0)
            for chunk in self._abstract
        ]
        return self._reranker.rerank(query, candidates, top_k=top_k)
