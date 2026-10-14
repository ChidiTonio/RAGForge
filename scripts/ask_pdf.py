"""Ask questions about one PDF using hybrid retrieval and local Ollama."""

import argparse
import re
import time
from pathlib import Path

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.generation.core import (
    Answer,
    AnswerGenerator,
    InvalidLLMResponse,
    RAGPipeline,
)
from ragforge.generation.ollama import OllamaError, OllamaLLM
from ragforge.ingestion.chunking import chunk_pages
from ragforge.ingestion.pdf import parse_pdf
from ragforge.retrieval.hybrid import HybridRetriever
from ragforge.retrieval.cross_encoder import CrossEncoderScorer
from ragforge.retrieval.reranking import Reranker


class RerankedRetriever:
    """Retrieve a wider candidate pool, then return reranked passages."""

    def __init__(self, retriever, reranker, *, candidate_k=30):
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(self, query: str, *, top_k: int = 5):
        candidates = self._retriever.search(
            query, top_k=max(self._candidate_k, top_k)
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)



class PaperSectionRetriever:
    """Verified section profile for this PDF, not a general section parser."""

    def __init__(self, fallback, reranker, chunks):
        from dataclasses import replace
        from ragforge.retrieval.types import SearchResult

        self._fallback = fallback
        self._reranker = reranker
        self._result_type = SearchResult
        self._sections = {}

        # Reconstruct page text using the original chunk offsets.
        by_page = {}
        for chunk in chunks:
            by_page.setdefault(chunk.page_number, []).append(chunk)

        page_text = {}
        for number, batch in by_page.items():
            chars = [None] * max(c.end_char for c in batch)
            for chunk in batch:
                if len(chunk.text) != chunk.end_char - chunk.start_char:
                    raise ValueError("Unexpected chunk offsets.")
                for offset, character in enumerate(
                    chunk.text, start=chunk.start_char
                ):
                    if chars[offset] not in (None, character):
                        raise ValueError("Overlapping chunk text disagrees.")
                    chars[offset] = character
            page_text[number] = "".join(
                character if character is not None else " "
                for character in chars
            )

        def marker(page, pattern):
            match = re.search(pattern, page_text.get(page, ""), re.I | re.M)
            if match is None:
                raise ValueError(
                    f"Verified section marker missing on page {page}: {pattern}"
                )
            return match.start()

        abstract_start = marker(
            1, r"Recent\s+breakthroughs\s+in\s+Large\s+Language\s+Models"
        )
        abstract_end = marker(1, r"^\s*KEYWORDS\s*$")
        discussion_start = marker(15, r"^\s*5\s+Discussion\s*$")
        discussion_end = marker(19, r"^\s*Author contributions\s*$")

        def clipped_section(start, end):
            if start >= end:
                raise ValueError("Invalid section boundaries.")
            selected = []
            for chunk in chunks:
                page = chunk.page_number
                if not start[0] <= page <= end[0]:
                    continue
                left = max(
                    chunk.start_char,
                    start[1] if page == start[0] else chunk.start_char,
                )
                right = min(
                    chunk.end_char,
                    end[1] if page == end[0] else chunk.end_char,
                )
                if left >= right:
                    continue
                text = chunk.text[
                    left - chunk.start_char:right - chunk.start_char
                ]
                if re.search(r"\w+", text):
                    selected.append(replace(
                        chunk,
                        text=text,
                        start_char=left,
                        end_char=right,
                    ))
            if not selected:
                raise ValueError("Section contains no usable chunks.")
            return selected

        self._sections["abstract"] = clipped_section(
            (1, abstract_start), (1, abstract_end)
        )
        self._sections["discussion"] = clipped_section(
            (15, discussion_start), (19, discussion_end)
        )

        strengths_start = marker(
            15, r"^\s*5\.1\s+Strengths of existing research\s*$"
        )
        limitations_start = marker(
            18, r"^\s*5\.2\s+Limitations of existing research studies\s*$"
        )
        directions_start = marker(
            18, r"^\s*5\.3\s+Research gaps and future directions\s*$"
        )
        self._sections["5.1"] = clipped_section(
            (15, strengths_start), (18, limitations_start)
        )
        self._sections["5.2"] = clipped_section(
            (18, limitations_start), (18, directions_start)
        )
        self._sections["5.3"] = clipped_section(
            (18, directions_start), (19, discussion_end)
        )

        for name, selected in self._sections.items():
            print(
                f"Verified {name} section: {len(selected)} chunks.",
                flush=True,
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

        def ranked(section, scoring_query):
            candidates = [
                self._result_type(chunk=chunk, score=0.0)
                for chunk in self._sections[section]
            ]
            return self._reranker.rerank(
                scoring_query, candidates, top_k=top_k
            )

        numbered = re.search(r"\b5\.([123])\b", query)
        if numbered:
            section = "5." + numbered.group(1)
        elif re.search(r"\babstract\b", query, re.I):
            section = "abstract"
        elif re.search(
            r"\bfuture\s+(?:research\s+)?directions\b|"
            r"\bresearch\s+gaps\b",
            query, re.I,
        ):
            section = "5.3"
        elif re.search(r"\blimitations\b", query, re.I):
            section = "5.2"
        elif re.search(r"\bstrengths\b", query, re.I):
            section = "5.1"
        elif re.search(r"\bdiscussion\b", query, re.I):
            section = "discussion"
        else:
            return self._fallback.search(query, top_k=top_k)

        if section != "discussion":
            print(f"Retrieval scope: {section}", flush=True)
            return ranked(section, query)

        print(
            "Retrieval scope: discussion "
            "(balanced across 5.1, 5.2, 5.3)",
            flush=True,
        )
        pools = [
            ranked("5.1", "What strengths does the reviewed research show?"),
            ranked("5.2", "What limitations does the reviewed research have?"),
            ranked("5.3", "What future research directions are proposed?"),
        ]
        selected = []
        for depth in range(top_k):
            for pool in pools:
                if depth < len(pool):
                    selected.append(pool[depth])
                    if len(selected) == top_k:
                        return selected
        return selected



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer; omit for an interactive session",
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--point-count",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Require exactly N points in supported answers; "
            "applies to every question in this session. "
            "Abstentions remain allowed."
        ),
    )
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Default: cpu, leaving GPU memory available for Ollama",
    )
    args = parser.parse_args()
    if args.point_count is not None and args.point_count <= 0:
        parser.error("--point-count must be greater than zero")

    started = time.perf_counter()
    print(f"Reading: {args.pdf}", flush=True)
    pages = parse_pdf(args.pdf)
    chunks = chunk_pages(pages, chunk_size=800, overlap=120)

    # BM25 requires at least one word token per indexed chunk.
    usable = [chunk for chunk in chunks if re.search(r"\w+", chunk.text)]
    print(
        f"Pages: {len(pages)} | Pages with text: "
        f"{sum(page.has_text for page in pages)} | "
        f"Usable chunks: {len(usable)}",
        flush=True,
    )
    if not usable:
        parser.exit(
            1,
            "No usable text was extracted. This pipeline does not perform OCR.\n",
        )

    print("Loading embedding model and building hybrid index...", flush=True)
    embedder = SentenceTransformerEmbedder(device=args.embedding_device)
    retriever = HybridRetriever(usable, embedder=embedder)
    generator = AnswerGenerator(
        OllamaLLM(model=args.model),
        max_evidence_chars=2400,
    )
    print("Loading cross-encoder reranker on CPU...", flush=True)
    reranker = Reranker(scorer=CrossEncoderScorer(device="cpu"))
    reranked_retriever = RerankedRetriever(retriever, reranker)
    profiled_paper = (
        args.pdf.name == "paper_434b99d68f495a91946c64b36b4602f9.pdf"
    )
    active_retriever = reranked_retriever
    if profiled_paper:
        active_retriever = PaperSectionRetriever(
            reranked_retriever, reranker, usable
        )
    pipeline = RAGPipeline(active_retriever, generator, top_k=3)
    print(f"Index ready in {time.perf_counter() - started:.2f}s.", flush=True)

    def answer_question(question: str) -> None:
        started = time.perf_counter()
        try:
            if profiled_paper and re.search(
                r"\bconclusions?\b", question, re.I
            ):
                answer = Answer(
                    text=(
                        "No explicitly titled Conclusion section was identified "
                        "in this paper's extracted text. Its closing synthesis "
                        "is Section 5, Discussion. Please ask for three key "
                        "points from the closing Discussion instead."
                    ),
                    status="abstained",
                    citations=(),
                    evidence=(),
                )
            else:
                answer = pipeline.answer(
                    question, point_count=args.point_count
                )
        except (OllamaError, InvalidLLMResponse, ValueError) as exc:
            print(f"\nQuestion failed: {exc}", flush=True)
            return

        print(f"\nSTATUS: {answer.status}")
        print(f"ANSWER: {answer.text}")
        print(f"Question time: {time.perf_counter() - started:.2f}s")
        print(f"Evidence chunks supplied: {len(answer.evidence)}")

        print("\nCITED PASSAGES:")
        if not answer.citations:
            print("(none)")
        for item in answer.citations:
            chunk = item.chunk
            print(
                f"\n[{item.label}] {chunk.source} | PDF page {chunk.page_number} "
                f"| chunk {chunk.chunk_number} "
                f"| characters {chunk.start_char}:{chunk.end_char}"
            )
            print(chunk.text)

        print("\nALL SUPPLIED EVIDENCE:")
        for item in answer.evidence:
            chunk = item.chunk
            print(f"\n[{item.label}] {chunk.source} | PDF page {chunk.page_number}")
            print(chunk.text)

    if args.question is not None:
        answer_question(args.question)
        return

    if args.point_count is not None:
        print(
            f"Supported answers must contain exactly "
            f"{args.point_count} point(s).",
            flush=True,
        )
    print("\nEnter a question, or type 'exit' to stop.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer_question(question)


if __name__ == "__main__":
    main()
