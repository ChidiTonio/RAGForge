"""Shared local PDF question-answering backend for CLI and browser clients."""

import hashlib
import json
import re
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.generation.core import Answer, AnswerGenerator, RAGPipeline
from ragforge.generation.ollama import OllamaLLM
from ragforge.ingestion.chunking import chunk_pages
from ragforge.ingestion.pdf import parse_pdf
from ragforge.retrieval.abstract import AbstractSectionRetriever
from ragforge.retrieval.cross_encoder import CrossEncoderScorer
from ragforge.retrieval.hybrid import HybridRetriever
from ragforge.retrieval.reranking import Reranker
from ragforge.storage import InvalidIndexCache, hash_file, load_index, save_index

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _cache_compatibility(pdf, *, revision, device):
    """Describe configured compatibility, not immutable upstream identity."""
    dependencies = {}
    for package in ("pypdf", "numpy", "sentence-transformers", "transformers", "torch"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "pdf_sha256": hash_file(pdf),
        "source": pdf.name,
        "ingestion_version": "pdf-strip-page-bound-chunks-word-filter-v1",
        "chunk_size": 800,
        "overlap": 120,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": revision,
        "embedding_device": device,
        "embedding_batch_size": 32,
        "dependencies": dependencies,
    }


def _cache_path(directory, compatibility):
    encoded = json.dumps(compatibility, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )
    key = hashlib.sha256(encoded).hexdigest()
    return directory / f"{key}.npz"


class RerankedRetriever:
    """Retrieve a wider candidate pool, then return reranked passages."""

    def __init__(self, retriever, reranker, *, candidate_k=30):
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(self, query: str, *, top_k: int = 5):
        candidates = self._retriever.search(query, top_k=max(self._candidate_k, top_k))
        return self._reranker.rerank(query, candidates, top_k=top_k)


class PaperSectionRetriever:
    """Verified section profile for this PDF, not a general section parser."""

    def __init__(self, fallback, reranker, chunks, *, report=print):
        self._report = report
        from dataclasses import replace

        from ragforge.retrieval.types import SearchResult

        self._fallback = fallback
        self._reranker = reranker
        self._result_type = SearchResult
        self._sections = {}
        by_page = {}
        for chunk in chunks:
            by_page.setdefault(chunk.page_number, []).append(chunk)
        page_text = {}
        for number, batch in by_page.items():
            chars = [None] * max(c.end_char for c in batch)
            for chunk in batch:
                if len(chunk.text) != chunk.end_char - chunk.start_char:
                    raise ValueError("Unexpected chunk offsets.")
                for offset, character in enumerate(chunk.text, start=chunk.start_char):
                    if chars[offset] not in (None, character):
                        raise ValueError("Overlapping chunk text disagrees.")
                    chars[offset] = character
            page_text[number] = "".join(
                character if character is not None else " " for character in chars
            )

        def marker(page, pattern):
            match = re.search(pattern, page_text.get(page, ""), re.IGNORECASE | re.MULTILINE)
            if match is None:
                raise ValueError(f"Verified section marker missing on page {page}: {pattern}")
            return match.start()

        abstract_start = marker(1, "Recent\\s+breakthroughs\\s+in\\s+Large\\s+Language\\s+Models")
        abstract_end = marker(1, "^\\s*KEYWORDS\\s*$")
        discussion_start = marker(15, "^\\s*5\\s+Discussion\\s*$")
        discussion_end = marker(19, "^\\s*Author contributions\\s*$")

        def clipped_section(start, end):
            if start >= end:
                raise ValueError("Invalid section boundaries.")
            selected = []
            for chunk in chunks:
                page = chunk.page_number
                if not start[0] <= page <= end[0]:
                    continue
                left = max(chunk.start_char, start[1] if page == start[0] else chunk.start_char)
                right = min(chunk.end_char, end[1] if page == end[0] else chunk.end_char)
                if left >= right:
                    continue
                text = chunk.text[left - chunk.start_char : right - chunk.start_char]
                if re.search("\\w+", text):
                    selected.append(replace(chunk, text=text, start_char=left, end_char=right))
            if not selected:
                raise ValueError("Section contains no usable chunks.")
            return selected

        self._sections["abstract"] = clipped_section((1, abstract_start), (1, abstract_end))
        self._sections["discussion"] = clipped_section((15, discussion_start), (19, discussion_end))
        strengths_start = marker(15, "^\\s*5\\.1\\s+Strengths of existing research\\s*$")
        limitations_start = marker(
            18, "^\\s*5\\.2\\s+Limitations of existing research studies\\s*$"
        )
        directions_start = marker(18, "^\\s*5\\.3\\s+Research gaps and future directions\\s*$")
        self._sections["5.1"] = clipped_section((15, strengths_start), (18, limitations_start))
        self._sections["5.2"] = clipped_section((18, limitations_start), (18, directions_start))
        self._sections["5.3"] = clipped_section((18, directions_start), (19, discussion_end))
        for name, selected in self._sections.items():
            self._report(f"Verified {name} section: {len(selected)} chunks.")

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
                self._result_type(chunk=chunk, score=0.0) for chunk in self._sections[section]
            ]
            return self._reranker.rerank(scoring_query, candidates, top_k=top_k)

        numbered = re.search("\\b5\\.([123])\\b", query)
        if numbered:
            section = "5." + numbered.group(1)
        elif re.search("\\babstract\\b", query, re.IGNORECASE):
            section = "abstract"
        elif re.search(
            "\\bfuture\\s+(?:research\\s+)?directions\\b|\\bresearch\\s+gaps\\b",
            query,
            re.IGNORECASE,
        ):
            section = "5.3"
        elif re.search("\\blimitations\\b", query, re.IGNORECASE):
            section = "5.2"
        elif re.search("\\bstrengths\\b", query, re.IGNORECASE):
            section = "5.1"
        elif re.search("\\bdiscussion\\b", query, re.IGNORECASE):
            section = "discussion"
        else:
            return self._fallback.search(query, top_k=top_k)
        if section != "discussion":
            self._report(f"Retrieval scope: {section}")
            return ranked(section, query)
        self._report("Retrieval scope: discussion (balanced across 5.1, 5.2, 5.3)")
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


def _discard_progress(message):
    """Default progress callback: do not print document information."""


class PDFSession:
    """Reuse a prepared pipeline across questions."""

    def __init__(self, pipeline, *, profiled_paper=False):
        self._pipeline = pipeline
        self._profiled_paper = profiled_paper

    def answer(self, question: str, *, point_count: int | None = None) -> Answer:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        if point_count is not None:
            if isinstance(point_count, bool) or not isinstance(point_count, int):
                raise TypeError("point_count must be an integer")
            if point_count <= 0:
                raise ValueError("point_count must be greater than zero")
        if self._profiled_paper and re.search("\\bconclusions?\\b", question, re.IGNORECASE):
            return Answer(
                text="No explicitly titled Conclusion section was identified in this paper's extracted text. Its closing synthesis is Section 5, Discussion. Please ask for three key points from the closing Discussion instead.",
                status="abstained",
                citations=(),
                evidence=(),
            )
        return self._pipeline.answer(question, point_count=point_count)


def build_pdf_session(
    pdf: str | Path,
    *,
    model: str = "qwen2.5:3b",
    embedding_device: str = "cpu",
    cache_dir: str | Path | None = None,
    rebuild_index: bool = False,
    embedding_revision: str | None = None,
    use_paper_profile: bool = False,
    report=None,
) -> PDFSession:
    """Prepare a local PDF session with an optional progress callback.

    Generic retrieval is the default. Enable the paper profile only for
    the verified paper; this option is not a general section detector.
    Exceptions propagate to the calling interface.
    """
    pdf = Path(pdf)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if rebuild_index and cache_dir is None:
        raise ValueError("rebuild_index requires cache_dir")
    if embedding_revision is not None:
        if not isinstance(embedding_revision, str):
            raise TypeError("embedding_revision must be a string")
        if not embedding_revision.strip():
            raise ValueError("embedding_revision must not be blank")
    if not isinstance(use_paper_profile, bool):
        raise TypeError("use_paper_profile must be a boolean")
    if report is None:
        report = _discard_progress
    if not callable(report):
        raise TypeError("report must be callable")
    started = time.perf_counter()
    compatibility = None
    cache_path = None
    stored = None
    if cache_dir is not None:
        if embedding_revision is None:
            report(
                "Warning: embedding model revision is unpinned; cache metadata cannot detect upstream model changes. Use an immutable --embedding-revision or --rebuild-index."
            )
        compatibility = _cache_compatibility(
            pdf, revision=embedding_revision, device=embedding_device
        )
        cache_path = _cache_path(cache_dir, compatibility)
        if not rebuild_index:
            try:
                stored = load_index(cache_path, compatibility=compatibility)
            except FileNotFoundError:
                report("Index cache miss; rebuilding.")
            except (InvalidIndexCache, OSError) as exc:
                report(f"Warning: cannot reuse index cache: {exc}")
    embedder = None
    if stored is not None:
        report("Loading embedding model for cached index...")
        if embedding_revision is None:
            embedder = SentenceTransformerEmbedder(device=embedding_device)
        else:
            embedder = SentenceTransformerEmbedder(
                device=embedding_device, revision=embedding_revision
            )
        if stored.document_embeddings.shape[1] != embedder.dimension:
            report("Warning: cached embedding dimension differs from the model; rebuilding.")
            stored = None
    if stored is None:
        report(f"Reading: {pdf}")
        pages = parse_pdf(pdf)
        chunks = chunk_pages(pages, chunk_size=800, overlap=120)
        usable = [chunk for chunk in chunks if re.search("\\w+", chunk.text)]
        report(
            f"Pages: {len(pages)} | Pages with text: {sum(page.has_text for page in pages)} | Usable chunks: {len(usable)}"
        )
        if not usable:
            raise ValueError("No usable text was extracted. This pipeline does not perform OCR.")
    else:
        usable = stored.chunks
        report(f"Index cache hit | Usable chunks: {len(usable)}")
    report("Loading embedding model and building hybrid index...")
    if embedder is None:
        if embedding_revision is None:
            embedder = SentenceTransformerEmbedder(device=embedding_device)
        else:
            embedder = SentenceTransformerEmbedder(
                device=embedding_device, revision=embedding_revision
            )
    if cache_path is None:
        retriever = HybridRetriever(usable, embedder=embedder)
    else:
        vectors = (
            stored.document_embeddings
            if stored is not None
            else embedder.embed([chunk.text for chunk in usable])
        )
        retriever = HybridRetriever(usable, embedder=embedder, document_embeddings=vectors)
        if stored is None:
            try:
                save_index(
                    cache_path,
                    chunks=usable,
                    document_embeddings=vectors,
                    compatibility=compatibility,
                )
            except (OSError, ValueError, TypeError) as exc:
                report(f"Warning: could not save index cache: {exc}")
            else:
                report(f"Saved index cache: {cache_path}")
    llm = OllamaLLM(model=model)
    generator = AnswerGenerator(
        llm,
        max_evidence_chars=2400,
        verification_llm=llm,
    )
    report("Loading cross-encoder reranker on CPU...")
    reranker = Reranker(scorer=CrossEncoderScorer(device="cpu"))
    reranked_retriever = RerankedRetriever(retriever, reranker)
    profiled_paper = use_paper_profile
    active_retriever = reranked_retriever
    if profiled_paper:
        active_retriever = PaperSectionRetriever(
            reranked_retriever, reranker, usable, report=report
        )
    else:
        active_retriever = AbstractSectionRetriever(
            reranked_retriever, reranker, usable, report=report
        )
    pipeline = RAGPipeline(active_retriever, generator, top_k=3)
    report(f"Index ready in {time.perf_counter() - started:.2f}s.")
    return PDFSession(pipeline, profiled_paper=profiled_paper)
