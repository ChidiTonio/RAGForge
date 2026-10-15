"""Model-free tests for conservative abstract routing."""

from ragforge.generation.core import AnswerGenerator, RAGPipeline
from ragforge.ingestion.chunking import chunk_pages
from ragforge.ingestion.pdf import PDFPage
from ragforge.retrieval.abstract import AbstractSectionRetriever


class Fallback:
    def __init__(self):
        self.calls = []

    def search(self, query, *, top_k=5):
        self.calls.append((query, top_k))
        return []


class Reranker:
    def rerank(self, query, results, *, top_k=5):
        return results[:top_k]


class NeverLLM:
    def complete(self, **kwargs):
        raise AssertionError("LLM must not run without scoped evidence.")


def make_scope(text):
    fallback = Fallback()
    page = PDFPage("sample.pdf", 1, text)
    chunks = chunk_pages([page], chunk_size=50, overlap=10)
    scope = AbstractSectionRetriever(fallback, Reranker(), chunks)
    return scope, fallback


def test_abstract_is_clipped_with_exact_offsets():
    text = (
        "Title\nAbstract\n"
        "Tokens connect computation and economic value. "
        "Useful outputs matter more than raw token counts.\n"
        "Keywords: AI; economics\n1. Introduction\nOther material."
    )
    scope, fallback = make_scope(text)
    results = scope.search("What does the abstract present?", top_k=20)
    assert results
    assert not fallback.calls
    left = text.index("Abstract") + len("Abstract")
    right = text.index("Keywords")
    for result in results:
        chunk = result.chunk
        assert left <= chunk.start_char < chunk.end_char <= right
        assert chunk.text == text[chunk.start_char:chunk.end_char]


def test_non_abstract_query_uses_fallback():
    scope, fallback = make_scope("Abstract\nSummary.\nKeywords: AI")
    assert scope.search("What is the methodology?", top_k=2) == []
    assert fallback.calls == [("What is the methodology?", 2)]


def test_missing_abstract_does_not_call_llm_or_fallback():
    scope, fallback = make_scope(
        "3.7 Screening\nTitle and Abstract Screening is a review step."
    )
    pipeline = RAGPipeline(scope, AnswerGenerator(NeverLLM()), top_k=3)
    answer = pipeline.answer("Summarize the abstract.")
    assert answer.status == "no_evidence"
    assert answer.evidence == ()
    assert not fallback.calls


def test_unbounded_abstract_returns_no_results():
    scope, fallback = make_scope("Abstract\nSummary without an end marker.")
    assert scope.search("Summarize the abstract.") == []
    assert not fallback.calls


def test_introduction_can_bound_abstract():
    scope, _ = make_scope(
        "Abstract\nThis paper proposes a framework.\n1. Introduction\nBody."
    )
    results = scope.search("Summarize the abstract.")
    assert results
    assert all("Introduction" not in result.chunk.text for result in results)
