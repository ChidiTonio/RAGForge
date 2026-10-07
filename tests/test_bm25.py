"""Tests for the BM25 lexical retrieval baseline."""

import pytest

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.bm25 import BM25Retriever, tokenize


def make_chunk(text: str, *, page_number: int = 1) -> TextChunk:
    """Create a synthetic chunk with explicit source metadata."""
    start_char = 10
    return TextChunk(
        source="synthetic.pdf",
        page_number=page_number,
        chunk_number=1,
        start_char=start_char,
        end_char=start_char + len(text),
        text=text,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Neural SEARCH!", ["neural", "search"]),
        ("dense-retrieval, sparse.retrieval", ["dense", "retrieval", "sparse", "retrieval"]),
        ("Café Straße", ["café", "strasse"]),
        ("model_v2 123", ["model_v2", "123"]),
        ("", []),
        (" \n\t ", []),
        ("...!?", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


@pytest.fixture
def corpus() -> list[TextChunk]:
    return [
        make_chunk("neural neural search", page_number=1),
        make_chunk("sparse lexical matching", page_number=2),
        make_chunk("database storage systems", page_number=3),
    ]


def test_search_returns_relevant_chunk(corpus: list[TextChunk]) -> None:
    retriever = BM25Retriever(corpus)

    results = retriever.search("NEURAL!")

    assert len(results) == 1
    assert results[0].chunk is corpus[0]
    assert isinstance(results[0].score, float)
    assert results[0].score > 0


def test_search_preserves_chunk_metadata(corpus: list[TextChunk]) -> None:
    result = BM25Retriever(corpus).search("lexical")[0]

    assert result.chunk is corpus[1]
    assert result.chunk.source == "synthetic.pdf"
    assert result.chunk.page_number == 2
    assert result.chunk.chunk_number == 1
    assert result.chunk.start_char == 10
    assert result.chunk.end_char == 10 + len(corpus[1].text)
    assert result.chunk.text == corpus[1].text


def test_search_ranks_by_descending_score() -> None:
    chunks = [
        make_chunk("alpha filler filler", page_number=1),
        make_chunk("beta filler filler", page_number=2),
        make_chunk("gamma filler filler", page_number=3),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.search("alpha beta beta")

    assert [result.chunk.page_number for result in results] == [2, 1]
    assert results[0].score > results[1].score > 0


@pytest.mark.parametrize("query", ["", " \n\t ", "...!?", "unseenword"])
def test_queries_without_matches_return_empty(
    corpus: list[TextChunk], query: str
) -> None:
    assert BM25Retriever(corpus).search(query) == []


def test_unknown_query_terms_do_not_remove_known_matches(
    corpus: list[TextChunk],
) -> None:
    retriever = BM25Retriever(corpus)

    results = retriever.search("neural unseenword")

    assert len(results) == 1
    assert results[0].chunk is corpus[0]
    assert results[0].score == pytest.approx(retriever.search("neural")[0].score)


def test_top_k_limits_results(corpus: list[TextChunk]) -> None:
    results = BM25Retriever(corpus).search("neural sparse database", top_k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_top_k_larger_than_matches_returns_only_matches(
    corpus: list[TextChunk],
) -> None:
    results = BM25Retriever(corpus).search("neural", top_k=100)

    assert len(results) == 1
    assert results[0].chunk is corpus[0]


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_is_rejected(corpus: list[TextChunk], top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        BM25Retriever(corpus).search("neural", top_k=top_k)


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one chunk"):
        BM25Retriever([])


@pytest.mark.parametrize("text", ["", " \n\t ", "...!?"])
def test_tokenless_chunk_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="word token"):
        BM25Retriever([make_chunk("valid text"), make_chunk(text)])


def test_accepts_generator_input(corpus: list[TextChunk]) -> None:
    retriever = BM25Retriever(chunk for chunk in corpus)

    assert retriever.search("database")[0].chunk is corpus[2]


def test_equal_scores_preserve_input_order() -> None:
    chunks = [
        make_chunk("alpha", page_number=3),
        make_chunk("alpha", page_number=1),
        make_chunk("beta", page_number=2),
        make_chunk("gamma", page_number=4),
        make_chunk("delta", page_number=5),
    ]
    retriever = BM25Retriever(chunks)

    results = retriever.search("alpha")

    assert len(results) == 2
    assert results[0].score == results[1].score
    assert [result.chunk.page_number for result in results] == [3, 1]
    assert retriever.search("alpha") == results


def test_zero_score_matching_chunk_is_retained() -> None:
    chunks = [make_chunk("alpha"), make_chunk("beta", page_number=2)]

    results = BM25Retriever(chunks).search("alpha")

    assert len(results) == 1
    assert results[0].chunk is chunks[0]
    assert results[0].score == pytest.approx(0.0)


def test_negative_score_matching_chunk_is_retained() -> None:
    chunk = make_chunk("alpha")

    results = BM25Retriever([chunk]).search("alpha")

    assert len(results) == 1
    assert results[0].chunk is chunk
    assert results[0].score < 0