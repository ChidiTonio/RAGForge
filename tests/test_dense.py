"""Test exact dense retrieval without downloading an embedding model."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.types import SearchResult


def make_chunk(text: str, *, page_number: int = 1) -> TextChunk:
    """Create a synthetic chunk with explicit source metadata."""
    return TextChunk(
        source="synthetic.pdf",
        page_number=page_number,
        chunk_number=1,
        start_char=10,
        end_char=10 + len(text),
        text=text,
    )


class StubEmbedder:
    """Return queued outputs and record calls, including malformed outputs."""

    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        if not self.outputs:
            raise AssertionError("Unexpected embedding call.")
        # Cast affects static typing only; malformed test data stays unchanged.
        return cast(NDArray[np.float32], self.outputs.pop(0))


@pytest.fixture
def corpus() -> list[TextChunk]:
    return [
        make_chunk("document alpha", page_number=1),
        make_chunk("document beta", page_number=2),
        make_chunk("document gamma", page_number=3),
    ]


def make_retriever(
    corpus: list[TextChunk],
) -> tuple[DenseRetriever, StubEmbedder]:
    """Use distinct directions with known cosine scores."""
    embedder = StubEmbedder(
        np.array([[0, 1], [3, 4], [1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    return DenseRetriever(corpus, embedder=embedder), embedder


def test_search_ranks_by_cosine_similarity(corpus: list[TextChunk]) -> None:
    retriever, _ = make_retriever(corpus)

    results = retriever.search("unrelated query wording")

    assert [result.chunk.page_number for result in results] == [3, 2, 1]
    assert [result.score for result in results] == pytest.approx([1.0, 0.6, 0.0])
    assert all(isinstance(result, SearchResult) for result in results)
    assert all(isinstance(result.score, float) for result in results)


def test_search_preserves_original_chunk_and_metadata(corpus: list[TextChunk]) -> None:
    retriever, _ = make_retriever(corpus)

    result = retriever.search("query")[0]

    assert result.chunk is corpus[2]
    assert result.chunk.source == "synthetic.pdf"
    assert result.chunk.page_number == 3
    assert result.chunk.chunk_number == 1
    assert result.chunk.start_char == 10
    assert result.chunk.end_char == 10 + len(corpus[2].text)
    assert result.chunk.text == corpus[2].text


def test_vector_magnitude_does_not_override_direction() -> None:
    chunks = [make_chunk("large"), make_chunk("aligned", page_number=2)]
    embedder = StubEmbedder(
        np.array([[100, 100], [1, 0]], dtype=np.float32),
        np.array([[7, 0]], dtype=np.float32),
    )

    results = DenseRetriever(chunks, embedder=embedder).search("query")

    assert results[0].chunk is chunks[1]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1 / np.sqrt(2))


def test_equal_scores_preserve_input_order() -> None:
    chunks = [
        make_chunk("first", page_number=3),
        make_chunk("second", page_number=1),
        make_chunk("third", page_number=2),
    ]
    embedder = StubEmbedder(
        np.array([[1, 0], [2, 0], [0, 1]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    retriever = DenseRetriever(chunks, embedder=embedder)

    results = retriever.search("query", top_k=2)

    assert [result.chunk.page_number for result in results] == [3, 1]
    assert results[0].score == results[1].score
    assert retriever.search("query", top_k=2) == results


def test_zero_and_negative_scores_are_retained(corpus: list[TextChunk]) -> None:
    embedder = StubEmbedder(
        np.array([[-1, 0], [0, 1], [1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )

    results = DenseRetriever(corpus, embedder=embedder).search("query")

    assert [result.chunk.page_number for result in results] == [3, 2, 1]
    assert [result.score for result in results] == pytest.approx([1.0, 0.0, -1.0])


@pytest.mark.parametrize("top_k", [1, 2, 3, 100])
def test_top_k_limits_results(corpus: list[TextChunk], top_k: int) -> None:
    retriever, _ = make_retriever(corpus)

    results = retriever.search("query", top_k=top_k)

    assert len(results) == min(top_k, len(corpus))
    assert [result.chunk.page_number for result in results] == [3, 2, 1][:top_k]


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_is_rejected(corpus: list[TextChunk], top_k: int) -> None:
    retriever, embedder = make_retriever(corpus)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("query", top_k=top_k)

    assert len(embedder.calls) == 1


@pytest.mark.parametrize("top_k", [True, False, 1.5, "2", None])
def test_noninteger_top_k_is_rejected(corpus: list[TextChunk], top_k: object) -> None:
    retriever, embedder = make_retriever(corpus)

    with pytest.raises(TypeError, match="top_k"):
        retriever.search("query", top_k=cast(int, top_k))

    assert len(embedder.calls) == 1


@pytest.mark.parametrize("query", ["", " \n\t "])
def test_blank_queries_skip_embedding(corpus: list[TextChunk], query: str) -> None:
    retriever, embedder = make_retriever(corpus)

    assert retriever.search(query) == []
    assert embedder.calls == [[chunk.text for chunk in corpus]]


def test_top_k_is_validated_before_blank_query(corpus: list[TextChunk]) -> None:
    retriever, _ = make_retriever(corpus)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("", top_k=0)


def test_documents_are_embedded_once(corpus: list[TextChunk]) -> None:
    embedder = StubEmbedder(
        np.array([[0, 1], [3, 4], [1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.float32),
    )
    retriever = DenseRetriever(corpus, embedder=embedder)

    retriever.search("first query")
    retriever.search("second query")

    assert embedder.calls == [
        [chunk.text for chunk in corpus],
        ["first query"],
        ["second query"],
    ]


def test_empty_corpus_is_rejected_before_embedding() -> None:
    embedder = StubEmbedder()

    with pytest.raises(ValueError, match="At least one chunk"):
        DenseRetriever([], embedder=embedder)

    assert embedder.calls == []


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_blank_chunk_is_rejected_before_embedding(text: str) -> None:
    embedder = StubEmbedder()

    with pytest.raises(ValueError, match="nonblank"):
        DenseRetriever([make_chunk("valid"), make_chunk(text)], embedder=embedder)

    assert embedder.calls == []


def test_punctuation_only_text_is_allowed() -> None:
    chunk = make_chunk("...!?")
    embedder = StubEmbedder(
        np.array([[1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )

    results = DenseRetriever([chunk], embedder=embedder).search("?!")

    assert results[0].chunk is chunk
    assert embedder.calls == [["...!?"], ["?!"]]


def test_accepts_generator_input(corpus: list[TextChunk]) -> None:
    embedder = StubEmbedder(
        np.array([[0, 1], [3, 4], [1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    retriever = DenseRetriever((chunk for chunk in corpus), embedder=embedder)

    assert retriever.search("query")[0].chunk is corpus[2]


@pytest.mark.parametrize("stage", ["documents", "query"])
@pytest.mark.parametrize(
    ("output", "error", "message"),
    [
        ([[1.0, 0.0]], TypeError, "NumPy array"),
        (np.array([1, 0], dtype=np.float32), ValueError, "two-dimensional"),
        (np.ones((1, 1, 2), dtype=np.float32), ValueError, "two-dimensional"),
        (np.empty((0, 2), dtype=np.float32), ValueError, "row count"),
        (np.ones((2, 2), dtype=np.float32), ValueError, "row count"),
        (np.empty((1, 0), dtype=np.float32), ValueError, "dimension"),
        (np.array([[1, 0]], dtype=np.float64), ValueError, "dtype float32"),
        (np.array([[1, 0]], dtype=np.int32), ValueError, "dtype float32"),
        (np.array([[np.nan, 0]], dtype=np.float32), ValueError, "finite"),
        (np.array([[np.inf, 0]], dtype=np.float32), ValueError, "finite"),
        (np.array([[-np.inf, 0]], dtype=np.float32), ValueError, "finite"),
        (np.zeros((1, 2), dtype=np.float32), ValueError, "nonzero norm"),
    ],
)
def test_invalid_embeddings_are_rejected(
    stage: str,
    output: object,
    error: type[Exception],
    message: str,
) -> None:
    chunks = [make_chunk("document")]

    if stage == "documents":
        with pytest.raises(error, match=message):
            DenseRetriever(chunks, embedder=StubEmbedder(output))
    else:
        embedder = StubEmbedder(np.array([[1, 0]], dtype=np.float32), output)
        retriever = DenseRetriever(chunks, embedder=embedder)

        with pytest.raises(error, match=message):
            retriever.search("query")


def test_query_dimension_must_match_index() -> None:
    embedder = StubEmbedder(
        np.array([[1, 0]], dtype=np.float32),
        np.array([[1, 0, 0]], dtype=np.float32),
    )
    retriever = DenseRetriever([make_chunk("document")], embedder=embedder)

    with pytest.raises(ValueError, match="dimension must match"):
        retriever.search("query")


def test_normalization_does_not_modify_embedder_arrays() -> None:
    documents = np.array([[3, 4]], dtype=np.float32)
    query = np.array([[6, 8]], dtype=np.float32)
    original_documents = documents.copy()
    original_query = query.copy()
    documents.setflags(write=False)
    query.setflags(write=False)
    embedder = StubEmbedder(documents, query)

    result = DenseRetriever([make_chunk("document")], embedder=embedder).search("query")[0]

    assert result.score == pytest.approx(1.0)
    np.testing.assert_array_equal(documents, original_documents)
    np.testing.assert_array_equal(query, original_query)


def test_index_owns_independent_document_vectors() -> None:
    documents = np.array([[1, 0], [0, 1]], dtype=np.float32)
    chunks = [make_chunk("first"), make_chunk("second", page_number=2)]
    embedder = StubEmbedder(documents, np.array([[1, 0]], dtype=np.float32))
    retriever = DenseRetriever(chunks, embedder=embedder)

    documents[:] = np.array([[0, 1], [1, 0]], dtype=np.float32)

    assert retriever.search("query")[0].chunk is chunks[0]


@pytest.mark.parametrize(
    "scale",
    [float(np.finfo(np.float32).max), float(np.finfo(np.float32).tiny)],
)
def test_extreme_finite_float32_vectors_normalize_safely(scale: float) -> None:
    embedder = StubEmbedder(
        np.array([[scale, scale]], dtype=np.float32),
        np.array([[scale, scale]], dtype=np.float32),
    )

    result = DenseRetriever([make_chunk("document")], embedder=embedder).search("query")[0]

    assert np.isfinite(result.score)
    assert -1.0 <= result.score <= 1.0
    assert result.score == pytest.approx(1.0)