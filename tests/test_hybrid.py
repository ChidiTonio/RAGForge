"""Test reciprocal rank fusion independently of embedding models."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

import ragforge.retrieval.hybrid as hybrid_module
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.hybrid import HybridRetriever
from ragforge.retrieval.types import SearchResult


@pytest.fixture
def chunks():
    return [
        TextChunk(
            source="example.pdf",
            page_number=1,
            chunk_number=index + 1,
            start_char=index * 20,
            end_char=index * 20 + len(text),
            text=text,
        )
        for index, text in enumerate(
            ("alpha evidence", "beta evidence", "gamma evidence")
        )
    ]


@pytest.fixture
def backends(monkeypatch):
    sparse = Mock()
    dense = Mock()
    sparse.search.return_value = []
    dense.search.return_value = []

    sparse_factory = Mock(return_value=sparse)
    dense_factory = Mock(return_value=dense)
    monkeypatch.setattr(hybrid_module, "BM25Retriever", sparse_factory)
    monkeypatch.setattr(hybrid_module, "DenseRetriever", dense_factory)

    return sparse, dense, sparse_factory, dense_factory


def test_constructs_backends_from_same_materialized_corpus(chunks, backends):
    _, _, sparse_factory, dense_factory = backends
    embedder = Mock()

    HybridRetriever(iter(chunks), embedder=embedder)

    sparse_factory.assert_called_once_with(tuple(chunks))
    dense_factory.assert_called_once_with(tuple(chunks), embedder=embedder)


def test_fuses_overlap_and_keeps_single_backend_candidates(chunks, backends):
    sparse, dense, _, _ = backends
    a, b, c = chunks
    sparse.search.return_value = [
        SearchResult(a, -100.0),
        SearchResult(b, 0.0),
    ]
    dense.search.return_value = [
        SearchResult(b, -0.8),
        SearchResult(c, -0.9),
    ]
    retriever = HybridRetriever(
        chunks, embedder=Mock(), candidate_k=7, rrf_k=60
    )

    results = retriever.search("evidence", top_k=10)

    assert [result.chunk for result in results] == [b, a, c]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)
    sparse.search.assert_called_once_with("evidence", top_k=7)
    dense.search.assert_called_once_with("evidence", top_k=7)


def test_fused_ties_follow_corpus_order_not_discovery_order(chunks, backends):
    sparse, dense, _, _ = backends
    a, _, c = chunks
    sparse.search.return_value = [SearchResult(c, 999.0)]
    dense.search.return_value = [SearchResult(a, -1.0)]
    retriever = HybridRetriever(chunks, embedder=Mock())

    results = retriever.search("query")

    assert [result.chunk for result in results] == [a, c]
    assert results[0].score == results[1].score


def test_equal_chunk_values_fuse_into_one_result(chunks, backends):
    sparse, dense, _, _ = backends
    original = chunks[0]
    equivalent = replace(original)
    assert equivalent is not original

    sparse.search.return_value = [SearchResult(original, 4.0)]
    dense.search.return_value = [SearchResult(equivalent, 0.5)]
    retriever = HybridRetriever(chunks, embedder=Mock())

    results = retriever.search("alpha")

    assert len(results) == 1
    assert results[0].chunk == original
    assert results[0].score == pytest.approx(2 / 61)


def test_custom_smoothing_and_top_k(chunks, backends):
    sparse, dense, _, _ = backends
    sparse.search.return_value = [
        SearchResult(chunks[0], 2.0),
        SearchResult(chunks[1], 1.0),
    ]
    retriever = HybridRetriever(chunks, embedder=Mock(), rrf_k=1)

    results = retriever.search("query", top_k=1)

    assert len(results) == 1
    assert results[0].chunk == chunks[0]
    assert results[0].score == pytest.approx(0.5)
    dense.search.assert_called_once_with("query", top_k=20)


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_blank_query_skips_both_backends(query, chunks, backends):
    sparse, dense, _, _ = backends
    retriever = HybridRetriever(chunks, embedder=Mock())

    assert retriever.search(query) == []
    sparse.search.assert_not_called()
    dense.search.assert_not_called()


def test_empty_backend_rankings_return_no_results(chunks, backends):
    retriever = HybridRetriever(chunks, embedder=Mock())

    assert retriever.search("query") == []


@pytest.mark.parametrize("query", [None, 42, ["alpha"]])
def test_rejects_non_string_queries(query, chunks, backends):
    sparse, dense, _, _ = backends
    retriever = HybridRetriever(chunks, embedder=Mock())

    with pytest.raises(TypeError, match="query"):
        retriever.search(query)

    sparse.search.assert_not_called()
    dense.search.assert_not_called()


@pytest.mark.parametrize("name", ["candidate_k", "rrf_k"])
@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (False, TypeError),
        (1.5, TypeError),
        ("5", TypeError),
        (None, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_rejects_invalid_constructor_limits(
    name, value, error, chunks, backends
):
    _, _, sparse_factory, dense_factory = backends

    with pytest.raises(error, match=name):
        HybridRetriever(chunks, embedder=Mock(), **{name: value})

    sparse_factory.assert_not_called()
    dense_factory.assert_not_called()


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (False, TypeError),
        (1.5, TypeError),
        ("5", TypeError),
        (None, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_rejects_invalid_top_k_even_for_blank_query(
    value, error, chunks, backends
):
    sparse, dense, _, _ = backends
    retriever = HybridRetriever(chunks, embedder=Mock())

    with pytest.raises(error, match="top_k"):
        retriever.search("", top_k=value)

    sparse.search.assert_not_called()
    dense.search.assert_not_called()


def test_rejects_empty_corpus_before_building_backends(backends):
    _, _, sparse_factory, dense_factory = backends

    with pytest.raises(ValueError, match="At least one"):
        HybridRetriever([], embedder=Mock())

    sparse_factory.assert_not_called()
    dense_factory.assert_not_called()


def test_rejects_non_chunk_items_before_building_backends(backends):
    _, _, sparse_factory, dense_factory = backends

    with pytest.raises(TypeError, match="TextChunk"):
        HybridRetriever(["not a chunk"], embedder=Mock())

    sparse_factory.assert_not_called()
    dense_factory.assert_not_called()


def test_rejects_value_equal_duplicates_before_building_backends(
    chunks, backends
):
    _, _, sparse_factory, dense_factory = backends

    with pytest.raises(ValueError, match="Duplicate"):
        HybridRetriever(
            [chunks[0], replace(chunks[0])],
            embedder=Mock(),
        )

    sparse_factory.assert_not_called()
    dense_factory.assert_not_called()


@pytest.mark.parametrize("backend_index", [0, 1])
def test_backend_errors_propagate(chunks, backends, backend_index):
    error = RuntimeError("backend failed")
    backends[backend_index].search.side_effect = error
    retriever = HybridRetriever(chunks, embedder=Mock())

    with pytest.raises(RuntimeError) as caught:
        retriever.search("query")

    assert caught.value is error


@pytest.mark.parametrize("text", [" ", "!!!"])
def test_real_bm25_rejects_invalid_text_before_embedding(chunks, text):
    embedder = Mock()

    with pytest.raises(ValueError, match="word token"):
        HybridRetriever(
            [replace(chunks[0], text=text)],
            embedder=embedder,
        )

    embedder.embed.assert_not_called()


def test_real_backends_work_without_model_downloads(chunks):
    vectors_by_text = {
        "alpha evidence": [1.0, 0.0],
        "beta evidence": [0.0, 1.0],
        "gamma evidence": [-1.0, 0.0],
        "alpha": [1.0, 0.0],
        "unknown": [0.0, 1.0],
        "!!!": [0.0, 1.0],
    }
    embedder = Mock()
    embedder.embed.side_effect = lambda texts: np.array(
        [vectors_by_text[text] for text in texts],
        dtype=np.float32,
    )
    retriever = HybridRetriever(
        chunks, embedder=embedder, candidate_k=2
    )

    alpha_results = retriever.search("alpha", top_k=10)
    assert [result.chunk for result in alpha_results] == chunks[:2]
    assert alpha_results[0].score == pytest.approx(2 / 61)
    assert alpha_results[1].score == pytest.approx(1 / 62)

    # With no lexical matches, dense retrieval still contributes.
    for query in ("unknown", "!!!"):
        results = retriever.search(query, top_k=10)
        assert [result.chunk for result in results] == [
            chunks[1], chunks[0]
        ]
        assert results[0].score == pytest.approx(1 / 61)
        assert results[1].score == pytest.approx(1 / 62)

        calls_before_blank_query = embedder.embed.call_count
    assert retriever.search(" ") == []
    assert embedder.embed.call_count == calls_before_blank_query
