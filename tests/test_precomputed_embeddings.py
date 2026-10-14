"""Test precomputed document vectors without loading real models."""

from collections.abc import Sequence

import numpy as np
import pytest

from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.dense import DenseRetriever
from ragforge.retrieval.hybrid import HybridRetriever


class StubEmbedder:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def embed(self, texts: Sequence[str]):
        self.calls.append(list(texts))
        if not self.outputs:
            raise AssertionError("Unexpected embedding call.")
        return self.outputs.pop(0)


def make_chunks():
    texts = ["alpha first passage", "beta second passage", "gamma third passage"]
    return [
        TextChunk(
            source="synthetic.pdf",
            page_number=index,
            chunk_number=1,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        for index, text in enumerate(texts, start=1)
    ]


@pytest.mark.parametrize("retriever_type", [DenseRetriever, HybridRetriever])
def test_precomputed_matches_fresh_index_and_skips_document_embedding(retriever_type):
    chunks = make_chunks()
    documents = np.array([[0, 1], [3, 4], [1, 0]], dtype=np.float32)
    query = np.array([[1, 0]], dtype=np.float32)
    fresh_embedder = StubEmbedder(documents.copy(), query.copy())
    cached_embedder = StubEmbedder(query.copy())

    fresh = retriever_type(chunks, embedder=fresh_embedder)
    cached = retriever_type(
        chunks,
        embedder=cached_embedder,
        document_embeddings=documents,
    )
    assert cached_embedder.calls == []

    fresh_results = fresh.search("alpha query", top_k=3)
    cached_results = cached.search("alpha query", top_k=3)

    assert [item.chunk for item in cached_results] == [
        item.chunk for item in fresh_results
    ]
    assert [item.score for item in cached_results] == pytest.approx(
        [item.score for item in fresh_results]
    )
    assert all(any(item.chunk is chunk for chunk in chunks) for item in cached_results)
    assert fresh_embedder.calls == [[chunk.text for chunk in chunks], ["alpha query"]]
    assert cached_embedder.calls == [["alpha query"]]


@pytest.mark.parametrize("retriever_type", [DenseRetriever, HybridRetriever])
@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (np.ones(3, dtype=np.float32), "two-dimensional"),
        (np.ones((2, 2), dtype=np.float32), "row count"),
        (np.empty((3, 0), dtype=np.float32), "dimension"),
        (np.ones((3, 2), dtype=np.float64), "dtype float32"),
        (np.full((3, 2), np.nan, dtype=np.float32), "finite"),
        (np.zeros((3, 2), dtype=np.float32), "nonzero norm"),
    ],
)
def test_invalid_precomputed_vectors_do_not_call_embedder(
    retriever_type, vectors, message
):
    embedder = StubEmbedder()

    with pytest.raises(ValueError, match=message):
        retriever_type(
            make_chunks(),
            embedder=embedder,
            document_embeddings=vectors,
        )

    assert embedder.calls == []


@pytest.mark.parametrize("retriever_type", [DenseRetriever, HybridRetriever])
def test_index_owns_vectors_independently_of_precomputed_input(retriever_type):
    chunks = make_chunks()
    documents = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    original = documents.copy()
    embedder = StubEmbedder(
        np.array([[1, 0]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
    )
    retriever = retriever_type(
        chunks,
        embedder=embedder,
        document_embeddings=documents,
    )

    np.testing.assert_array_equal(documents, original)
    before = retriever.search("unrelated query", top_k=3)
    documents[:] = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32)
    after = retriever.search("unrelated query", top_k=3)

    assert after == before
    assert before[0].chunk is chunks[0]
