"""Test the real adapter-to-retriever path without downloading a model."""

import sys
from types import ModuleType
from unittest.mock import Mock, call

import numpy as np
import pytest

from ragforge.embeddings.sentence_transformers import SentenceTransformerEmbedder
from ragforge.ingestion.chunking import TextChunk
from ragforge.retrieval.dense import DenseRetriever


@pytest.fixture
def corpus() -> list[TextChunk]:
    """Create chunks with distinct metadata and deliberately ordered texts."""
    texts = ["large off-axis document", "aligned document", "orthogonal document"]
    return [
        TextChunk(
            source="integration.pdf",
            page_number=index + 1,
            chunk_number=1,
            start_char=10,
            end_char=10 + len(text),
            text=text,
        )
        for index, text in enumerate(texts)
    ]


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Provide deterministic, unnormalized model output through the adapter."""
    vectors_by_text = {
        "large off-axis document": [100, 100, 0],
        "aligned document": [3, 0, 0],
        "orthogonal document": [0, 0, 5],
        "query": [7, 0, 0],
    }

    def encode(texts: list[str], **kwargs: object) -> np.ndarray:
        return np.array(
            [vectors_by_text[text] for text in texts],
            dtype=np.float32,
        )

    
    model = Mock(spec_set=["encode", "get_embedding_dimension"])
    model.get_embedding_dimension.return_value = 3
    model.encode.side_effect = encode
    factory = Mock(return_value=model)

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    return factory, model


def test_adapter_and_retriever_rank_by_cosine_and_preserve_metadata(
    corpus: list[TextChunk],
    fake_backend: tuple[Mock, Mock],
) -> None:
    embedder = SentenceTransformerEmbedder(device="cpu")
    retriever = DenseRetriever(corpus, embedder=embedder)

    results = retriever.search("query", top_k=3)

    # Raw dot product would rank the large off-axis vector first.
    # Cosine similarity must instead rank the aligned vector first.
    expected_chunks = [corpus[1], corpus[0], corpus[2]]
    assert len(results) == 3
    assert [result.score for result in results] == pytest.approx(
        [1.0, 1 / np.sqrt(2), 0.0]
    )

    for result, expected_chunk in zip(results, expected_chunks, strict=True):
        assert result.chunk is expected_chunk
        assert result.chunk.source == "integration.pdf"
        assert result.chunk.page_number == expected_chunk.page_number
        assert result.chunk.chunk_number == 1
        assert result.chunk.start_char == 10
        assert result.chunk.end_char == 10 + len(expected_chunk.text)
        assert result.chunk.text == expected_chunk.text


def test_index_and_queries_reuse_one_model_without_reembedding_documents(
    corpus: list[TextChunk],
    fake_backend: tuple[Mock, Mock],
) -> None:
    factory, model = fake_backend
    embedder = SentenceTransformerEmbedder(device="cpu", batch_size=8)
    retriever = DenseRetriever(corpus, embedder=embedder)

    first_results = retriever.search("query", top_k=2)
    second_results = retriever.search("query", top_k=2)

    assert first_results == second_results
    assert len(first_results) == 2
    assert first_results[0].chunk is corpus[1]
    factory.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    options = {
        "batch_size": 8,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }
    assert model.encode.call_args_list == [
        call([chunk.text for chunk in corpus], **options),
        call(["query"], **options),
        call(["query"], **options),
    ]


def test_blank_query_does_not_reach_model(
    corpus: list[TextChunk],
    fake_backend: tuple[Mock, Mock],
) -> None:
    _, model = fake_backend
    embedder = SentenceTransformerEmbedder(device="cpu")
    retriever = DenseRetriever(corpus, embedder=embedder)
    assert model.encode.call_count == 1

    results = retriever.search(" \n\t ")

    assert results == []
    assert model.encode.call_count == 1