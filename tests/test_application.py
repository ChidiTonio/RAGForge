"""Test the shared PDF backend without loading models or reading PDFs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from ragforge import application


@pytest.fixture
def backend(monkeypatch):
    page = SimpleNamespace(has_text=True)
    chunk = SimpleNamespace(text="Usable document text.")
    answer = SimpleNamespace(status="answered", text="Supported answer.", citations=(), evidence=())
    pipeline = Mock()
    pipeline.answer.return_value = answer
    vectors = np.array([[3, 4]], dtype=np.float32)
    stored = SimpleNamespace(chunks=(chunk,), document_embeddings=vectors)

    replacements = {
        "parse_pdf": Mock(return_value=[page]),
        "chunk_pages": Mock(return_value=[chunk]),
        "SentenceTransformerEmbedder": Mock(),
        "HybridRetriever": Mock(),
        "OllamaLLM": Mock(),
        "AnswerGenerator": Mock(),
        "CrossEncoderScorer": Mock(),
        "Reranker": Mock(),
        "RerankedRetriever": Mock(),
        "PaperSectionRetriever": Mock(),
        "AbstractSectionRetriever": Mock(),
        "RAGPipeline": Mock(return_value=pipeline),
        "_cache_compatibility": Mock(return_value={"source": "generic.pdf"}),
        "_cache_path": Mock(return_value=Path("mock-index.npz")),
        "load_index": Mock(return_value=stored),
        "save_index": Mock(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(application, name, replacement)

    embedder = replacements["SentenceTransformerEmbedder"].return_value
    embedder.dimension = 2
    embedder.embed.return_value = vectors
    return SimpleNamespace(
        pipeline=pipeline,
        answer=answer,
        chunk=chunk,
        stored=stored,
        vectors=vectors,
        embedder=embedder,
        mocks=replacements,
    )


def test_session_reuses_pipeline_and_returns_answers(backend, capsys):
    session = application.build_pdf_session("generic.pdf")
    assert session.answer("First question?") is backend.answer
    assert session.answer("Second question?", point_count=3) is backend.answer
    assert backend.pipeline.answer.call_args_list == [
        call("First question?", point_count=None),
        call("Second question?", point_count=3),
    ]
    application.RAGPipeline.assert_called_once()
    application.parse_pdf.assert_called_once_with(Path("generic.pdf"))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("question", "count", "error"),
    [
        (None, None, TypeError),
        ("", None, ValueError),
        ("   ", None, ValueError),
        ("Question?", True, TypeError),
        ("Question?", 1.5, TypeError),
        ("Question?", "3", TypeError),
        ("Question?", 0, ValueError),
        ("Question?", -2, ValueError),
    ],
)
def test_invalid_answer_arguments_do_not_call_pipeline(backend, question, count, error):
    session = application.PDFSession(backend.pipeline)
    with pytest.raises(error):
        session.answer(question, point_count=count)
    backend.pipeline.answer.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rebuild_index": True}, ValueError),
        ({"embedding_revision": ""}, ValueError),
        ({"embedding_revision": "  "}, ValueError),
        ({"embedding_revision": 123}, TypeError),
        ({"use_paper_profile": "yes"}, TypeError),
        ({"report": "not callable"}, TypeError),
    ],
)
def test_invalid_builder_arguments_fail_before_loading(backend, kwargs, error):
    with pytest.raises(error):
        application.build_pdf_session("generic.pdf", **kwargs)
    for replacement in backend.mocks.values():
        replacement.assert_not_called()


@pytest.mark.parametrize(
    "filename",
    ["generic.pdf", "paper_434b99d68f495a91946c64b36b4602f9.pdf"],
)
def test_generic_routing_is_default_regardless_of_filename(backend, filename):
    session = application.build_pdf_session(filename)
    application.PaperSectionRetriever.assert_not_called()
    application.RerankedRetriever.assert_called_once_with(
        application.HybridRetriever.return_value,
        application.Reranker.return_value,
    )
    application.AbstractSectionRetriever.assert_called_once_with(
        application.RerankedRetriever.return_value,
        application.Reranker.return_value,
        [backend.chunk],
        report=application._discard_progress,
    )
    application.RAGPipeline.assert_called_once_with(
        application.AbstractSectionRetriever.return_value,
        application.AnswerGenerator.return_value,
        top_k=3,
    )
    assert session.answer("Summarize the conclusion.") is backend.answer


def test_explicit_paper_profile_and_conclusion_abstention(backend):
    report = Mock()
    session = application.build_pdf_session("profiled.pdf", use_paper_profile=True, report=report)
    application.PaperSectionRetriever.assert_called_once_with(
        application.RerankedRetriever.return_value,
        application.Reranker.return_value,
        [backend.chunk],
        report=report,
    )
    application.AbstractSectionRetriever.assert_not_called()
    application.RAGPipeline.assert_called_once_with(
        application.PaperSectionRetriever.return_value,
        application.AnswerGenerator.return_value,
        top_k=3,
    )
    answer = session.answer("Summarize the conclusion.")
    assert answer.status == "abstained"
    assert "No explicitly titled Conclusion section" in answer.text
    backend.pipeline.answer.assert_not_called()
    assert session.answer("Summarize the discussion.") is backend.answer


def test_models_configuration_and_progress_callback(backend, capsys):
    messages = []
    application.build_pdf_session(
        "generic.pdf",
        model="qwen2.5:7b",
        embedding_device="cuda",
        embedding_revision="test-commit",
        report=messages.append,
    )
    application.SentenceTransformerEmbedder.assert_called_once_with(
        device="cuda", revision="test-commit"
    )
    application.OllamaLLM.assert_called_once_with(model="qwen2.5:7b")
    application.CrossEncoderScorer.assert_called_once_with(device="cpu")
    application.AnswerGenerator.assert_called_once_with(
        application.OllamaLLM.return_value,
        max_evidence_chars=2400,
        verification_llm=application.OllamaLLM.return_value,
    )
    assert any(message.startswith("Reading:") for message in messages)
    assert any(message.startswith("Index ready in") for message in messages)
    assert capsys.readouterr().out == ""


def test_no_usable_text_fails_before_loading_models(backend):
    application.chunk_pages.return_value = [SimpleNamespace(text="   ")]
    with pytest.raises(ValueError, match="does not perform OCR"):
        application.build_pdf_session("empty.pdf")
    application.SentenceTransformerEmbedder.assert_not_called()
    application.OllamaLLM.assert_not_called()
    application.RAGPipeline.assert_not_called()


def test_cache_hit_skips_ingestion_and_document_embedding(backend):
    application.build_pdf_session(
        "generic.pdf", cache_dir="data/indexes", embedding_revision="test-commit"
    )
    application._cache_compatibility.assert_called_once_with(
        Path("generic.pdf"), revision="test-commit", device="cpu"
    )
    application._cache_path.assert_called_once_with(Path("data/indexes"), {"source": "generic.pdf"})
    application.load_index.assert_called_once_with(
        Path("mock-index.npz"), compatibility={"source": "generic.pdf"}
    )
    application.parse_pdf.assert_not_called()
    application.chunk_pages.assert_not_called()
    backend.embedder.embed.assert_not_called()
    application.save_index.assert_not_called()
    kwargs = application.HybridRetriever.call_args.kwargs
    assert kwargs["document_embeddings"] is backend.vectors
    assert application.HybridRetriever.call_args.args == (backend.stored.chunks,)


@pytest.mark.parametrize("rebuild", [False, True])
def test_cache_miss_or_forced_rebuild_embeds_and_saves(backend, rebuild):
    application.load_index.side_effect = FileNotFoundError
    application.build_pdf_session(
        "generic.pdf",
        cache_dir="data/indexes",
        embedding_revision="test-commit",
        rebuild_index=rebuild,
    )
    if rebuild:
        application.load_index.assert_not_called()
    else:
        application.load_index.assert_called_once()
    application.parse_pdf.assert_called_once()
    backend.embedder.embed.assert_called_once_with(["Usable document text."])
    application.save_index.assert_called_once()
    kwargs = application.save_index.call_args.kwargs
    assert kwargs["document_embeddings"] is backend.vectors
    assert kwargs["chunks"] == [backend.chunk]
    assert kwargs["compatibility"] == {"source": "generic.pdf"}


def test_cache_dimension_mismatch_rebuilds_with_same_embedder(backend):
    backend.embedder.dimension = 3
    application.build_pdf_session(
        "generic.pdf", cache_dir="data/indexes", embedding_revision="test-commit"
    )
    application.SentenceTransformerEmbedder.assert_called_once()
    application.parse_pdf.assert_called_once()
    backend.embedder.embed.assert_called_once()
    application.save_index.assert_called_once()
