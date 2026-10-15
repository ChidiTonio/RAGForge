"""Test the PDF CLI without reading PDFs or constructing real models."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def cli(monkeypatch):
    from ragforge import application as backend

    script = Path(__file__).resolve().parents[1] / "scripts" / "ask_pdf.py"
    spec = importlib.util.spec_from_file_location("ragforge_ask_pdf_cli_tests", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = SimpleNamespace(has_text=True)
    chunk = SimpleNamespace(text="Usable document text.")
    answer = SimpleNamespace(
        status="answered", text="A supported point.", citations=(), evidence=()
    )
    pipeline = Mock()
    pipeline.answer.return_value = answer
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
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(backend, name, replacement)
    return SimpleNamespace(
        module=module, pipeline=pipeline, replacements=replacements, backend=backend
    )


def set_arguments(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["ask_pdf.py", *arguments])


@pytest.mark.parametrize(
    ("count", "message"),
    [
        ("0", "--point-count must be greater than zero"),
        ("-2", "--point-count must be greater than zero"),
        ("not-an-integer", "invalid int value"),
    ],
)
def test_invalid_count_exits_before_loading(cli, monkeypatch, capsys, count, message):
    set_arguments(monkeypatch, "/tmp/not-needed.pdf", "--point-count", count)
    with pytest.raises(SystemExit) as exc:
        cli.module.main()
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert message in output.err
    assert "Reading:" not in output.out
    for replacement in cli.replacements.values():
        replacement.assert_not_called()
    cli.pipeline.answer.assert_not_called()


@pytest.mark.parametrize("count", [None, 2])
def test_single_question_forwards_count(cli, monkeypatch, capsys, count):
    arguments = ["generic.pdf", "Summarize this document."]
    if count is not None:
        arguments.extend(["--point-count", str(count)])
    set_arguments(monkeypatch, *arguments)
    input_mock = Mock(side_effect=AssertionError("Unexpected interactive input"))
    monkeypatch.setattr("builtins.input", input_mock)
    cli.module.main()
    cli.pipeline.answer.assert_called_once_with("Summarize this document.", point_count=count)
    cli.backend.parse_pdf.assert_called_once_with(Path("generic.pdf"))
    cli.backend.PaperSectionRetriever.assert_not_called()
    input_mock.assert_not_called()
    assert "STATUS: answered" in capsys.readouterr().out


@pytest.mark.parametrize("count", [None, 3])
def test_interactive_session_reuses_count(cli, monkeypatch, capsys, count):
    arguments = ["generic.pdf"]
    if count is not None:
        arguments.extend(["--point-count", str(count)])
    set_arguments(monkeypatch, *arguments)
    input_mock = Mock(side_effect=["First question?", "Second question?", "exit"])
    monkeypatch.setattr("builtins.input", input_mock)
    cli.module.main()
    assert cli.pipeline.answer.call_count == 2
    calls = cli.pipeline.answer.call_args_list
    assert calls[0].args == ("First question?",)
    assert calls[0].kwargs == {"point_count": count}
    assert calls[1].args == ("Second question?",)
    assert calls[1].kwargs == {"point_count": count}
    assert input_mock.call_count == 3
    output = capsys.readouterr().out
    assert output.count("STATUS: answered") == 2
    if count is not None:
        assert f"Supported answers must contain exactly {count} point(s)." in output
    else:
        assert "Supported answers must contain exactly" not in output


def test_help_exits_without_loading(cli, monkeypatch, capsys):
    set_arguments(monkeypatch, "--help")
    with pytest.raises(SystemExit) as exc:
        cli.module.main()
    assert exc.value.code == 0
    output = capsys.readouterr()
    assert "--point-count N" in output.out
    assert "Reading:" not in output.out
    for replacement in cli.replacements.values():
        replacement.assert_not_called()


@pytest.fixture
def cached_cli(cli, monkeypatch):
    vectors = np.array([[3, 4]], dtype=np.float32)
    chunk = cli.backend.chunk_pages.return_value[0]
    stored = SimpleNamespace(chunks=(chunk,), document_embeddings=vectors)
    embedder = cli.backend.SentenceTransformerEmbedder.return_value
    embedder.dimension = 2
    embedder.embed.return_value = vectors
    replacements = {
        "_cache_compatibility": Mock(return_value={"source": "generic.pdf"}),
        "hash_file": Mock(),
        "load_index": Mock(return_value=stored),
        "save_index": Mock(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(cli.backend, name, replacement)
    return SimpleNamespace(
        cli=cli, vectors=vectors, stored=stored, embedder=embedder, replacements=replacements
    )


def cache_arguments(monkeypatch, *extra):
    set_arguments(
        monkeypatch,
        "generic.pdf",
        "Summarize this document.",
        "--cache-dir",
        "data/indexes",
        "--embedding-revision",
        "test-commit",
        *extra,
    )


def test_cache_hit_skips_parsing_and_document_embedding(cached_cli, monkeypatch):
    cache_arguments(monkeypatch)
    cli = cached_cli.cli
    cli.module.main()
    cli.backend.parse_pdf.assert_not_called()
    cli.backend.chunk_pages.assert_not_called()
    cached_cli.embedder.embed.assert_not_called()
    cli.backend.save_index.assert_not_called()
    call = cli.backend.HybridRetriever.call_args
    assert call.args == (cached_cli.stored.chunks,)
    assert call.kwargs["document_embeddings"] is cached_cli.vectors
    cli.backend.SentenceTransformerEmbedder.assert_called_once_with(
        device="cpu", revision="test-commit"
    )
    cli.pipeline.answer.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing"), ValueError("placeholder"), PermissionError("unreadable")],
)
def test_cache_miss_or_invalid_rebuilds(cached_cli, monkeypatch, failure):
    cli = cached_cli.cli
    if type(failure) is ValueError:
        failure = cli.backend.InvalidIndexCache("malformed cache")
    cli.backend.load_index.side_effect = failure
    cache_arguments(monkeypatch)
    cli.module.main()
    cli.backend.parse_pdf.assert_called_once()
    cached_cli.embedder.embed.assert_called_once_with(["Usable document text."])
    cli.backend.save_index.assert_called_once()
    assert cli.backend.HybridRetriever.call_args.kwargs["document_embeddings"] is cached_cli.vectors


def test_explicit_rebuild_bypasses_loading(cached_cli, monkeypatch):
    cache_arguments(monkeypatch, "--rebuild-index")
    cli = cached_cli.cli
    cli.module.main()
    cli.backend.load_index.assert_not_called()
    cached_cli.embedder.embed.assert_called_once()
    cli.backend.save_index.assert_called_once()


def test_cache_dimension_mismatch_rebuilds(cached_cli, monkeypatch):
    cached_cli.embedder.dimension = 3
    cache_arguments(monkeypatch)
    cli = cached_cli.cli
    cli.module.main()
    cli.backend.parse_pdf.assert_called_once()
    cli.backend.SentenceTransformerEmbedder.assert_called_once()
    cli.backend.save_index.assert_called_once()


def test_cache_write_failure_still_answers(cached_cli, monkeypatch, capsys):
    cli = cached_cli.cli
    cli.backend.load_index.side_effect = FileNotFoundError()
    cli.backend.save_index.side_effect = OSError("disk full")
    cache_arguments(monkeypatch)
    cli.module.main()
    assert "could not save index cache" in capsys.readouterr().out
    cli.pipeline.answer.assert_called_once()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--rebuild-index"],
        ["--embedding-revision", "   "],
        ["--cache-dir", "data/indexes", "--point-count", "0"],
    ],
)
def test_invalid_cache_arguments_exit_before_io(cached_cli, monkeypatch, arguments):
    set_arguments(monkeypatch, "generic.pdf", *arguments)
    with pytest.raises(SystemExit) as exc:
        cached_cli.cli.module.main()
    assert exc.value.code == 2
    for replacement in cached_cli.cli.replacements.values():
        replacement.assert_not_called()
    for replacement in cached_cli.replacements.values():
        replacement.assert_not_called()


def test_no_cache_does_not_hash_or_load(cached_cli, monkeypatch):
    set_arguments(monkeypatch, "generic.pdf", "Question?")
    cli = cached_cli.cli
    cli.module.main()
    for replacement in cached_cli.replacements.values():
        replacement.assert_not_called()
    cached_cli.embedder.embed.assert_not_called()
    assert "document_embeddings" not in cli.backend.HybridRetriever.call_args.kwargs


def test_cache_without_revision_warns(cached_cli, monkeypatch, capsys):
    set_arguments(monkeypatch, "generic.pdf", "Question?", "--cache-dir", "data/indexes")
    cached_cli.cli.module.main()
    assert "revision is unpinned" in capsys.readouterr().out


def test_cache_identity_includes_content_source_and_configuration(cli, monkeypatch, tmp_path):
    monkeypatch.setattr(cli.backend, "hash_file", Mock(return_value="first-hash"))
    monkeypatch.setattr(cli.backend, "version", Mock(return_value="test-version"))
    first = cli.backend._cache_compatibility(Path("first.pdf"), revision="commit-a", device="cpu")
    assert first["source"] == "first.pdf"
    assert first["chunk_size"] == 800
    assert first["overlap"] == 120
    path = cli.backend._cache_path(tmp_path, first)
    assert path == cli.backend._cache_path(tmp_path, dict(reversed(list(first.items()))))
    for key in (
        "source",
        "pdf_sha256",
        "embedding_revision",
        "embedding_device",
        "ingestion_version",
        "dependencies",
    ):
        changed = dict(first, **{key: "changed"})
        assert cli.backend._cache_path(tmp_path, changed) != path


@pytest.mark.parametrize("profiled", [False, True])
def test_retriever_selection_by_filename(cli, monkeypatch, profiled):
    filename = "paper_434b99d68f495a91946c64b36b4602f9.pdf" if profiled else "generic.pdf"
    set_arguments(monkeypatch, filename, "Summarize this document.")
    cli.module.main()
    reranked = cli.backend.RerankedRetriever.return_value
    cli.backend.RerankedRetriever.assert_called_once_with(
        cli.backend.HybridRetriever.return_value, cli.backend.Reranker.return_value
    )
    if profiled:
        cli.backend.PaperSectionRetriever.assert_called_once_with(
            reranked,
            cli.backend.Reranker.return_value,
            cli.backend.chunk_pages.return_value,
            report=cli.module._report_progress,
        )
        cli.backend.AbstractSectionRetriever.assert_not_called()
        active = cli.backend.PaperSectionRetriever.return_value
    else:
        cli.backend.PaperSectionRetriever.assert_not_called()
        cli.backend.AbstractSectionRetriever.assert_called_once_with(
            reranked,
            cli.backend.Reranker.return_value,
            cli.backend.chunk_pages.return_value,
            report=cli.module._report_progress,
        )
        active = cli.backend.AbstractSectionRetriever.return_value
    cli.backend.RAGPipeline.assert_called_once_with(
        active, cli.backend.AnswerGenerator.return_value, top_k=3
    )


@pytest.mark.parametrize("profiled", [False, True])
def test_conclusion_abstention_is_profile_specific(cli, monkeypatch, capsys, profiled):
    filename = "paper_434b99d68f495a91946c64b36b4602f9.pdf" if profiled else "generic.pdf"
    question = "Summarize the conclusion."
    set_arguments(monkeypatch, filename, question)
    cli.module.main()
    if profiled:
        cli.pipeline.answer.assert_not_called()
        output = capsys.readouterr().out
        assert "STATUS: abstained" in output
        assert "No explicitly titled Conclusion section" in output
    else:
        cli.pipeline.answer.assert_called_once_with(question, point_count=None)
        assert "STATUS: answered" in capsys.readouterr().out


def test_cli_forwards_configuration_to_shared_builder(cli, monkeypatch):
    session = Mock()
    session.answer.return_value = cli.pipeline.answer.return_value
    builder = Mock(return_value=session)
    monkeypatch.setattr(cli.module, "build_pdf_session", builder)
    set_arguments(
        monkeypatch,
        "generic.pdf",
        "Question?",
        "--model",
        "qwen2.5:7b",
        "--embedding-device",
        "cuda",
        "--cache-dir",
        "data/indexes",
        "--embedding-revision",
        "test-commit",
        "--rebuild-index",
        "--point-count",
        "2",
    )
    cli.module.main()
    builder.assert_called_once_with(
        Path("generic.pdf"),
        model="qwen2.5:7b",
        embedding_device="cuda",
        cache_dir=Path("data/indexes"),
        rebuild_index=True,
        embedding_revision="test-commit",
        use_paper_profile=False,
        report=cli.module._report_progress,
    )
    builder.return_value.answer.assert_called_once_with("Question?", point_count=2)
    for replacement in cli.replacements.values():
        replacement.assert_not_called()


def test_cli_no_text_error_exits_cleanly(cli, monkeypatch, capsys):
    cli.backend.chunk_pages.return_value = []
    set_arguments(monkeypatch, "empty.pdf", "Question?")
    with pytest.raises(SystemExit) as exc:
        cli.module.main()
    assert exc.value.code == 1
    assert "does not perform OCR" in capsys.readouterr().err
    cli.backend.SentenceTransformerEmbedder.assert_not_called()
    cli.pipeline.answer.assert_not_called()
