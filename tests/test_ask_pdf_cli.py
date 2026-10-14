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
    script = Path(__file__).resolve().parents[1] / "scripts" / "ask_pdf.py"
    spec = importlib.util.spec_from_file_location("ragforge_ask_pdf_cli_tests", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Use a generic filename to avoid the paper-specific section profile.
    page = SimpleNamespace(has_text=True)
    chunk = SimpleNamespace(text="Usable document text.")
    answer = SimpleNamespace(
        status="answered",
        text="A supported point.",
        citations=(),
        evidence=(),
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
        "RAGPipeline": Mock(return_value=pipeline),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(module, name, replacement)

    return SimpleNamespace(
        module=module,
        pipeline=pipeline,
        replacements=replacements,
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

    # Single-question mode must not enter the interactive input loop.
    input_mock = Mock(side_effect=AssertionError("Unexpected interactive input"))
    monkeypatch.setattr("builtins.input", input_mock)

    cli.module.main()

    cli.pipeline.answer.assert_called_once_with(
        "Summarize this document.", point_count=count
    )
    cli.module.parse_pdf.assert_called_once_with(Path("generic.pdf"))
    cli.module.PaperSectionRetriever.assert_not_called()
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
    chunk = cli.module.chunk_pages.return_value[0]
    stored = SimpleNamespace(chunks=(chunk,), document_embeddings=vectors)
    embedder = cli.module.SentenceTransformerEmbedder.return_value
    embedder.dimension = 2
    embedder.embed.return_value = vectors
    replacements = {
        "_cache_compatibility": Mock(return_value={"source": "generic.pdf"}),
        "hash_file": Mock(),
        "load_index": Mock(return_value=stored),
        "save_index": Mock(),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(cli.module, name, replacement)
    return SimpleNamespace(
        cli=cli, vectors=vectors, stored=stored,
        embedder=embedder, replacements=replacements,
    )


def cache_arguments(monkeypatch, *extra):
    set_arguments(
        monkeypatch,
        "generic.pdf",
        "Summarize this document.",
        "--cache-dir", "data/indexes",
        "--embedding-revision", "test-commit",
        *extra,
    )


def test_cache_hit_skips_parsing_and_document_embedding(cached_cli, monkeypatch):
    cache_arguments(monkeypatch)
    cli = cached_cli.cli
    cli.module.main()

    cli.module.parse_pdf.assert_not_called()
    cli.module.chunk_pages.assert_not_called()
    cached_cli.embedder.embed.assert_not_called()
    cli.module.save_index.assert_not_called()
    call = cli.module.HybridRetriever.call_args
    assert call.args == (cached_cli.stored.chunks,)
    assert call.kwargs["document_embeddings"] is cached_cli.vectors
    cli.module.SentenceTransformerEmbedder.assert_called_once_with(
        device="cpu", revision="test-commit"
    )
    cli.pipeline.answer.assert_called_once()


@pytest.mark.parametrize("failure", [
    FileNotFoundError("missing"),
    ValueError("placeholder"),
    PermissionError("unreadable"),
])
def test_cache_miss_or_invalid_rebuilds(cached_cli, monkeypatch, failure):
    cli = cached_cli.cli
    if type(failure) is ValueError:
        failure = cli.module.InvalidIndexCache("malformed cache")
    cli.module.load_index.side_effect = failure
    cache_arguments(monkeypatch)
    cli.module.main()

    cli.module.parse_pdf.assert_called_once()
    cached_cli.embedder.embed.assert_called_once_with(["Usable document text."])
    cli.module.save_index.assert_called_once()
    assert (
        cli.module.HybridRetriever.call_args.kwargs["document_embeddings"]
        is cached_cli.vectors
    )


def test_explicit_rebuild_bypasses_loading(cached_cli, monkeypatch):
    cache_arguments(monkeypatch, "--rebuild-index")
    cli = cached_cli.cli
    cli.module.main()
    cli.module.load_index.assert_not_called()
    cached_cli.embedder.embed.assert_called_once()
    cli.module.save_index.assert_called_once()


def test_cache_dimension_mismatch_rebuilds(cached_cli, monkeypatch):
    cached_cli.embedder.dimension = 3
    cache_arguments(monkeypatch)
    cli = cached_cli.cli
    cli.module.main()
    cli.module.parse_pdf.assert_called_once()
    cli.module.SentenceTransformerEmbedder.assert_called_once()
    cli.module.save_index.assert_called_once()


def test_cache_write_failure_still_answers(cached_cli, monkeypatch, capsys):
    cli = cached_cli.cli
    cli.module.load_index.side_effect = FileNotFoundError()
    cli.module.save_index.side_effect = OSError("disk full")
    cache_arguments(monkeypatch)
    cli.module.main()
    assert "could not save index cache" in capsys.readouterr().out
    cli.pipeline.answer.assert_called_once()


@pytest.mark.parametrize("arguments", [
    ["--rebuild-index"],
    ["--embedding-revision", "   "],
    ["--cache-dir", "data/indexes", "--point-count", "0"],
])
def test_invalid_cache_arguments_exit_before_io(
    cached_cli, monkeypatch, arguments
):
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
    assert "document_embeddings" not in cli.module.HybridRetriever.call_args.kwargs


def test_cache_without_revision_warns(cached_cli, monkeypatch, capsys):
    set_arguments(
        monkeypatch, "generic.pdf", "Question?",
        "--cache-dir", "data/indexes",
    )
    cached_cli.cli.module.main()
    assert "revision is unpinned" in capsys.readouterr().out


def test_cache_identity_includes_content_source_and_configuration(
    cli, monkeypatch, tmp_path
):
    monkeypatch.setattr(cli.module, "hash_file", Mock(return_value="first-hash"))
    monkeypatch.setattr(cli.module, "version", Mock(return_value="test-version"))
    first = cli.module._cache_compatibility(
        Path("first.pdf"), revision="commit-a", device="cpu"
    )
    assert first["source"] == "first.pdf"
    assert first["chunk_size"] == 800
    assert first["overlap"] == 120
    path = cli.module._cache_path(tmp_path, first)
    assert path == cli.module._cache_path(tmp_path, dict(reversed(list(first.items()))))
    for key in (
        "source", "pdf_sha256", "embedding_revision",
        "embedding_device", "ingestion_version", "dependencies",
    ):
        changed = dict(first, **{key: "changed"})
        assert cli.module._cache_path(tmp_path, changed) != path
