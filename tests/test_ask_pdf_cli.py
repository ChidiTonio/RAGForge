"""Test the PDF CLI without reading PDFs or constructing real models."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
