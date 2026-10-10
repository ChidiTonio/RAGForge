"""Offline adapter tests: no running Ollama server is required."""

import io
import json
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from ragforge.generation.ollama import OllamaError, OllamaLLM


def install_response(monkeypatch, body):
    opener = Mock()
    opener.open.return_value = io.BytesIO(json.dumps(body).encode("utf-8"))
    monkeypatch.setattr(
        "ragforge.generation.ollama.build_opener", lambda *args: opener
    )
    return opener


def test_complete_returns_text_and_sends_expected_request(monkeypatch):
    content = '{"answer":"EUR 12 million","citations":["S1"],"abstain":false}'
    opener = install_response(
        monkeypatch, {"done": True, "message": {"content": content}}
    )
    llm = OllamaLLM(base_url="http://127.0.0.1:11434/")
    assert llm.complete(system_prompt="System", user_prompt="Evidence") == content

    request = opener.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert request.get_method() == "POST"
    assert opener.open.call_args.kwargs["timeout"] == 180
    payload = json.loads(request.data)
    assert payload["model"] == "qwen2.5:3b"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Evidence"},
    ]
    assert payload["options"]["temperature"] == 0


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"error": "model not found"},
        {"done": False, "message": {"content": "{}"}},
        {"done": True},
        {"done": True, "message": {"content": ""}},
        {"done": True, "message": {"content": 42}},
        {"done": True, "done_reason": "length", "message": {"content": "{}"}},
    ],
)
def test_invalid_api_response(monkeypatch, body):
    install_response(monkeypatch, body)
    with pytest.raises(OllamaError):
        OllamaLLM().complete(system_prompt="System", user_prompt="Question")


@pytest.mark.parametrize(
    "error",
    [
        URLError("connection refused"),
        TimeoutError("timed out"),
        HTTPError("http://localhost/api/chat", 404, "Not Found", {}, None),
    ],
)
def test_transport_errors(monkeypatch, error):
    opener = install_response(monkeypatch, {})
    opener.open.side_effect = error
    with pytest.raises(OllamaError) as caught:
        OllamaLLM().complete(system_prompt="System", user_prompt="Question")
    assert caught.value.__cause__ is error


def test_invalid_api_json(monkeypatch):
    opener = install_response(monkeypatch, {})
    opener.open.return_value = io.BytesIO(b"not JSON")
    with pytest.raises(OllamaError, match="invalid API JSON"):
        OllamaLLM().complete(system_prompt="System", user_prompt="Question")


def test_answer_validation_is_left_to_generator(monkeypatch):
    install_response(
        monkeypatch, {"done": True, "message": {"content": "not answer JSON"}}
    )
    assert OllamaLLM().complete(
        system_prompt="System", user_prompt="Question"
    ) == "not answer JSON"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"base_url": "localhost:11434"},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"num_ctx": True},
        {"num_predict": -1},
        {"keep_alive": ""},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OllamaLLM(**kwargs)
