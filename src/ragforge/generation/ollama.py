"""Synchronous Ollama backend using the Python standard library."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class OllamaError(RuntimeError):
    """Ollama communication failed or its API response was invalid."""


class OllamaLLM:
    """Implement the generation LLM protocol through Ollama's chat API.

    Answer-contract validation belongs to AnswerGenerator.
    Requests bypass environment proxies for direct local-server access.
    """

    def __init__(
        self,
        *,
        model: str = "qwen2.5:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 180,
        num_ctx: int = 4096,
        num_predict: int = 512,
        keep_alive: str = "10m",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string.")
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("base_url must start with http:// or https://.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number.")
        for name, value in (("num_ctx", num_ctx), ("num_predict", num_predict)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("keep_alive must be a nonempty string.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self._opener = build_opener(ProxyHandler({}))

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return model text; raise OllamaError for backend failures."""
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise TypeError("system_prompt and user_prompt must be strings.")

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.code} for model {self.model!r}. "
                "Check that the model is installed and inspect the Ollama logs."
            ) from exc
        except (URLError, OSError) as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError("Ollama returned invalid API JSON.") from exc

        if not isinstance(data, dict):
            raise OllamaError("Ollama API response must be an object.")
        if data.get("error"):
            raise OllamaError(f"Ollama reported an error: {data['error']}")
        if data.get("done") is not True:
            raise OllamaError("Ollama returned an incomplete response.")
        if data.get("done_reason") == "length":
            raise OllamaError(
                "Ollama reached the output limit. Increase num_predict."
            )
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response is missing its message object.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned empty or invalid message content.")
        return content
