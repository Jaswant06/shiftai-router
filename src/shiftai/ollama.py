"""Thin client for the local Ollama HTTP API.

Only the endpoints ShiftAI needs: list installed models, inspect one model,
see which models are loaded in memory, chat, and embed. Timing numbers come
straight from Ollama's own response fields, which are measured server-side and
are more precise than timing the HTTP round trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = f"http://{DEFAULT_HOST}"

NS_PER_S = 1e9


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


@dataclass
class ChatResult:
    """A model's reply plus Ollama's server-side timing breakdown."""

    model: str
    text: str
    total_s: float
    load_s: float
    prompt_tokens: int
    prompt_s: float
    output_tokens: int
    output_s: float
    done_reason: str

    @property
    def ttft_s(self) -> float:
        """Time to first token: model load plus reading the prompt."""
        return self.load_s + self.prompt_s

    @property
    def truncated(self) -> bool:
        """True when generation stopped because it hit the token limit."""
        return self.done_reason == "length"


class OllamaClient:
    """Synchronous client for one Ollama server."""

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 600.0):
        self.host = host.rstrip("/")
        self._http = httpx.Client(base_url=self.host, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise OllamaUnavailable(
                f"Ollama not reachable at {self.host}. Is it running? "
                "Start it with `ollama serve` or open the Ollama app."
            ) from exc
        response.raise_for_status()
        return response.json()

    def tags(self) -> list[dict]:
        """Installed models, as returned by GET /api/tags."""
        return self._request("GET", "/api/tags").get("models", [])

    def show(self, model: str) -> dict:
        """Details for one model, including its capabilities."""
        return self._request("POST", "/api/show", json={"model": model})

    def ps(self) -> list[dict]:
        """Models currently loaded in memory, as returned by GET /api/ps."""
        return self._request("GET", "/api/ps").get("models", [])

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 0,
        keep_alive: str | None = None,
    ) -> ChatResult:
        """Run one non-streaming chat turn with thinking turned off.

        Thinking is disabled so every model in the ladder is compared on the
        same footing; a thinking model would otherwise spend hidden tokens.
        """
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens},
        }
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        data = self._request("POST", "/api/chat", json=body)
        return ChatResult(
            model=model,
            text=data.get("message", {}).get("content", ""),
            total_s=data.get("total_duration", 0) / NS_PER_S,
            load_s=data.get("load_duration", 0) / NS_PER_S,
            prompt_tokens=data.get("prompt_eval_count", 0),
            prompt_s=data.get("prompt_eval_duration", 0) / NS_PER_S,
            output_tokens=data.get("eval_count", 0),
            output_s=data.get("eval_duration", 0) / NS_PER_S,
            done_reason=data.get("done_reason", ""),
        )

    def embed(self, model: str, texts: list[str], keep_alive: str = "30m") -> list[list[float]]:
        """Embed a batch of texts with an Ollama embedding model."""
        data = self._request(
            "POST", "/api/embed", json={"model": model, "input": texts, "keep_alive": keep_alive}
        )
        return data["embeddings"]

    def unload(self, model: str) -> None:
        """Evict a model from memory so the next call measures a cold load."""
        self._request("POST", "/api/generate", json={"model": model, "keep_alive": 0})

    def close(self) -> None:
        self._http.close()
