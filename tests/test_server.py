"""OpenAI-compatible server, with a fake Ollama client and a hand-built router."""

import json

from fastapi.testclient import TestClient

from shiftai.ollama import ChatResult
from shiftai.router import Router
from shiftai.server import create_app, requested_quality, routing_prompt
from test_router import PLUGGED_IN, _artifact, _profile


class FakeOllama:
    def __init__(self):
        self.calls = []

    def embed(self, model, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def ps(self):
        return []

    def tags(self):
        return [{"name": "small"}, {"name": "large"}]

    def _result(self, model, text):
        return ChatResult(model, text, 1.0, 0.0, 5, 0.1, 3, 0.5, "stop")

    def chat(self, model, messages, **options):
        self.calls.append((model, messages, options))
        return self._result(model, f"answer from {model}")

    def chat_stream(self, model, messages, **options):
        self.calls.append((model, messages, options))
        for piece in ("answer ", "from ", model):
            yield piece
        yield self._result(model, f"answer from {model}")


def _client(p_small=0.85, p_large=0.90):
    ollama = FakeOllama()
    router = Router(_artifact(p_small, p_large), _profile(), ollama)
    router_decide = router.decide
    router.decide = lambda prompt, quality=95: router_decide(prompt, quality, state=PLUGGED_IN)
    return TestClient(create_app(router=router, client=ollama)), ollama


def test_routed_request_uses_cheap_model_and_says_so():
    client, ollama = _client()
    response = client.post("/v1/chat/completions", json={
        "model": "shiftai", "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    })
    body = response.json()
    assert response.status_code == 200
    assert body["model"] == "small"
    assert response.headers["X-ShiftAI-Model"] == "small"
    assert body["choices"][0]["message"]["content"] == "answer from small"
    assert body["usage"]["total_tokens"] == 8
    assert body["shiftai"]["quality_target"] == 90
    assert ollama.calls[0][1] == [{"role": "user", "content": "What is 2 + 2?"}]


def test_hard_prompt_goes_to_largest_and_quality_in_model_name():
    client, _ = _client(p_small=0.40)
    body = client.post("/v1/chat/completions", json={
        "model": "shiftai-99", "messages": [{"role": "user", "content": "Prove it."}],
    }).json()
    assert body["model"] == "large"
    assert body["shiftai"]["quality_target"] == 99


def test_direct_model_name_skips_routing():
    client, ollama = _client()
    body = client.post("/v1/chat/completions", json={
        "model": "large", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50,
    }).json()
    assert body["model"] == "large"
    assert "shiftai" not in body
    assert ollama.calls[0][2]["max_tokens"] == 50


def test_streaming_follows_openai_event_format():
    client, _ = _client()
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "shiftai", "stream": True, "messages": [{"role": "user", "content": "hi"}],
    }) as response:
        lines = [line for line in response.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line[6:]) for line in lines[:-1]]
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == "answer from small"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["shiftai"]["model"] == "small"


def test_models_stats_and_decide_endpoints():
    client, _ = _client()
    names = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "shiftai" in names and "shiftai-95" in names and "large" in names
    client.post("/v1/chat/completions", json={"model": "shiftai", "messages": [{"role": "user", "content": "hi"}]})
    stats = client.get("/shiftai/stats").json()
    assert stats["requests"] == 1 and stats["requests_by_model"] == {"small": 1}
    assert stats["estimated_seconds_saved_vs_largest"] > 0
    decision = client.post("/shiftai/decide", json={"prompt": "hi", "quality": 95}).json()
    assert decision["model"] == "small" and decision["quality_target"] == 95


def test_helpers():
    assert requested_quality("shiftai", {}, 90) == 90
    assert requested_quality("shiftai-85", {}, 90) == 85
    assert requested_quality("shiftai", {"quality": 99}, 90) == 99
    assert requested_quality("qwen3.5:4b", {}, 90) is None
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second question"},
    ]
    assert routing_prompt(messages) == "second question"


def test_empty_messages_rejected():
    client, _ = _client()
    assert client.post("/v1/chat/completions", json={"model": "shiftai", "messages": []}).status_code == 400
