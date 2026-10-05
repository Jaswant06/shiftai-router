"""Model discovery against a fake Ollama client (no server needed)."""

import httpx
import pytest

from shiftai.discovery import chat_models, list_models, parse_params
from shiftai.ollama import OllamaClient, OllamaUnavailable


class FakeClient:
    def tags(self):
        return [
            {"name": "big:9b", "size": 6_000, "details": {"family": "qwen35", "parameter_size": "9.7B", "quantization_level": "Q4_K_M"}},
            {"name": "embed:latest", "size": 300, "details": {"family": "nomic-bert", "parameter_size": "137M", "quantization_level": "F16"}},
            {"name": "small:0.8b", "size": 1_000, "details": {"family": "qwen35", "parameter_size": "0.8B", "quantization_level": "Q4_K_M"}},
        ]

    def show(self, name):
        return {"capabilities": ["embedding"] if name.startswith("embed") else ["completion"]}


def test_parse_params():
    assert parse_params("9.7B") == 9.7
    assert parse_params("137M") == pytest.approx(0.137)
    assert parse_params("") is None


def test_models_sorted_by_size_and_embedders_excluded():
    names = [m.name for m in list_models(FakeClient())]
    assert names == ["embed:latest", "small:0.8b", "big:9b"]
    assert [m.name for m in chat_models(FakeClient())] == ["small:0.8b", "big:9b"]


def test_friendly_error_when_ollama_is_down():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    client = OllamaClient(host="http://localhost:1")
    client._http = httpx.Client(base_url=client.host, transport=httpx.MockTransport(refuse))
    with pytest.raises(OllamaUnavailable, match="Is it running"):
        client.tags()


def test_friendly_error_when_model_missing():
    from shiftai.ollama import ModelNotInstalled

    def not_found(request):
        return httpx.Response(404, json={"error": "model 'ghost' not found"})

    client = OllamaClient(host="http://localhost:1")
    client._http = httpx.Client(base_url=client.host, transport=httpx.MockTransport(not_found))
    with pytest.raises(ModelNotInstalled, match="ollama pull ghost"):
        client.embed("ghost", ["hi"])
    with pytest.raises(ModelNotInstalled, match="ollama pull ghost"):
        list(client.chat_stream("ghost", [{"role": "user", "content": "hi"}]))
