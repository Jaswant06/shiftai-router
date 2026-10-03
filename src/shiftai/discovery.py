"""Find the language models installed in the local Ollama runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ollama import OllamaClient

_SIZE_UNITS = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}


@dataclass
class ModelInfo:
    """What ShiftAI knows about one installed model before benchmarking it."""

    name: str
    size_bytes: int
    family: str
    params_b: float | None
    quantization: str
    capabilities: list[str]

    @property
    def can_chat(self) -> bool:
        return "completion" in self.capabilities


def parse_params(parameter_size: str) -> float | None:
    """Turn Ollama's parameter_size string ("9.7B", "137M") into billions."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([KMBT])\s*", parameter_size or "", re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()]


def list_models(client: OllamaClient | None = None) -> list[ModelInfo]:
    """All installed models, smallest first by parameter count."""
    client = client or OllamaClient()
    models = []
    for entry in client.tags():
        details = entry.get("details", {})
        capabilities = client.show(entry["name"]).get("capabilities", [])
        models.append(
            ModelInfo(
                name=entry["name"],
                size_bytes=entry.get("size", 0),
                family=details.get("family", ""),
                params_b=parse_params(details.get("parameter_size", "")),
                quantization=details.get("quantization_level", ""),
                capabilities=capabilities,
            )
        )
    return sorted(models, key=lambda m: (m.params_b is None, m.params_b or 0.0, m.name))


def chat_models(client: OllamaClient | None = None) -> list[ModelInfo]:
    """Installed models that can answer prompts (skips embedding-only models)."""
    return [m for m in list_models(client) if m.can_chat]


def loaded_models(client: OllamaClient | None = None) -> set[str]:
    """Names of the models currently held in memory by Ollama."""
    client = client or OllamaClient()
    return {entry["name"] for entry in client.ps()}
