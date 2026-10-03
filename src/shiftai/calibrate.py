"""Measure what each installed model costs on this particular machine.

This is the hardware half of ShiftAI's design. Whether a model can answer a
prompt does not depend on the computer, but how long it takes and how much
energy it uses does, so `shiftai setup` measures it locally:

    load_s      cold start: reading the model from disk into memory
    prompt_tps  prompt tokens processed per second
    gen_tps     output tokens generated per second
    watts       average power while generating (when telemetry is available)
    memory_gb   memory the loaded model occupies
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .discovery import ModelInfo, chat_models
from .energy import PowerSampler
from .ollama import OllamaClient
from .system import machine_info

CONFIG_DIR = Path.home() / ".shiftai"
PROFILE_PATH = CONFIG_DIR / "hardware.json"

# A fixed, medium-length prompt so speeds are comparable across models.
CALIBRATION_PROMPT = (
    "Explain, in about one hundred words, why the sky looks blue during the day "
    "and red or orange at sunset. Mention how the angle of sunlight and the "
    "distance light travels through the atmosphere affect what we see."
)


@dataclass
class ModelCost:
    """Measured cost profile of one model on this machine."""

    name: str
    params_b: float | None
    family: str
    quantization: str
    load_s: float
    prompt_tps: float
    gen_tps: float
    watts: float | None
    memory_gb: float | None


@dataclass
class HardwareProfile:
    machine: dict
    energy_source: str
    created: str
    models: dict[str, ModelCost]

    def save(self, path: Path = PROFILE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = PROFILE_PATH) -> "HardwareProfile":
        if not path.exists():
            raise FileNotFoundError(f"No hardware profile at {path}. Run `shiftai setup` first.")
        data = json.loads(path.read_text())
        data["models"] = {name: ModelCost(**m) for name, m in data["models"].items()}
        return cls(**data)


def _memory_gb(client: OllamaClient, model: str) -> float | None:
    for entry in client.ps():
        if entry["name"] == model:
            return entry.get("size", 0) / 1e9
    return None


def calibrate_model(
    model: ModelInfo,
    client: OllamaClient,
    sampler: PowerSampler,
    repeats: int = 3,
) -> ModelCost:
    """Cold-load the model once, then time a few warm generations."""
    client.unload(model.name)
    time.sleep(1.0)
    cold = client.chat(model.name, [{"role": "user", "content": "Reply with OK."}], max_tokens=4)

    prompt_tps, gen_tps, watts = [], [], []
    messages = [{"role": "user", "content": CALIBRATION_PROMPT}]
    for i in range(repeats):
        t0 = time.monotonic()
        result = client.chat(model.name, messages, max_tokens=160, seed=i)
        t1 = time.monotonic()
        if result.prompt_s > 0:
            prompt_tps.append(result.prompt_tokens / result.prompt_s)
        if result.output_s > 0:
            gen_tps.append(result.output_tokens / result.output_s)
        w = sampler.mean_watts(t0, t1)
        if w is not None:
            watts.append(w)

    return ModelCost(
        name=model.name,
        params_b=model.params_b,
        family=model.family,
        quantization=model.quantization,
        load_s=cold.load_s,
        prompt_tps=statistics.median(prompt_tps) if prompt_tps else 0.0,
        gen_tps=statistics.median(gen_tps) if gen_tps else 0.0,
        watts=statistics.median(watts) if watts else None,
        memory_gb=_memory_gb(client, model.name),
    )


def calibrate(
    client: OllamaClient | None = None,
    sampler: PowerSampler | None = None,
    models: list[str] | None = None,
    on_model: Callable[[ModelCost], None] | None = None,
) -> HardwareProfile:
    """Calibrate every installed chat model (or just the named ones)."""
    client = client or OllamaClient()
    sampler = sampler or PowerSampler()
    candidates = [m for m in chat_models(client) if models is None or m.name in models]
    costs = {}
    with sampler:
        for model in candidates:
            cost = calibrate_model(model, client, sampler)
            costs[model.name] = cost
            if on_model:
                on_model(cost)
    return HardwareProfile(
        machine=machine_info(),
        energy_source=sampler.source,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        models=costs,
    )
