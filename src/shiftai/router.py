"""The runtime router: prompt in, model choice (and answer) out.

It combines the shipped capability predictor (which model can answer this?)
with the local hardware profile and the live machine state (what does each
model cost here, right now?).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .calibrate import HardwareProfile
from .discovery import loaded_models
from .features import embed, prompt_kind
from .ollama import ChatResult, OllamaClient
from .policy import choose, pick_target
from .predictor import CapabilityPredictor
from .system import ResourceState, resource_state

# Below this absolute predicted quality for the largest model, warn the user
# even though routing to it is still the best available choice.
WARN_BELOW = 0.5
CHARS_PER_TOKEN = 4.0


@dataclass
class RouterArtifact:
    """Everything learned offline that ships inside the package."""

    ladder: list[str]
    # One predictor per cluster count; each quality target uses the count that
    # saved the most on development data, with its own tuned delta.
    predictors: dict[int, CapabilityPredictor]
    targets: dict[float, dict]  # tau -> {"clusters": k, "delta": d}
    expected_tokens: dict[str, dict[str, float]]
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        data = {
            "ladder": self.ladder,
            "predictors": {str(k): p.to_dict() for k, p in self.predictors.items()},
            "targets": {str(t): v for t, v in self.targets.items()},
            "expected_tokens": self.expected_tokens,
            "meta": self.meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RouterArtifact":
        if path is None:
            source = resources.files("shiftai") / "artifacts" / "router.json"
            if not source.is_file():
                raise FileNotFoundError(
                    "No trained router is bundled with this install yet. Train one with "
                    "scripts/train_router.py or pass --artifact."
                )
            data = json.loads(source.read_text())
        else:
            data = json.loads(Path(path).read_text())
        return cls(
            ladder=data["ladder"],
            predictors={int(k): CapabilityPredictor.from_dict(p) for k, p in data["predictors"].items()},
            targets={float(t): v for t, v in data["targets"].items()},
            expected_tokens=data["expected_tokens"],
            meta=data.get("meta", {}),
        )


@dataclass
class Decision:
    """Which model was picked, why, and what the router believed."""

    model: str
    reason: str
    quality_target: float
    predicted: dict[str, float]
    relative: dict[str, float]
    est_latency_s: dict[str, float]
    est_energy_j: dict[str, float | None]
    loaded: list[str]
    novelty: str
    optimizing: str
    warning: str | None
    overhead_ms: float


class Router:
    def __init__(
        self,
        artifact: RouterArtifact | None = None,
        profile: HardwareProfile | None = None,
        client: OllamaClient | None = None,
    ):
        self.client = client or OllamaClient()
        self.artifact = artifact or RouterArtifact.load()
        self.profile = profile or HardwareProfile.load()
        self.models = [m for m in self.artifact.ladder if m in self.profile.models]
        if not self.models:
            raise RuntimeError(
                "None of the router's models are installed and calibrated. Router models: "
                f"{', '.join(self.artifact.ladder)}. Pull them with `ollama pull <model>` "
                "and run `shiftai setup`."
            )
        self.largest = self.models[-1]

    def estimate(self, model: str, prompt: str, loaded: set[str]) -> tuple[float, float | None]:
        """Expected seconds and joules for this model to answer this prompt now."""
        cost = self.profile.models[model]
        kind = prompt_kind(prompt)
        out_tokens = self.artifact.expected_tokens.get(model, {}).get(kind, 256.0)
        prompt_tokens = len(prompt) / CHARS_PER_TOKEN
        seconds = 0.0 if model in loaded else cost.load_s
        if cost.prompt_tps > 0:
            seconds += prompt_tokens / cost.prompt_tps
        if cost.gen_tps > 0:
            seconds += out_tokens / cost.gen_tps
        joules = cost.watts * seconds if cost.watts is not None else None
        return seconds, joules

    def decide(self, prompt: str, quality: float = 95, state: ResourceState | None = None) -> Decision:
        start = time.perf_counter()
        state = state or resource_state(cpu_sample_s=0.0)
        loaded = loaded_models(self.client)

        target = pick_target(self.artifact.targets, quality)
        predictor = self.artifact.predictors[target["clusters"]]
        vectors = embed([prompt], self.client)
        row = predictor.predict(vectors, [prompt])[0]
        probs = {m: float(row[predictor.models.index(m)]) for m in self.models}
        novelty = predictor.novelty(vectors, [prompt])[0]

        latency, energy = {}, {}
        for m in self.models:
            latency[m], energy[m] = self.estimate(m, prompt, loaded)

        # On battery, rank models by energy when it is known; otherwise by time.
        by_energy = state.on_battery and all(e is not None for e in energy.values())
        costs = energy if by_energy else latency
        if quality >= 100:
            model, reason = self.largest, "quality target 100%: always the largest model"
        else:
            model, reason = choose(probs, costs, self.largest, target["delta"], novelty)

        reference = max(probs[self.largest], 1e-9)
        warning = None
        if probs[self.largest] < WARN_BELOW:
            warning = "Low confidence: even the largest local model may struggle with this prompt."

        return Decision(
            model=model,
            reason=reason,
            quality_target=quality,
            predicted=probs,
            relative={m: probs[m] / reference for m in self.models},
            est_latency_s=latency,
            est_energy_j=energy,
            loaded=sorted(loaded),
            novelty=novelty,
            optimizing="energy" if by_energy else "latency",
            warning=warning,
            overhead_ms=(time.perf_counter() - start) * 1000,
        )

    def ask(self, prompt: str, quality: float = 95, max_tokens: int = 1024) -> tuple[Decision, ChatResult]:
        decision = self.decide(prompt, quality)
        result = self.client.chat(decision.model, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
        return decision, result


def explain(decision: Decision) -> str:
    """Human-readable table of the routing decision."""
    lines = [f"Quality target: {decision.quality_target:g}% of the largest model ({decision.optimizing} optimised)"]
    lines.append(f"{'model':24} {'pred. quality':>13} {'relative':>9} {'est. time':>10} {'est. energy':>12}  loaded")
    for m, p in decision.predicted.items():
        e = decision.est_energy_j[m]
        lines.append(
            f"{m:24} {p:13.2f} {decision.relative[m]:9.0%} {decision.est_latency_s[m]:9.2f}s "
            f"{(f'{e:.1f} J' if e is not None else 'n/a'):>12}  {'yes' if m in decision.loaded else ''}"
        )
    lines.append(f"Selected: {decision.model}  ({decision.reason})")
    lines.append(f"Prompt familiarity: {decision.novelty}   Router overhead: {decision.overhead_ms:.0f} ms")
    if decision.warning:
        lines.append(f"Warning: {decision.warning}")
    return "\n".join(lines)
