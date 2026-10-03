"""Router decisions end to end, with a fake Ollama client and a hand-built predictor."""

import numpy as np
import pytest

from shiftai.calibrate import HardwareProfile, ModelCost
from shiftai.predictor import CapabilityPredictor, ModelHead, Novelty
from shiftai.router import Router, RouterArtifact, explain, prompt_kind
from shiftai.system import ResourceState

DIM = 4


class FakeClient:
    def __init__(self, loaded=()):
        self.loaded = list(loaded)

    def embed(self, model, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def ps(self):
        return [{"name": n} for n in self.loaded]


def _head(p):
    # A constant head: zero weights, isotonic map pinned to probability p.
    return ModelHead(np.zeros(DIM + 1), 0.0, np.array([0.0, 1.0]), np.array([p, p]))


def _artifact(p_small, p_large):
    predictor = CapabilityPredictor(
        mean=np.zeros(DIM + 1),
        std=np.ones(DIM + 1),
        heads={"small": _head(p_small), "large": _head(p_large)},
        novelty=Novelty(np.array([[1.0, 0.0, 0.0, 0.0]]), medium=0.5, low=0.2),
    )
    return RouterArtifact(
        ladder=["small", "large"],
        predictor=predictor,
        deltas={0.9: 0.2, 0.95: 0.1, 1.0: 0.0},
        expected_tokens={"small": {"open": 100, "choice": 2}, "large": {"open": 100, "choice": 2}},
    )


def _profile(watts=None):
    def cost(name, load, gen):
        return ModelCost(name, None, "f", "q", load_s=load, prompt_tps=1000, gen_tps=gen, watts=watts, memory_gb=1)

    return HardwareProfile(machine={}, energy_source="none", created="", models={
        "small": cost("small", 1.0, 100.0),
        "large": cost("large", 5.0, 20.0),
    })


PLUGGED_IN = ResourceState(on_battery=False, battery_percent=90, cpu_percent=5, available_memory_gb=10, total_memory_gb=24)


def test_easy_prompt_goes_to_small_model():
    router = Router(_artifact(0.85, 0.90), _profile(), FakeClient())
    decision = router.decide("What is 2 + 2?", quality=95, state=PLUGGED_IN)
    assert decision.model == "small"
    assert decision.relative["large"] == 1.0
    assert decision.warning is None


def test_hard_prompt_goes_to_large_model():
    router = Router(_artifact(0.40, 0.90), _profile(), FakeClient())
    assert router.decide("Prove the theorem.", quality=95, state=PLUGGED_IN).model == "large"


def test_warning_comes_from_absolute_confidence():
    router = Router(_artifact(0.20, 0.30), _profile(), FakeClient())
    decision = router.decide("Something very hard", quality=95, state=PLUGGED_IN)
    assert decision.model == "large"
    assert "Low confidence" in decision.warning
    assert "Selected: large" in explain(decision)


def test_loaded_model_skips_load_time():
    router = Router(_artifact(0.85, 0.90), _profile(), FakeClient(loaded=["large"]))
    decision = router.decide("hi", state=PLUGGED_IN)
    cold = Router(_artifact(0.85, 0.90), _profile(), FakeClient()).decide("hi", state=PLUGGED_IN)
    assert decision.est_latency_s["large"] == pytest.approx(cold.est_latency_s["large"] - 5.0)


def test_battery_switches_to_energy_ranking():
    on_battery = ResourceState(True, 15, 5, 10, 24)
    decision = Router(_artifact(0.85, 0.90), _profile(watts=20), FakeClient()).decide("hi", state=on_battery)
    assert decision.optimizing == "energy"


def test_prompt_kind():
    assert prompt_kind("Pick one:\nA. red\nB. blue") == "choice"
    assert prompt_kind("Write me an email") == "open"
