"""Router decisions end to end, with a fake Ollama client and a hand-built predictor."""

import numpy as np
import pytest

from shiftai.calibrate import HardwareProfile, ModelCost
from shiftai.predictor import CapabilityPredictor, KindClusters
from shiftai.features import prompt_kind
from shiftai.router import Router, RouterArtifact, explain
from shiftai.system import ResourceState

class FakeClient:
    def __init__(self, loaded=(), vector=(1.0, 0.0, 0.0, 0.0)):
        self.loaded = list(loaded)
        self.vector = list(vector)

    def embed(self, model, texts):
        return [self.vector for _ in texts]

    def ps(self):
        return [{"name": n} for n in self.loaded]


def _artifact(p_small, p_large):
    # One cluster per prompt type, sitting exactly where the fake embedding lands.
    cluster = KindClusters(
        centroids=np.array([[1.0, 0.0, 0.0, 0.0]]),
        accuracy=np.array([[p_small, p_large]]),
        medium=0.5,
        low=0.2,
    )
    predictor = CapabilityPredictor(["small", "large"], {"open": cluster, "choice": cluster})
    return RouterArtifact(
        ladder=["small", "large"],
        predictors={1: predictor},
        targets={0.9: {"clusters": 1, "delta": 0.2}, 0.95: {"clusters": 1, "delta": 0.1}, 1.0: {"clusters": 1, "delta": 0.0}},
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


def test_unfamiliar_prompt_goes_to_largest_model():
    far_away = FakeClient(vector=(0.0, 1.0, 0.0, 0.0))
    decision = Router(_artifact(0.85, 0.90), _profile(), far_away).decide("hi", state=PLUGGED_IN)
    assert decision.novelty == "low"
    assert decision.model == "large"


def test_artifact_round_trip(tmp_path):
    artifact = _artifact(0.85, 0.90)
    artifact.save(tmp_path / "router.json")
    restored = RouterArtifact.load(tmp_path / "router.json")
    assert restored.targets == artifact.targets
    router = Router(restored, _profile(), FakeClient())
    assert router.decide("What is 2 + 2?", state=PLUGGED_IN).model == "small"


def test_quality_100_always_uses_largest():
    router = Router(_artifact(0.99, 0.90), _profile(), FakeClient())
    assert router.decide("What is 2 + 2?", quality=100, state=PLUGGED_IN).model == "large"
