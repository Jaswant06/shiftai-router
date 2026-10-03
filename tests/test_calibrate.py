"""Hardware calibration with a fake Ollama client, plus profile save/load."""

from shiftai.calibrate import HardwareProfile, calibrate
from shiftai.energy import PowerSampler
from shiftai.ollama import ChatResult


class FakeClient:
    def __init__(self):
        self.unloaded = []

    def tags(self):
        return [{"name": "tiny:1b", "size": 1, "details": {"family": "f", "parameter_size": "1B", "quantization_level": "Q4"}}]

    def show(self, name):
        return {"capabilities": ["completion"]}

    def ps(self):
        return [{"name": "tiny:1b", "size": 2_000_000_000}]

    def unload(self, name):
        self.unloaded.append(name)

    def chat(self, model, messages, max_tokens=512, seed=0, **kwargs):
        cold = max_tokens == 4
        return ChatResult(
            model=model, text="ok", total_s=1.0, load_s=2.5 if cold else 0.0,
            prompt_tokens=50, prompt_s=0.1, output_tokens=100, output_s=2.0, done_reason="stop",
        )


def test_calibrate_measures_speeds_and_round_trips(tmp_path):
    client = FakeClient()
    profile = calibrate(client=client, sampler=PowerSampler())
    cost = profile.models["tiny:1b"]
    assert client.unloaded == ["tiny:1b"]  # cold load is measured from an empty cache
    assert cost.load_s == 2.5
    assert cost.prompt_tps == 500
    assert cost.gen_tps == 50
    assert cost.watts is None  # no telemetry, so not invented
    assert cost.memory_gb == 2.0

    path = profile.save(tmp_path / "hw.json")
    assert HardwareProfile.load(path).models["tiny:1b"] == cost
