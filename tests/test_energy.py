"""Energy integration and telemetry parsing, without real hardware."""

import pytest

from shiftai.energy import PowerSampler, PowermetricsSampler


def test_energy_is_mean_power_times_duration():
    sampler = PowerSampler()
    for t, watts in [(0.0, 10), (1.0, 20), (2.0, 30), (3.0, 40)]:
        sampler.add_sample(t, watts)
    assert sampler.energy_j(1.0, 2.0) == pytest.approx(25 * 1.0)
    assert sampler.mean_watts(0.0, 3.0) == pytest.approx(25)


def test_short_window_uses_nearest_sample():
    sampler = PowerSampler()
    sampler.add_sample(0.0, 10)
    sampler.add_sample(1.0, 20)
    assert sampler.energy_j(0.4, 0.5) == pytest.approx(20 * 0.1)


def test_no_samples_means_not_measured():
    assert PowerSampler().energy_j(0, 1) is None


def test_powermetrics_line_parsing():
    sampler = PowermetricsSampler()
    assert sampler.parse("Combined Power (CPU + GPU + ANE): 12345 mW") == pytest.approx(12.345)
    assert sampler.parse("CPU Power: 900 mW") is None
