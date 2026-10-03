"""Snapshot of the machine's current resources, used by the router and in run logs."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import psutil


@dataclass
class ResourceState:
    """What the router knows about the machine at decision time."""

    on_battery: bool
    battery_percent: float | None
    cpu_percent: float
    available_memory_gb: float
    total_memory_gb: float

    @property
    def low_battery(self) -> bool:
        return self.on_battery and self.battery_percent is not None and self.battery_percent < 20

    def as_dict(self) -> dict:
        return asdict(self)


def resource_state(cpu_sample_s: float = 0.1) -> ResourceState:
    battery = psutil.sensors_battery()
    memory = psutil.virtual_memory()
    return ResourceState(
        on_battery=bool(battery) and not battery.power_plugged,
        battery_percent=battery.percent if battery else None,
        cpu_percent=psutil.cpu_percent(interval=cpu_sample_s),
        available_memory_gb=memory.available / 1e9,
        total_memory_gb=memory.total / 1e9,
    )


def machine_info() -> dict:
    """Static description of the hardware, stored alongside every profiling run."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "python": platform.python_version(),
    }
