"""Energy measurement across platforms, with an honest "not measured" fallback.

Power telemetry differs by hardware, so each backend is a small class with the
same interface. A sampler runs in the background and records (time, watts);
the energy of one request is the power integrated over that request's window.

    Apple Silicon : powermetrics  (needs sudo; run `sudo -v` first)
    NVIDIA GPU    : nvidia-smi    (GPU board power only)
    Linux Intel/AMD: RAPL counters (may need root on recent kernels)

When nothing is available the meter reports None and results are labelled
"none", so measured and estimated energy are never mixed up.
"""

from __future__ import annotations

import bisect
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


class PowerSampler:
    """Base class: keeps a time-ordered list of power samples in watts."""

    source = "none"

    def __init__(self):
        self._times: list[float] = []
        self._watts: list[float] = []
        self._lock = threading.Lock()

    @classmethod
    def available(cls) -> bool:
        return False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def add_sample(self, t: float, watts: float) -> None:
        with self._lock:
            self._times.append(t)
            self._watts.append(watts)

    def energy_j(self, t0: float, t1: float) -> float | None:
        """Joules used between two time.monotonic() readings, or None."""
        with self._lock:
            if not self._times or t1 <= t0:
                return None
            lo = bisect.bisect_left(self._times, t0)
            hi = bisect.bisect_right(self._times, t1)
            window = self._watts[lo:hi]
            if not window:
                # Window shorter than the sampling interval: use the nearest sample.
                nearest = min(lo, len(self._watts) - 1)
                window = [self._watts[nearest]]
        return sum(window) / len(window) * (t1 - t0)

    def mean_watts(self, t0: float, t1: float) -> float | None:
        energy = self.energy_j(t0, t1)
        return None if energy is None else energy / (t1 - t0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class _LineSampler(PowerSampler):
    """Runs a command that prints power readings and parses each line."""

    interval_ms = 200

    def command(self) -> list[str]:
        raise NotImplementedError

    def parse(self, line: str) -> float | None:
        raise NotImplementedError

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(self.interval_ms / 1000 * 2)  # let the first samples arrive

    def _read(self) -> None:
        for line in self._proc.stdout:
            watts = self.parse(line)
            if watts is not None:
                self.add_sample(time.monotonic(), watts)

    def stop(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


_COMBINED = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")


class PowermetricsSampler(_LineSampler):
    """Apple Silicon package power (CPU + GPU + Neural Engine) via powermetrics."""

    source = "powermetrics"

    @classmethod
    def available(cls) -> bool:
        if sys.platform != "darwin" or not Path("/usr/bin/powermetrics").exists():
            return False
        # -n: never prompt. Succeeds only if sudo credentials are cached or allowed.
        return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0

    def command(self) -> list[str]:
        return [
            "sudo", "-n", "/usr/bin/powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power",
            "-i", str(self.interval_ms),
        ]

    def parse(self, line: str) -> float | None:
        match = _COMBINED.search(line)
        return float(match.group(1)) / 1000 if match else None


class NvidiaSmiSampler(_LineSampler):
    """NVIDIA GPU board power via nvidia-smi."""

    source = "nvidia-smi"

    @classmethod
    def available(cls) -> bool:
        return shutil.which("nvidia-smi") is not None

    def command(self) -> list[str]:
        # GPU 0 only: Ollama runs a single model on one GPU in the common case.
        return [
            "nvidia-smi", "-i", "0", "--query-gpu=power.draw",
            "--format=csv,noheader,nounits", "-lms", str(self.interval_ms),
        ]

    def parse(self, line: str) -> float | None:
        try:
            return float(line.strip())
        except ValueError:
            return None


class RaplSampler(PowerSampler):
    """Linux CPU package energy from the RAPL counter (reads energy, not power)."""

    source = "rapl"
    COUNTER = Path("/sys/class/powercap/intel-rapl:0/energy_uj")

    @classmethod
    def available(cls) -> bool:
        try:
            int(cls.COUNTER.read_text())
            return True
        except (OSError, ValueError):
            return False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        time.sleep(0.4)

    def _poll(self) -> None:
        last_t, last_uj = time.monotonic(), int(self.COUNTER.read_text())
        while self._running:
            time.sleep(0.2)
            t, uj = time.monotonic(), int(self.COUNTER.read_text())
            if uj >= last_uj:  # skip the sample where the counter wraps around
                self.add_sample(t, (uj - last_uj) / 1e6 / (t - last_t))
            last_t, last_uj = t, uj

    def stop(self) -> None:
        self._running = False


BACKENDS = [PowermetricsSampler, NvidiaSmiSampler, RaplSampler]


def best_sampler() -> PowerSampler:
    """The first backend that works on this machine, or a no-op sampler."""
    for backend in BACKENDS:
        if backend.available():
            return backend()
    return PowerSampler()
