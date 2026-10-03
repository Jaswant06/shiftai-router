"""Offline evaluation of routing policies on a completed profiling run.

Every model answered every question during profiling, so any routing policy
can be scored exactly by looking up the logged outcome of the model it would
have chosen. Routing never changes a model's answer, only which model is
asked, so this replay is faithful. The router's own overhead (embedding and
prediction) is measured separately and added to its latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .profiler import load_records


@dataclass
class ProfileMatrix:
    """Per-question outcomes for every model, aligned by row and column."""

    qids: list[str]
    tasks: list[str]
    models: list[str]
    correct: np.ndarray       # (n, m) bool
    latency: np.ndarray       # (n, m) seconds, warm (load time excluded)
    energy: np.ndarray        # (n, m) joules, NaN when not measured
    output_tokens: np.ndarray  # (n, m)

    @property
    def has_energy(self) -> bool:
        return not np.isnan(self.energy).all()


def build_matrix(records: list[dict], models: list[str]) -> ProfileMatrix:
    """Keep only questions that every model answered, in a stable order."""
    by_key = {(r["model"], r["qid"]): r for r in records}
    task_of = {r["qid"]: r["task"] for r in records}
    qids = sorted(q for q in task_of if all((m, q) in by_key for m in models))
    shape = (len(qids), len(models))
    correct = np.zeros(shape, dtype=bool)
    latency = np.zeros(shape)
    energy = np.full(shape, np.nan)
    tokens = np.zeros(shape)
    for i, q in enumerate(qids):
        for j, m in enumerate(models):
            r = by_key[(m, q)]
            correct[i, j] = r["correct"]
            latency[i, j] = max(r["total_s"] - r["load_s"], 0.0)
            if r.get("energy_j") is not None:
                energy[i, j] = r["energy_j"]
            tokens[i, j] = r["output_tokens"]
    return ProfileMatrix(qids, [task_of[q] for q in qids], models, correct, latency, energy, tokens)


def load_matrix(run_dir: str | Path, models: list[str] | None = None) -> ProfileMatrix:
    run_dir = Path(run_dir)
    if models is None:
        models = json.loads((run_dir / "meta.json").read_text())["models"]
    return build_matrix(load_records(run_dir / "records.jsonl"), models)


def split_indices(n: int, seed: int = 0, fractions=(0.6, 0.2, 0.2)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random train/validation/test split by question."""
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def oracle_choice(correct: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Cheapest correct model per question; cheapest overall when none is correct."""
    masked = np.where(correct, cost, np.inf)
    best = masked.argmin(axis=1)
    none_right = ~correct.any(axis=1)
    best[none_right] = cost[none_right].argmin(axis=1)
    return best


def _bootstrap(values: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_policy(
    name: str,
    chosen: np.ndarray,
    matrix: ProfileMatrix,
    rows: np.ndarray,
    overhead_s: float = 0.0,
) -> dict:
    """Quality, latency, energy and usage for one policy on the given rows."""
    largest = len(matrix.models) - 1
    idx = rows
    right = matrix.correct[idx, chosen]
    lat = matrix.latency[idx, chosen] + overhead_s
    ref_acc = matrix.correct[idx, largest].mean()
    ref_lat = matrix.latency[idx, largest].mean()
    result = {
        "policy": name,
        "accuracy": float(right.mean()),
        "accuracy_ci95": _bootstrap(right.astype(float)),
        "relative_quality": float(right.mean() / ref_acc) if ref_acc > 0 else float("nan"),
        "mean_latency_s": float(lat.mean()),
        "p50_latency_s": float(np.percentile(lat, 50)),
        "p95_latency_s": float(np.percentile(lat, 95)),
        "latency_saving": float(1 - lat.mean() / ref_lat) if ref_lat > 0 else float("nan"),
        "mean_output_tokens": float(matrix.output_tokens[idx, chosen].mean()),
        "share": {m: float((chosen == j).mean()) for j, m in enumerate(matrix.models)},
    }
    if matrix.has_energy:
        energy = matrix.energy[idx, chosen]
        ref_energy = np.nanmean(matrix.energy[idx, largest])
        result["mean_energy_j"] = float(np.nanmean(energy))
        result["energy_saving"] = float(1 - np.nanmean(energy) / ref_energy)
    return result
