"""The routing rule and how its aggressiveness is tuned to a quality target.

Rule: walk the models from cheapest to most expensive and take the first whose
predicted quality is at least (1 - delta) times the largest model's predicted
quality. The margin is relative, matching the relative quality target, so a
weak prediction for the largest model never makes a much weaker model look
"close enough". A bigger delta saves more compute but risks more quality.

The user sets a relative quality target tau ("95" = keep at least 95% of the
largest model's quality). Delta is not guessed: on held-out validation data we
pick the largest delta whose routed accuracy still reaches tau times the
largest model's accuracy. The guarantee is therefore on the whole workload,
not on any single prompt.
"""

from __future__ import annotations

import numpy as np

# 100% is not tuned: asking for all of the largest model's quality means using it.
TAU_GRID = (0.80, 0.85, 0.90, 0.95, 0.99)
DELTA_GRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def choose(
    probs: dict[str, float],
    costs: dict[str, float],
    largest: str,
    delta: float,
    novelty: str = "high",
) -> tuple[str, str]:
    """Pick a model for one prompt. Returns (model, reason)."""
    if novelty == "low":
        return largest, "prompt unlike the training data: using the largest model"
    if novelty == "medium":
        delta = delta / 2
    bar = (1 - delta) * probs[largest]
    for name in sorted(costs, key=costs.get):
        if probs[name] >= bar:
            if name == largest:
                return name, "no cheaper model is predicted to keep enough quality"
            return name, "cheapest model predicted close enough to the largest for your quality target"
    return largest, "no cheaper model is predicted to keep enough quality"


def simulate(probs: np.ndarray, order: list[int], largest: int, delta: float) -> np.ndarray:
    """Vectorised `choose` over many prompts: probs is (n_prompts, n_models)."""
    chosen = np.full(probs.shape[0], largest)
    undecided = np.ones(probs.shape[0], dtype=bool)
    bar = (1 - delta) * probs[:, largest]
    for j in order:
        hit = undecided & (probs[:, j] >= bar)
        chosen[hit] = j
        undecided &= ~hit
    return chosen


def tune_deltas(
    probs: np.ndarray,
    correct: np.ndarray,
    order: list[int],
    largest: int,
    taus: tuple[float, ...] = TAU_GRID,
) -> dict[float, float]:
    """For each target tau, the most aggressive delta that still meets it on validation."""
    reference = correct[:, largest].mean()
    rows = np.arange(len(correct))
    accuracy = {}
    for delta in DELTA_GRID:
        chosen = simulate(probs, order, largest, float(delta))
        accuracy[float(delta)] = correct[rows, chosen].mean()
    table = {}
    for tau in taus:
        ok = [d for d, acc in accuracy.items() if acc >= tau * reference - 1e-12]
        table[tau] = max(ok) if ok else 0.0
    return table


def pick_target(table: dict, quality: float):
    """Settings for a requested quality (0-100), rounding the target up to be safe.

    A request stricter than anything tuned gets the strictest tuned settings.
    """
    tau = quality / 100 if quality > 1 else quality
    eligible = [t for t in sorted(table) if t >= tau - 1e-9]
    return table[eligible[0]] if eligible else table[max(table)]
