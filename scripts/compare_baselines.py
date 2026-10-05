"""Where per-prompt routing beats a random split of models, and where it does not.

    python scripts/compare_baselines.py --run runs/qwen35 runs/qwen35-open \
        --questions data/questions.jsonl data/open.jsonl

On the whole test set a well-tuned random split is about as efficient as
ShiftAI between 80% and 95%. This script looks past the average:

1. Workload mix. The random split's proportion is tuned on the development
   data's mix of prompts. A real workload rarely matches it, so each kind of
   prompt is scored on its own, as if it were the whole workload.
2. Broken answers. How often the routed model gets wrong a prompt the 9B gets
   right, and how often hard prompts (only the 9B gets them right) reach the 9B.

The random split is averaged over 200 seeds. ShiftAI's decisions come from the
shipped router (router.json), so overall numbers can differ from
results/metrics.json by a tenth of a point. Writes results/baselines.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shiftai.evaluate import load_matrix, merge_matrices, split_indices
from shiftai.features import cached_embeddings
from shiftai.ollama import OllamaClient
from shiftai.policy import TAU_GRID, simulate
from shiftai.router import RouterArtifact
from shiftai.tasks import load_questions
from train_router import upper_hull

FAMILY = {"math": "Math", "choice": "Multiple choice", "code": "Coding", "open": "Open-ended"}
SEEDS = 200


def mix_share(dev_acc: np.ndarray, dev_cost: np.ndarray, tau: float, largest: int):
    """The random split's two models and the share sent to the better one (as in train_router.py)."""
    target = tau * dev_acc[largest]
    hull = upper_hull([(c, a) for c, a in zip(dev_cost, dev_acc)])
    index = {(c, a): j for j, (c, a) in enumerate(zip(dev_cost, dev_acc))}
    if target <= hull[0][1]:
        return index[hull[0]], index[hull[0]], 0.0
    for lo, hi in zip(hull, hull[1:]):
        if lo[1] < target <= hi[1]:
            return index[lo], index[hi], (target - lo[1]) / (hi[1] - lo[1])
    return largest, largest, 1.0


def stats(chosen: np.ndarray, correct: np.ndarray, latency: np.ndarray, energy: np.ndarray, largest: int) -> dict:
    rows = np.arange(len(chosen))
    right = correct[rows, chosen]
    big_right = correct[:, largest] >= 0.5
    hard = big_right & (correct[:, :largest] < 0.5).all(axis=1)
    return {
        "quality": right.mean() / correct[:, largest].mean(),
        "faster": 1 - latency[rows, chosen].mean() / latency[:, largest].mean(),
        "less_energy": 1 - np.nanmean(energy[rows, chosen]) / np.nanmean(energy[:, largest]),
        "broken": ((right < 0.5) & big_right).sum() / max(big_right.sum(), 1),
        "hard_to_9b": (chosen[hard] == largest).mean() if hard.any() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/baselines.md"))
    args = parser.parse_args()

    questions = {q.id: q for path in args.questions for q in load_questions(path)}
    matrix = merge_matrices([load_matrix(run, questions=questions) for run in args.run])
    largest = len(matrix.models) - 1
    texts = [questions[q].feature_text() for q in matrix.qids]
    family = np.array([FAMILY[questions[q].kind] for q in matrix.qids])
    emb = cached_embeddings(texts, OllamaClient(), args.run[0])
    train, val, test = split_indices(len(texts), seed=0)
    dev = np.concatenate([train, val])
    order = list(np.argsort(matrix.latency[dev].mean(axis=0)))
    dev_acc, dev_cost = matrix.correct[dev].mean(axis=0), matrix.latency[dev].mean(axis=0)

    artifact = RouterArtifact.load()
    test_texts = [texts[i] for i in test]
    probs = {k: p.predict(emb[test], test_texts) for k, p in artifact.predictors.items()}
    groups = {"All prompts": np.ones(len(test), dtype=bool)}
    groups.update({f: family[test] == f for f in FAMILY.values()})

    table = {}
    for tau in TAU_GRID:
        settings = artifact.targets[tau]
        shift = simulate(probs[settings["clusters"]], order, largest, settings["delta"])
        lo, hi, share = mix_share(dev_acc, dev_cost, tau, largest)
        draws = [np.where(np.random.default_rng(s).uniform(size=len(test)) < share, hi, lo) for s in range(SEEDS)]
        for name, mask in groups.items():
            rows = test[mask]
            c, lat, en = matrix.correct[rows], matrix.latency[rows], matrix.energy[rows]
            ours = stats(shift[mask], c, lat, en, largest)
            rand = {k: float(np.nanmean([stats(d[mask], c, lat, en, largest)[k] for d in draws])) for k in ours}
            table[(tau, name)] = (int(mask.sum()), ours, rand)

    pct = lambda x: "n/a" if np.isnan(x) else f"{x:.1%}"
    lines = ["# ShiftAI versus a tuned random split", "",
             f"{len(test)} held-out prompts. The random split is averaged over {SEEDS} seeds. Each workload row scores",
             "one kind of prompt on its own, as if it were the whole workload. \"Broken\" counts prompts the 9B gets",
             "right that the routed model gets wrong; \"hard\" prompts are the ones only the 9B gets right.", "",
             "| Target | Workload | Prompts | Quality kept (ShiftAI / random) | Faster (ShiftAI / random) "
             "| 9B-correct answers broken (ShiftAI / random) | Hard prompts sent to 9B (ShiftAI / random) |",
             "|---|---|---:|---|---|---|---|"]
    for (tau, name), (n, ours, rand) in table.items():
        lines.append(f"| {int(round(tau * 100))}% | {name} | {n} | {pct(ours['quality'])} / {pct(rand['quality'])} "
                     f"| {pct(ours['faster'])} / {pct(rand['faster'])} | {pct(ours['broken'])} / {pct(rand['broken'])} "
                     f"| {pct(ours['hard_to_9b'])} / {pct(rand['hard_to_9b'])} |")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json_out = {f"{int(round(t * 100))}|{g}": {"n": n, "shiftai": o, "random": r} for (t, g), (n, o, r) in table.items()}
    args.out.with_suffix(".json").write_text(json.dumps(json_out, indent=2, default=float))


if __name__ == "__main__":
    main()
