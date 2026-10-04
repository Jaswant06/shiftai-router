"""Train the router on a profiling run and evaluate it against baselines.

    python scripts/train_router.py --run runs/qwen35

Steps:
  1. align the run into a (question x model) outcome matrix, re-grading every
     stored reply with the current grader
  2. hold out 20% of questions as a test set that nothing below looks at
  3. on the other 80% (dev), use 5-fold out-of-fold predictions to pick, for
     each quality target, the number of clusters and the delta
  4. fit the final predictors on all dev questions and save the router artifact
  5. score single models, the oracle, the per-type router and ShiftAI on test,
     with the router's measured overhead added to its latency
  6. write metrics.json, a summary table and the Pareto charts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from shiftai.evaluate import load_matrix, merge_matrices, oracle_choice, score_policy, split_indices
from shiftai.features import cached_embeddings, embed
from shiftai.ollama import OllamaClient
from shiftai.policy import TAU_GRID, simulate, tune_deltas
from shiftai.predictor import train_predictor
from shiftai.router import RouterArtifact
from shiftai.tasks import load_questions

DEFAULT_ARTIFACT = Path("src/shiftai/artifacts/router.json")
FAMILY = {"math": "benchmark", "choice": "benchmark", "code": "coding", "open": "open-ended"}
CLUSTER_OPTIONS = (1, 2, 4, 8, 16)


def out_of_fold_probs(emb, texts, correct, models, rows, k, seed=0) -> np.ndarray:
    """Predictions for each dev question from a predictor that never saw it."""
    probs = np.zeros((len(rows), len(models)))
    for fit, held in KFold(5, shuffle=True, random_state=seed).split(rows):
        f, h = rows[fit], rows[held]
        predictor = train_predictor(emb[f], [texts[i] for i in f], correct[f], models, k, seed=seed)
        probs[held] = predictor.predict(emb[h], [texts[i] for i in h])
    return probs


def savings(probs, latency, order, largest, deltas) -> dict[float, float]:
    """Latency saved at each quality target's tuned delta (higher is better)."""
    rows = np.arange(len(latency))
    out = {}
    for tau in TAU_GRID:
        chosen = simulate(probs, order, largest, deltas[tau])
        out[tau] = float(1 - latency[rows, chosen].mean() / latency[:, largest].mean())
    return out


def random_mix(dev_acc, dev_cost, tau, largest, n, seed=0) -> np.ndarray:
    """Baseline: send each prompt at random to one of two models, in the
    proportion that just meets the target on dev (the best mix of fixed models)."""
    target = tau * dev_acc[largest]
    hull = upper_hull([(c, a) for c, a in zip(dev_cost, dev_acc)])
    index = {(c, a): j for j, (c, a) in enumerate(zip(dev_cost, dev_acc))}
    if target <= hull[0][1]:
        return np.full(n, index[hull[0]])
    for lo, hi in zip(hull, hull[1:]):
        if lo[1] < target <= hi[1]:
            share_hi = (target - lo[1]) / (hi[1] - lo[1])
            pick_hi = np.random.default_rng(seed).uniform(size=n) < share_hi
            return np.where(pick_hi, index[hi], index[lo])
    return np.full(n, largest)


def measure_overhead(texts: list[str], predictor, client: OllamaClient, n: int = 30) -> float:
    """Mean seconds to embed and score one prompt, the way the live router does."""
    embed(texts[:1], client)  # warm the embedding model
    timings = []
    for text in texts[:n]:
        t0 = time.perf_counter()
        predictor.predict(embed([text], client), [text])
        timings.append(time.perf_counter() - t0)
    return float(np.mean(timings))


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Best accuracy reachable at each cost by randomly mixing the given models."""
    hull = []
    for x, y in sorted(points):
        if hull and y <= hull[-1][1]:
            continue  # costs more and is no more accurate: never worth mixing in
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x1) <= (y - y1) * (x2 - x1):
                hull.pop()  # the middle point sits below the line: drop it
            else:
                break
        hull.append((x, y))
    return hull


def plot_pareto(results: list[dict], out: Path, metric: str, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(prefix):
        rows = [r for r in results if r["policy"].startswith(prefix)]
        return sorted(rows, key=lambda r: r[metric])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    singles = series("only:")
    ax.plot([r[metric] for r in singles], [r["accuracy"] for r in singles], "o", color="#8b949e", label="single models")
    hull = upper_hull([(r[metric], r["accuracy"]) for r in singles])
    ax.plot(*zip(*hull), "--", color="#8b949e", label="best random mix of single models")
    for r in singles:
        ax.annotate(r["policy"][5:], (r[metric], r["accuracy"]), fontsize=8, xytext=(5, -11), textcoords="offset points")
    by_type = series("per-type@")
    ax.plot([r[metric] for r in by_type], [r["accuracy"] for r in by_type], "-^", color="#a371f7", label="per-type router")
    shift = series("shiftai@")
    ax.plot([r[metric] for r in shift], [r["accuracy"] for r in shift], "-s", color="#1f6feb", label="ShiftAI")
    for r in shift:
        ax.annotate(r["policy"].split("@")[1] + "%", (r[metric], r["accuracy"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    oracle = series("oracle")
    ax.plot([r[metric] for r in oracle], [r["accuracy"] for r in oracle], "*", ms=15, color="#d29922", label="oracle")
    ax.set_xlabel(label)
    ax.set_ylabel("accuracy on held-out test questions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary(results: list[dict], path: Path) -> str:
    has_energy = "mean_energy_j" in results[0]
    header = "| policy | accuracy | relative quality | mean latency (s) | p95 latency (s) | latency saved |"
    if has_energy:
        header += " energy (J) | energy saved |"
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for r in results:
        row = (
            f"| {r['policy']} | {r['accuracy']:.3f} | {r['relative_quality']:.1%} | "
            f"{r['mean_latency_s']:.2f} | {r['p95_latency_s']:.2f} | {r['latency_saving']:.1%} |"
        )
        if has_energy:
            row += f" {r['mean_energy_j']:.1f} | {r['energy_saving']:.1%} |"
        lines.append(row)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the ShiftAI router.")
    parser.add_argument("--run", type=Path, nargs="+", required=True, help="profiling run folder(s)")
    parser.add_argument("--questions", type=Path, nargs="+", default=[Path("data/questions.jsonl")])
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = OllamaClient()
    questions = {q.id: q for path in args.questions for q in load_questions(path)}
    # Re-graded with the current grader; open-ended questions use the judge's labels.
    matrix = merge_matrices([load_matrix(run, questions=questions) for run in args.run])
    models, largest = matrix.models, len(matrix.models) - 1
    print(f"{len(matrix.qids)} questions answered by all of: {', '.join(models)}")

    # The router sees what a user would type, not the benchmark's format instruction.
    texts = [questions[q].feature_text() for q in matrix.qids]
    kinds = [questions[q].kind for q in matrix.qids]
    family = np.array([FAMILY[k] for k in kinds])
    print("  by family:", {str(f): int((family == f).sum()) for f in dict.fromkeys(family)})
    emb = cached_embeddings(texts, client, args.run[0])
    correct = matrix.correct.astype(float)

    train, val, test = split_indices(len(texts), seed=args.seed)
    dev = np.concatenate([train, val])
    order = list(np.argsort(matrix.latency[dev].mean(axis=0)))  # cheapest first

    # For every quality target, pick the cluster count and delta using only dev questions.
    tuned = {}
    for k in CLUSTER_OPTIONS:
        oof = out_of_fold_probs(emb, texts, correct, models, dev, k, args.seed)
        deltas = tune_deltas(oof, matrix.correct[dev], order, largest)
        tuned[k] = (deltas, savings(oof, matrix.latency[dev], order, largest, deltas))
        print(f"  {k:2d} clusters per type, dev latency saved: "
              + "  ".join(f"{int(t * 100)}%:{v:.1%}" for t, v in tuned[k][1].items()))
    targets = {}
    for tau in TAU_GRID:
        # Ties go to fewer clusters: the simpler model.
        k = max(CLUSTER_OPTIONS, key=lambda c: (round(tuned[c][1][tau], 4), -c))
        targets[tau] = {"clusters": k, "delta": tuned[k][0][tau]}
    print("Chosen per target:", {int(t * 100): v["clusters"] for t, v in targets.items()})

    predictors = {
        k: train_predictor(emb[dev], [texts[i] for i in dev], correct[dev], models, k, seed=args.seed)
        for k in sorted({v["clusters"] for v in targets.values()} | {1})
    }

    expected_tokens = {}
    for j, m in enumerate(models):
        per_kind = {}
        for key in ("choice", "open"):
            # The live router only tells multiple choice from everything else.
            rows = [i for i in dev if (kinds[i] == "choice") == (key == "choice")]
            if rows:
                per_kind[key] = float(np.median(matrix.output_tokens[rows, j]))
        expected_tokens[m] = per_kind

    artifact = RouterArtifact(
        ladder=models,
        predictors=predictors,
        targets=targets,
        expected_tokens=expected_tokens,
        meta={
            "runs": [str(r) for r in args.run],
            "n_questions": len(texts),
            "split": {"dev": len(dev), "test": len(test), "seed": args.seed},
            "trained": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    artifact.save(args.artifact)
    print(f"Saved router artifact to {args.artifact}")

    overhead_s = measure_overhead([texts[i] for i in test], predictors[max(predictors)], client)
    print(f"Router overhead: {overhead_s * 1000:.1f} ms per prompt")

    results = []
    for j, m in enumerate(models):
        results.append(score_policy(f"only:{m}", np.full(len(test), j), matrix, test))
    results.append(score_policy("oracle", oracle_choice(matrix.correct[test], matrix.latency[test]), matrix, test))

    # Baseline: the same router with one cluster per prompt type, i.e. routing by type alone.
    test_texts = [texts[i] for i in test]
    probs = {k: p.predict(emb[test], test_texts) for k, p in predictors.items()}
    dev_acc, dev_cost = matrix.correct[dev].mean(axis=0), matrix.latency[dev].mean(axis=0)
    by_family = []
    for tau in TAU_GRID:
        tag = int(round(tau * 100))
        mix = random_mix(dev_acc, dev_cost, tau, largest, len(test), seed=args.seed)
        results.append(score_policy(f"random-mix@{tag}", mix, matrix, test))
        r = score_policy(f"per-type@{tag}", simulate(probs[1], order, largest, tuned[1][0][tau]), matrix, test, overhead_s)
        results.append(r)
        k, delta = targets[tau]["clusters"], targets[tau]["delta"]
        r = score_policy(f"shiftai@{tag}", simulate(probs[k], order, largest, delta), matrix, test, overhead_s)
        r.update(target=tau, clusters=k, delta=delta, meets_target=r["relative_quality"] >= tau - 1e-9)
        results.append(r)
        chosen = simulate(probs[k], order, largest, delta)
        for fam in dict.fromkeys(family[test]):
            mask = family[test] == fam
            fr = score_policy(f"shiftai@{tag}", chosen[mask], matrix, test[mask], overhead_s)
            fr.update(family=fam, n=int(mask.sum()))
            by_family.append(fr)

    args.results.mkdir(parents=True, exist_ok=True)
    summary = {
        "models": models,
        "dev_latency_saving": {str(k): {str(t): v for t, v in s[1].items()} for k, s in tuned.items()},
        "targets": {str(t): v for t, v in targets.items()},
        "overhead_ms": overhead_s * 1000,
        "results": results,
        "by_family": by_family,
    }
    (args.results / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(write_summary(results, args.results / "summary.md"))
    if len(set(family)) > 1:
        lines = ["| target | prompts | n | quality kept | latency saved | energy saved |", "|---|---|---|---|---|---|"]
        for r in by_family:
            energy = f"{r['energy_saving']:.1%}" if "energy_saving" in r else "n/a"
            lines.append(f"| {r['policy'].split('@')[1]}% | {r['family']} | {r['n']} | "
                         f"{r['relative_quality']:.1%} | {r['latency_saving']:.1%} | {energy} |")
        (args.results / "summary_by_family.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))

    plot_pareto(results, args.results / "pareto_latency.png", "mean_latency_s", "mean latency per prompt (s)")
    if "mean_energy_j" in results[0]:
        plot_pareto(results, args.results / "pareto_energy.png", "mean_energy_j", "mean energy per prompt (J)")
    print(f"Wrote metrics, summary and charts to {args.results}/")


if __name__ == "__main__":
    main()
