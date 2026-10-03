"""Train the capability predictor on a profiling run and evaluate every policy.

    python scripts/train_router.py --run runs/qwen35-ladder

Steps:
  1. align the run into a (question x model) outcome matrix
  2. split questions 60/20/20 into train / validation / test (fixed seed)
  3. embed every question once with nomic-embed-text (cached in the run folder)
  4. fit one calibrated head per model on train, tune delta per quality target on validation
  5. score baselines, the oracle and the router on test, with measured router overhead
  6. save the router artifact, metrics.json, a summary table and the Pareto chart
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from shiftai.evaluate import load_matrix, oracle_choice, score_policy, split_indices
from shiftai.features import cached_embeddings, embed, featurize
from shiftai.ollama import OllamaClient
from shiftai.policy import TAU_GRID, simulate, tune_deltas
from shiftai.predictor import train_predictor
from shiftai.router import RouterArtifact
from shiftai.tasks import load_questions

DEFAULT_ARTIFACT = Path("src/shiftai/artifacts/router.json")


def measure_overhead(texts: list[str], predictor, client: OllamaClient, n: int = 30) -> float:
    """Mean seconds to embed and score one prompt, the way the live router does."""
    embed(texts[:1], client)  # warm the embedding model
    timings = []
    for text in texts[:n]:
        t0 = time.perf_counter()
        vec = embed([text], client)
        predictor.predict(featurize(vec, [text]))
        timings.append(time.perf_counter() - t0)
    return float(np.mean(timings))


def plot_pareto(results: list[dict], out: Path, metric: str, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    singles = [r for r in results if r["policy"].startswith("only:")]
    routers = [r for r in results if r["policy"].startswith("router@")]
    oracle = [r for r in results if r["policy"] == "oracle"]
    ax.plot([r[metric] for r in singles], [r["accuracy"] for r in singles], "o", color="#888", label="single model")
    for r in singles:
        ax.annotate(r["policy"][5:], (r[metric], r["accuracy"]), fontsize=8, xytext=(4, -10), textcoords="offset points")
    ax.plot([r[metric] for r in routers], [r["accuracy"] for r in routers], "-s", color="#1f6feb", label="ShiftAI router")
    for r in routers:
        ax.annotate(r["policy"][7:], (r[metric], r["accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.plot([r[metric] for r in oracle], [r["accuracy"] for r in oracle], "*", ms=14, color="#d29922", label="oracle")
    ax.set_xlabel(label)
    ax.set_ylabel("accuracy (test set)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the ShiftAI router.")
    parser.add_argument("--run", type=Path, required=True, help="profiling run folder")
    parser.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = OllamaClient()
    questions = {q.id: q for q in load_questions(args.questions)}
    matrix = load_matrix(args.run, questions=questions)  # re-graded with the current grader
    models, largest = matrix.models, len(matrix.models) - 1
    print(f"{len(matrix.qids)} questions answered by all of: {', '.join(models)}")

    # The router sees what a user would type, not the benchmark's format instruction.
    texts = [questions[q].feature_text() for q in matrix.qids]
    kinds = [questions[q].kind for q in matrix.qids]
    emb = cached_embeddings(texts, client, args.run)
    x = featurize(emb, texts)

    train, val, test = split_indices(len(texts), seed=args.seed)
    y = {m: matrix.correct[:, j].astype(int) for j, m in enumerate(models)}
    predictor = train_predictor(
        x[train], {m: v[train] for m, v in y.items()},
        x[val], {m: v[val] for m, v in y.items()},
        emb[train], emb[val], seed=args.seed,
    )

    def prob_matrix(rows: np.ndarray) -> np.ndarray:
        p = predictor.predict(x[rows])
        return np.column_stack([p[m] for m in models])

    # Cost order for routing: cheapest first by mean warm latency on train.
    order = list(np.argsort(matrix.latency[train].mean(axis=0)))
    deltas = tune_deltas(prob_matrix(val), matrix.correct[val], order, largest)

    expected_tokens = {}
    for j, m in enumerate(models):
        per_kind = {}
        for kind, key in (("choice", "choice"), ("math", "open")):
            rows = [i for i in train if kinds[i] == kind]
            if rows:
                per_kind[key] = float(np.median(matrix.output_tokens[rows, j]))
        expected_tokens[m] = per_kind

    artifact = RouterArtifact(
        ladder=models,
        predictor=predictor,
        deltas=deltas,
        expected_tokens=expected_tokens,
        meta={
            "run": str(args.run),
            "n_questions": len(texts),
            "split": {"train": len(train), "val": len(val), "test": len(test), "seed": args.seed},
            "trained": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    artifact.save(args.artifact)
    print(f"Saved router artifact to {args.artifact}")

    overhead_s = measure_overhead([texts[i] for i in test], predictor, client)
    print(f"Router overhead: {overhead_s * 1000:.1f} ms per prompt")

    results = []
    for j, m in enumerate(models):
        results.append(score_policy(f"only:{m}", np.full(len(test), j), matrix, test))
    cost = matrix.latency[test]
    results.append(score_policy("oracle", oracle_choice(matrix.correct[test], cost), matrix, test))
    test_probs = prob_matrix(test)
    for tau in TAU_GRID:
        chosen = simulate(test_probs, order, largest, deltas[tau])
        r = score_policy(f"router@{int(round(tau * 100))}", chosen, matrix, test, overhead_s=overhead_s)
        r["target"] = tau
        r["delta"] = deltas[tau]
        r["meets_target"] = r["relative_quality"] >= tau - 1e-9
        results.append(r)

    args.results.mkdir(parents=True, exist_ok=True)
    summary = {"models": models, "overhead_ms": overhead_s * 1000, "deltas": deltas, "results": results}
    (args.results / "metrics.json").write_text(json.dumps(summary, indent=2))

    has_energy = "mean_energy_j" in results[0]
    header = "| policy | accuracy | relative quality | mean latency (s) | p95 latency (s) | latency saved |"
    header += " energy (J) | energy saved |" if has_energy else ""
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for r in results:
        row = (
            f"| {r['policy']} | {r['accuracy']:.3f} | {r['relative_quality']:.1%} | "
            f"{r['mean_latency_s']:.2f} | {r['p95_latency_s']:.2f} | {r['latency_saving']:.1%} |"
        )
        if has_energy:
            row += f" {r['mean_energy_j']:.1f} | {r['energy_saving']:.1%} |"
        lines.append(row)
    (args.results / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_pareto(results, args.results / "pareto_latency.png", "mean_latency_s", "mean latency per prompt (s)")
    if has_energy:
        plot_pareto(results, args.results / "pareto_energy.png", "mean_energy_j", "mean energy per prompt (J)")
    print(f"Wrote metrics, summary and charts to {args.results}/")


if __name__ == "__main__":
    main()
