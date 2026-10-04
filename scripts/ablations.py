"""Two designs that were tried and rejected, with the numbers behind each decision.

    python scripts/ablations.py --run runs/qwen35

1. Per-prompt logistic regression on the full embedding, calibrated and tuned
   exactly like ShiftAI (out-of-fold on dev, same routing rule), then routed on
   the held-out test questions. It was comparable between the 80% and 95%
   targets but missed the 99% target and saved much less there, so ShiftAI
   uses the cluster-based estimates, which met every target.

2. Agreement cascades: run two cheaper models, accept the answer if they
   agree, otherwise escalate. Scored on the held-out test questions. Running
   models one after another costs more than running the 9B model once, so
   every cascade was slower and used more energy than always using the 9B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from shiftai.evaluate import load_matrix, score_policy, split_indices
from shiftai.features import cached_embeddings
from shiftai.grading import extract_choice, extract_number
from shiftai.ollama import OllamaClient
from shiftai.profiler import load_records
from shiftai.tasks import load_questions


def fit_lr(features, labels, fit_rows, c):
    mu, sd = features[fit_rows].mean(0), features[fit_rows].std(0) + 1e-6
    model = LogisticRegression(C=c, max_iter=5000).fit((features[fit_rows] - mu) / sd, labels[fit_rows])
    return lambda rows: model.predict_proba((features[rows] - mu) / sd)[:, 1]


def oof_scores(features, labels, rows, c) -> np.ndarray:
    scores = np.zeros(len(rows))
    for fit, held in KFold(5, shuffle=True, random_state=0).split(rows):
        scores[held] = fit_lr(features, labels, rows[fit], c)(rows[held])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejected-design ablations.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    args = parser.parse_args()

    questions = {q.id: q for q in load_questions(args.questions)}
    matrix = load_matrix(args.run, questions=questions)
    texts = [questions[q].feature_text() for q in matrix.qids]
    emb = cached_embeddings(texts, OllamaClient(), args.run)
    train, val, test = split_indices(len(texts))
    dev = np.concatenate([train, val])

    from shiftai.policy import TAU_GRID, simulate, tune_deltas

    print("1. Per-prompt logistic regression router (held-out test questions)")
    features = np.hstack([emb, np.log1p([[len(t)] for t in texts])])
    largest = len(matrix.models) - 1
    order = list(np.argsort(matrix.latency[dev].mean(axis=0)))
    oof, test_probs = [], []
    for j, m in enumerate(matrix.models):
        labels = matrix.correct[:, j].astype(int)
        c = max((0.001, 0.01, 0.1), key=lambda c: roc_auc_score(labels[dev], oof_scores(features, labels, dev, c)))
        raw = oof_scores(features, labels, dev, c)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, labels[dev])
        print(f"   {m:14} out-of-fold AUC {roc_auc_score(labels[dev], raw):.3f} (C={c})")
        oof.append(iso.predict(raw))
        test_probs.append(iso.predict(fit_lr(features, labels, dev, c)(test)))
    oof, test_probs = np.column_stack(oof), np.column_stack(test_probs)
    deltas = tune_deltas(oof, matrix.correct[dev], order, largest)
    shift = {}
    metrics = Path("results/metrics.json")
    if metrics.exists():
        shift = {r["policy"]: r for r in json.loads(metrics.read_text())["results"]}
    for tau in TAU_GRID:
        r = score_policy("lr", simulate(test_probs, order, largest, deltas[tau]), matrix, test)
        line = f"   target {tau:.0%}: LR quality {r['relative_quality']:.1%}, latency saved {r['latency_saving']:.1%}"
        if f"shiftai@{int(round(tau * 100))}" in shift:
            s = shift[f"shiftai@{int(round(tau * 100))}"]
            line += f"   | ShiftAI {s['relative_quality']:.1%}, {s['latency_saving']:.1%}"
        print(line + ("" if r["relative_quality"] >= tau else "   <- LR misses target"))

    print("\n2. Agreement cascades vs always using the largest model (test questions)")
    records = {(r["model"], r["qid"]): r for r in load_records(args.run / "records.jsonl")}

    def answer(model, qid):
        q, reply = questions[qid], records[(model, qid)]["reply"]
        return extract_number(reply) if q.kind == "math" else extract_choice(reply, len(q.choices))

    answers = [[answer(m, q) for m in matrix.models] for q in matrix.qids]
    names = [m.split(":")[-1] for m in matrix.models]
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        right, latency, energy = [], [], []
        for i in test:
            x, y = answers[i][a], answers[i][b]
            agree = x is not None and x == y
            ran = [a, b] if agree else [a, b, largest]
            right.append(matrix.correct[i, b if agree else largest])
            latency.append(matrix.latency[i, ran].sum())
            energy.append(matrix.energy[i, ran].sum())
        rel = np.mean(right) / matrix.correct[test, largest].mean()
        lat = np.mean(latency) / matrix.latency[test, largest].mean() - 1
        en = np.mean(energy) / matrix.energy[test, largest].mean() - 1
        print(f"   {names[a]} and {names[b]} agree, else {names[largest]}: quality {rel:.1%}, "
              f"latency {lat:+.1%}, energy {en:+.1%} vs always {names[largest]}")


if __name__ == "__main__":
    main()
