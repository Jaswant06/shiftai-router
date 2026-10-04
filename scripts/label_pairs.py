"""Label a sample of judged pairs by hand and measure how often the judge agrees.

    python scripts/label_pairs.py --run runs/qwen35-open              # label
    python scripts/label_pairs.py --run runs/qwen35-open --report     # agreement only

You see a request and two answers in random order, without knowing which model
wrote which. Type a (A is better), b (B is better), t (tie), s (skip) or
q (save and quit). Labels are saved after every answer, so you can stop and
continue any time.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from shiftai.judge import acceptable, load_judgments, pair_scores
from shiftai.profiler import load_records
from shiftai.tasks import load_questions

SEED = 23
WIDTH = 100


def _show(title: str, text: str, limit: int = 2500) -> None:
    print(f"\n\033[1m{title}\033[0m")
    body = text if len(text) <= limit else text[:limit] + " [...]"
    for paragraph in body.splitlines() or [""]:
        print(textwrap.fill(paragraph, WIDTH) if paragraph.strip() else "")


def load_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_pairs(scores: dict, n: int) -> list[tuple[str, str]]:
    """An even sample across candidate models, in a fixed random order."""
    rng = random.Random(SEED)
    by_model: dict[str, list] = {}
    for model, qid in sorted(scores):
        by_model.setdefault(model, []).append((model, qid))
    per_model = max(1, n // max(1, len(by_model)))
    picked = []
    for pairs in by_model.values():
        picked += rng.sample(pairs, min(per_model, len(pairs)))
    rng.shuffle(picked)
    return picked


def report(labels: list[dict], scores: dict) -> None:
    rows = [(l, scores[(l["candidate"], l["qid"])]) for l in labels if (l["candidate"], l["qid"]) in scores]
    if not rows:
        print("No labelled pairs yet.")
        return
    def category(score: float) -> str:
        return "better" if score > 0.5 else "worse" if score < 0.5 else "tie"
    exact = sum(category(s) == l["human"] for l, s in rows) / len(rows)
    human_ok = [l["human"] != "worse" for l, _ in rows]
    judge_ok = [acceptable(s) for _, s in rows]
    agree = sum(h == j for h, j in zip(human_ok, judge_ok)) / len(rows)
    # Cohen's kappa on the yes/no "good enough" decision the router actually uses.
    p_h, p_j = sum(human_ok) / len(rows), sum(judge_ok) / len(rows)
    chance = p_h * p_j + (1 - p_h) * (1 - p_j)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else float("nan")
    print(f"\n{len(rows)} pairs labelled by hand")
    print(f"  same three-way verdict (better / tie / worse): {exact:.1%}")
    print(f"  same 'good enough' decision:                    {agree:.1%}   (Cohen's kappa {kappa:.2f})")
    print(f"  good enough according to you: {p_h:.1%}, according to the judge: {p_j:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-label judged pairs.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/open.jsonl"))
    parser.add_argument("-n", type=int, default=120, help="pairs to label (default 120)")
    parser.add_argument("--report", action="store_true", help="only print agreement")
    args = parser.parse_args()

    scores = pair_scores(load_judgments(args.run))
    labels_path = args.run / "human_labels.jsonl"
    labels = load_labels(labels_path)
    if args.report:
        report(labels, scores)
        return

    questions = {q.id: q for q in load_questions(args.questions)}
    reference = json.loads((args.run / "meta.json").read_text())["models"][-1]
    replies = {(r["model"], r["qid"]): r["reply"] for r in load_records(args.run / "records.jsonl")}
    done = {(l["candidate"], l["qid"]) for l in labels}
    todo = [p for p in sample_pairs(scores, args.n) if p not in done]
    rng = random.Random(SEED + len(done))

    with labels_path.open("a") as out:
        for i, (candidate, qid) in enumerate(todo, start=len(done) + 1):
            candidate_first = rng.random() < 0.5
            a, b = (candidate, reference) if candidate_first else (reference, candidate)
            print("\n" + "=" * WIDTH + f"\nPair {i} of {args.n}")
            _show("Request", questions[qid].prompt())
            _show("Answer A", replies[(a, qid)])
            _show("Answer B", replies[(b, qid)])
            while True:
                key = input("\nWhich is better?  a / b / t (tie) / s (skip) / q (quit): ").strip().lower()
                if key in {"a", "b", "t", "s", "q"}:
                    break
            if key == "q":
                break
            if key == "s":
                continue
            if key == "t":
                human = "tie"
            else:
                human = "better" if (key == "a") == candidate_first else "worse"
            out.write(json.dumps({"qid": qid, "candidate": candidate, "human": human}) + "\n")
            out.flush()

    report(load_labels(labels_path), scores)


if __name__ == "__main__":
    main()
