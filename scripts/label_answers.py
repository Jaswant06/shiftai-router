"""Hand-label single answers (acceptable or not) to validate the pointwise judge.

    python scripts/label_answers.py --run runs/qwen35-open              # label
    python scripts/label_answers.py --run runs/qwen35-open --report     # agreement only

Shows a request, the human-written reference answer and one model's answer,
without saying which model wrote it. Type y (a typical user would be
satisfied), n (not), s (skip) or q (save and quit). The answers are the same
100 that the AI reference annotator labelled, so the report compares you with
the judge and with that annotator. Labels are saved after every answer.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from shiftai.judge import pointwise_labels
from shiftai.profiler import load_records
from shiftai.tasks import load_questions

WIDTH = 100
ITEMS = Path("results/judge_validation/reference_pointwise.jsonl")


def _show(title: str, text: str, limit: int = 2500) -> None:
    print(f"\n\033[1m{title}\033[0m")
    body = text if len(text) <= limit else text[:limit] + " [...]"
    for paragraph in body.splitlines() or [""]:
        print(textwrap.fill(paragraph, WIDTH) if paragraph.strip() else "")


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return (observed - chance) / (1 - chance) if chance < 1 else float("nan")


def report(human: list[dict], run: Path) -> None:
    if not human:
        print("No human labels yet.")
        return
    judge = pointwise_labels(run)
    ai = {(r["model"], r["qid"]): r["acceptable"] for r in _load(ITEMS)}
    print(f"\n{len(human)} answers labelled by a person")
    for name, other in (("the judge", judge), ("the AI reference annotator", ai)):
        pairs = [(h["acceptable"], other[(h["model"], h["qid"])]) for h in human if (h["model"], h["qid"]) in other]
        if pairs:
            a, b = [p[0] for p in pairs], [p[1] for p in pairs]
            agree = sum(x == y for x, y in pairs) / len(pairs)
            print(f"  vs {name:28} agreement {agree:.0%}  kappa {_kappa(a, b):.2f}  "
                  f"(acceptable: person {sum(a)/len(a):.0%}, {name} {sum(b)/len(b):.0%})")
    if len(human) < 30:
        print("  Fewer than 30 labels: too few to report as a result.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-label single answers.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/open.jsonl"))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    out_path = args.run / "human_pointwise.jsonl"
    human = _load(out_path)
    if args.report:
        report(human, args.run)
        return

    questions = {q.id: q for q in load_questions(args.questions)}
    replies = {(r["model"], r["qid"]): r["reply"] for r in load_records(args.run / "records.jsonl")}
    done = {(h["model"], h["qid"]) for h in human}
    items = [(r["model"], r["qid"]) for r in _load(ITEMS) if (r["model"], r["qid"]) not in done]

    with out_path.open("a") as out:
        for i, (model, qid) in enumerate(items, start=len(done) + 1):
            q = questions[qid]
            print("\n" + "=" * WIDTH + f"\nAnswer {i} of {len(items) + len(done)}")
            _show("Request", q.prompt())
            _show("Reference answer (written by a person)", q.answer, limit=1200)
            _show("Answer to judge", replies[(model, qid)])
            while True:
                key = input("\nWould a typical user be satisfied?  y / n / s (skip) / q (quit): ").strip().lower()
                if key in {"y", "n", "s", "q"}:
                    break
            if key == "q":
                break
            if key == "s":
                continue
            out.write(json.dumps({"qid": qid, "model": model, "acceptable": key == "y", "annotator": "human"}) + "\n")
            out.flush()

    report(_load(out_path), args.run)


if __name__ == "__main__":
    main()
