"""Judge every model's open-ended replies on their own (acceptable or not).

    python scripts/judge_pointwise.py --run runs/qwen35-open --questions data/open.jsonl

Resumable: run it again to continue where it stopped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from shiftai.judge import DEFAULT_JUDGE, load_pointwise, run_pointwise
from shiftai.tasks import load_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Pointwise judging of open-ended replies.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/open.jsonl"))
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    args = parser.parse_args()

    models = json.loads((args.run / "meta.json").read_text())["models"]
    questions = {q.id: q for q in load_questions(args.questions)}

    def show(row, i, n):
        verdict = {True: "acceptable", False: "unacceptable", None: "unreadable"}[row["acceptable"]]
        print(f"[{i}/{n}] {row['model']:14} {row['qid']:16} {verdict:12} {row['seconds']:4.1f}s", flush=True)

    run_pointwise(args.run, questions, models, judge=args.judge, on_verdict=show)

    by_model = defaultdict(list)
    for v in load_pointwise(args.run):
        by_model[v["model"]].append(v["acceptable"])
    print()
    for model in models:
        verdicts = by_model.get(model, [])
        readable = [v for v in verdicts if v is not None]
        if readable:
            print(f"  {model:14} acceptable {sum(readable) / len(readable):.1%}  "
                  f"(unreadable {len(verdicts) - len(readable)})")


if __name__ == "__main__":
    main()
