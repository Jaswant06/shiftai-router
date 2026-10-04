"""Judge the open-ended replies of a profiling run.

    python scripts/judge_run.py --run runs/qwen35-open --questions data/open.jsonl

Every smaller model's reply is compared with the largest model's reply, twice
(answers swapped). Resumable: run it again to continue where it stopped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from shiftai.judge import DEFAULT_JUDGE, acceptable, load_judgments, pair_scores, run_judging
from shiftai.tasks import load_questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge open-ended replies against the largest model.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/open.jsonl"))
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    args = parser.parse_args()

    models = json.loads((args.run / "meta.json").read_text())["models"]
    reference, candidates = models[-1], models[:-1]
    questions = {q.id: q for q in load_questions(args.questions)}
    print(f"Judge {args.judge}: comparing {', '.join(candidates)} against {reference}")

    def show(j, i, n):
        print(f"[{i}/{n}] {j.candidate:14} {j.qid:16} {'cand first' if j.candidate_first else 'ref first ':10} "
              f"{j.verdict:8} {j.seconds:4.1f}s", flush=True)

    run_judging(args.run, questions, reference, candidates, judge=args.judge, on_judgment=show)

    judgments = load_judgments(args.run)
    verdicts = Counter(j["verdict"] for j in judgments)
    flips = Counter()
    by_pair: dict[tuple, list[str]] = {}
    for j in judgments:
        by_pair.setdefault((j["candidate"], j["qid"]), []).append(j["verdict"])
    for pair in by_pair.values():
        if len(pair) == 2 and pair[0] != pair[1]:
            flips["order changed the verdict"] += 1
    scores = pair_scores(judgments)
    print(f"\nVerdicts: {dict(verdicts)}; pairs where swapping order changed the verdict: "
          f"{flips['order changed the verdict']} of {len(by_pair)}")
    for cand in candidates:
        mine = [s for (c, _), s in scores.items() if c == cand]
        if mine:
            print(f"  {cand:14} judged at least as good as {reference}: {sum(map(acceptable, mine)) / len(mine):.1%}")


if __name__ == "__main__":
    main()
