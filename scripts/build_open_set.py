"""Sample the open-ended profiling set (800 prompts) from Dolly and MBPP.

Dolly gives real human-written requests across eight categories; MBPP gives
short Python tasks with unit tests. Sampled once with a fixed seed.

    python scripts/build_open_set.py                 # 600 Dolly + 200 MBPP
    python scripts/build_open_set.py --scale 0.03    # tiny pilot set
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from shiftai.tasks import Question, save_questions

SEED = 17
# 75 per category keeps generation plus judging to about one night.
DOLLY_PER_CATEGORY = 75
MBPP_COUNT = 200
# Long reading passages would dominate profiling time; real chat prompts are
# usually short. Requests above this length are skipped.
MAX_REQUEST_CHARS = 3000


def dolly(per_category: int, rng: random.Random) -> list[Question]:
    rows = load_dataset("databricks/databricks-dolly-15k", split="train")
    by_category: dict[str, list] = {}
    for i, row in enumerate(rows):
        request = row["instruction"].strip()
        if row["context"].strip():
            request += "\n\n" + row["context"].strip()
        if len(request) <= MAX_REQUEST_CHARS:
            by_category.setdefault(row["category"], []).append((i, request, row["response"]))
    questions = []
    for category in sorted(by_category):
        for i, request, response in rng.sample(by_category[category], min(per_category, len(by_category[category]))):
            questions.append(
                Question(id=f"dolly-{i}", task=f"dolly_{category}", question=request, answer=response, subject=category)
            )
    return questions


def mbpp(n: int, rng: random.Random) -> list[Question]:
    rows = list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    return [
        Question(
            id=f"mbpp-{row['task_id']}",
            task="mbpp",
            question=row["text"].strip(),
            answer=json.dumps({"tests": row["test_list"], "setup": row["test_setup_code"]}),
        )
        for row in rng.sample(rows, min(n, len(rows)))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rng = random.Random(SEED)
    questions = dolly(max(1, round(DOLLY_PER_CATEGORY * args.scale)), rng) + mbpp(max(1, round(MBPP_COUNT * args.scale)), rng)
    rng.shuffle(questions)
    out = args.out or Path("data") / ("open.jsonl" if args.scale == 1.0 else "open_pilot.jsonl")
    save_questions(questions, out)
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.task] = counts.get(q.task, 0) + 1
    print(f"Wrote {len(questions)} prompts to {out}: {counts}")


if __name__ == "__main__":
    main()
