"""Sample the profiling question set from GSM8K, MMLU and ARC.

The sample is drawn once with a fixed seed and saved as JSONL, so every
profiling run, on any machine, uses exactly the same questions.

    python scripts/build_question_set.py                 # full set, ~3,000 questions
    python scripts/build_question_set.py --scale 0.02    # tiny pilot set
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from shiftai.tasks import LETTERS, Question, save_questions

SEED = 13
# Questions per task at scale 1.0.
SIZES = {"gsm8k": 1000, "mmlu": 1000, "arc_easy": 500, "arc_challenge": 500}


def _sample(rows: list, n: int, rng: random.Random) -> list:
    return rng.sample(rows, min(n, len(rows)))


def gsm8k(n: int, rng: random.Random) -> list[Question]:
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    indexed = list(enumerate(rows))
    return [
        Question(
            id=f"gsm8k-{i}",
            task="gsm8k",
            question=row["question"],
            answer=row["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for i, row in _sample(indexed, n, rng)
    ]


def mmlu(n: int, rng: random.Random) -> list[Question]:
    rows = list(load_dataset("cais/mmlu", "all", split="test"))
    indexed = list(enumerate(rows))
    return [
        Question(
            id=f"mmlu-{i}",
            task="mmlu",
            question=row["question"],
            choices=list(row["choices"]),
            answer=LETTERS[row["answer"]],
            subject=row["subject"],
        )
        for i, row in _sample(indexed, n, rng)
    ]


def arc(config: str, task: str, n: int, rng: random.Random) -> list[Question]:
    rows = list(load_dataset("allenai/ai2_arc", config, split="test"))
    questions = []
    for row in _sample(rows, n, rng):
        labels = row["choices"]["label"]
        # A few ARC items label options 1-4 instead of A-D; normalise to letters.
        answer = LETTERS[labels.index(row["answerKey"])]
        questions.append(
            Question(
                id=f"{task}-{row['id']}",
                task=task,
                question=row["question"],
                choices=list(row["choices"]["text"]),
                answer=answer,
            )
        )
    return questions


def build(scale: float) -> list[Question]:
    rng = random.Random(SEED)
    sizes = {task: max(1, round(n * scale)) for task, n in SIZES.items()}
    questions = (
        gsm8k(sizes["gsm8k"], rng)
        + mmlu(sizes["mmlu"], rng)
        + arc("ARC-Easy", "arc_easy", sizes["arc_easy"], rng)
        + arc("ARC-Challenge", "arc_challenge", sizes["arc_challenge"], rng)
    )
    rng.shuffle(questions)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, default=1.0, help="fraction of the full sizes")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or Path("data") / ("questions.jsonl" if args.scale == 1.0 else "questions_pilot.jsonl")
    questions = build(args.scale)
    save_questions(questions, out)
    counts = {}
    for q in questions:
        counts[q.task] = counts.get(q.task, 0) + 1
    print(f"Wrote {len(questions)} questions to {out}: {counts}")


if __name__ == "__main__":
    main()
