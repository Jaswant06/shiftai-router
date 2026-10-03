"""Benchmark questions with answers a script can check, and how to prompt them.

Three task families give a spread of difficulty: GSM8K (multi-step math word
problems), MMLU (multiple choice across 57 subjects) and ARC (grade-school
science, Easy and Challenge sets).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LETTERS = "ABCDEFGH"

MATH_TASKS = {"gsm8k"}
CHOICE_TASKS = {"mmlu", "arc_easy", "arc_challenge"}

# Generation limits per task family. Math needs room for working (512 tokens
# cut off some verbose 9B replies in the pilot); multiple choice only needs the
# letter, which keeps profiling time sensible.
MAX_TOKENS = {"math": 1024, "choice": 16}

MATH_INSTRUCTION = (
    "Solve the problem. Show brief working, then finish with a final line "
    "in the form 'Answer: <number>'."
)
CHOICE_INSTRUCTION = "Reply with only the letter of the correct option."


@dataclass
class Question:
    """One gradable item: the text, optional choices, and the gold answer."""

    id: str
    task: str
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    subject: str = ""

    @property
    def kind(self) -> str:
        return "math" if self.task in MATH_TASKS else "choice"

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS[self.kind]

    def prompt(self) -> str:
        """The user message sent to every model, identical across the ladder."""
        if self.kind == "math":
            return f"{MATH_INSTRUCTION}\n\nProblem: {self.question}"
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(self.choices))
        return f"{CHOICE_INSTRUCTION}\n\nQuestion: {self.question}\n{options}"

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.prompt()}]


def save_questions(questions: list[Question], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in questions:
            f.write(json.dumps(asdict(q)) + "\n")


def load_questions(path: str | Path) -> list[Question]:
    with Path(path).open() as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]
