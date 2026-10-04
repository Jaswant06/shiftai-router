"""Profiling questions, how to prompt them, and how each kind is graded.

Benchmark questions have answers a script can check: GSM8K (multi-step math
word problems), MMLU (multiple choice across 57 subjects) and ARC
(grade-school science, Easy and Challenge sets).

Open-ended questions look like what people actually ask: Dolly (human-written
requests for writing, brainstorming, summarization, extraction, classification
and question answering) and MBPP (short Python coding tasks). MBPP is graded by
running its unit tests. Dolly has no single right answer, so its replies are
compared with the largest model's reply by a judge model (see judge.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LETTERS = "ABCDEFGH"

MATH_TASKS = {"gsm8k"}
CHOICE_TASKS = {"mmlu", "arc_easy", "arc_challenge"}
CODE_TASKS = {"mbpp"}
OPEN_PREFIX = "dolly_"

# Generation limits per kind. Math needs room for working (512 tokens cut off
# some verbose 9B replies in the pilot); multiple choice only needs the letter.
# Open-ended replies run long: in the pilot, 384 tokens cut off every creative
# writing answer, while uncapped answers were 400 to 1,200 tokens, so 1,024
# lets nearly all of them finish.
MAX_TOKENS = {"math": 1024, "choice": 16, "code": 512, "open": 1024}

MATH_INSTRUCTION = (
    "Solve the problem. Show brief working, then finish with a final line "
    "in the form 'Answer: <number>'."
)
CHOICE_INSTRUCTION = "Reply with only the letter of the correct option."
CODE_INSTRUCTION = "Reply with only the Python code in a ```python code block."


@dataclass
class Question:
    """One profiling item: the text, optional choices, and how to grade it.

    `answer` holds the gold answer: a number for math, a letter for multiple
    choice, JSON {"tests": [...asserts], "setup": "..."} for code, and a
    human-written reference (kept for inspection, not used for grading) for
    open-ended.
    """

    id: str
    task: str
    question: str
    answer: str
    choices: list[str] = field(default_factory=list)
    subject: str = ""

    @property
    def kind(self) -> str:
        if self.task in MATH_TASKS:
            return "math"
        if self.task in CODE_TASKS:
            return "code"
        if self.task.startswith(OPEN_PREFIX):
            return "open"
        return "choice"

    @property
    def judged(self) -> bool:
        """True when quality comes from a judge instead of a script."""
        return self.kind == "open"

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS[self.kind]

    @property
    def tests(self) -> list[str]:
        return json.loads(self.answer)["tests"] if self.kind == "code" else []

    @property
    def test_setup(self) -> str:
        return json.loads(self.answer).get("setup", "") if self.kind == "code" else ""

    def prompt(self) -> str:
        """The user message sent to every model, identical across the ladder."""
        if self.kind == "math":
            return f"{MATH_INSTRUCTION}\n\nProblem: {self.question}"
        if self.kind == "code":
            return f"{self.question}\nYour code should pass this test:\n{self.tests[0]}\n{CODE_INSTRUCTION}"
        if self.kind == "open":
            return self.question
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(self.choices))
        return f"{CHOICE_INSTRUCTION}\n\nQuestion: {self.question}\n{options}"

    def feature_text(self) -> str:
        """What a real user would type: the question and any options, without
        the benchmark's answer-format instruction. The router learns from this,
        so its training inputs look like live prompts."""
        if not self.choices:
            return self.question
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(self.choices))
        return f"{self.question}\n{options}"

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
