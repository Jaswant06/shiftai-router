"""Deterministic grading: pull the final answer out of a reply and check it.

Small models do not always follow the requested format, so each extractor
tries the strict format first and then falls back to looser patterns. A reply
that yields no answer at all is graded wrong, not skipped.

Code is graded by running the task's unit tests in a separate Python process,
in an empty temporary folder, with a time limit. Model-written code that
reaches for the file system, the shell or the network is not run at all and
counts as a failure; MBPP tasks never need those.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile

from .tasks import LETTERS, Question

CODE_TIMEOUT_S = 10
_CODE_BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)
_UNSAFE = re.compile(
    r"\b(?:import|from)\s+(?:os|subprocess|shutil|socket|pathlib|requests|urllib|http|ctypes|multiprocessing)\b"
    r"|__import__|\bopen\s*\(|\bexec\s*\(|\beval\s*\(|\bcompile\s*\("
)

_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_number(reply: str) -> float | None:
    """The final numeric answer: the 'Answer:' line if present, else the last number."""
    tagged = re.findall(rf"answer\s*[:=]?\s*\**\s*({_NUMBER})", reply, re.IGNORECASE)
    if tagged:
        return _to_float(tagged[-1])
    numbers = re.findall(_NUMBER, reply)
    return _to_float(numbers[-1]) if numbers else None


def extract_choice(reply: str, n_choices: int) -> str | None:
    """The chosen option letter, restricted to the letters actually offered."""
    valid = LETTERS[:n_choices]
    text = reply.strip()
    # Letters are matched case-sensitively so the article "a" never counts as A,
    # and a capital letter followed by a lowercase word ("A stem-boring beetle")
    # is read as a sentence, not as choosing option A.
    patterns = [
        rf"^\(?([{valid}])\)?(?:[.):]|\s*$)",                       # "B", "B.", "(B) ..."
        rf"(?i:answer)\s*(?i:is)?\s*[:=]?\s*\(?\**([{valid}])\b",   # "Answer: B", "the answer is B"
        rf"(?i:option)\s*\(?([{valid}])\b",                          # "option B"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)
    standalone = set(re.findall(rf"\b([{valid}])\b(?!\s+[a-z])", text))
    return standalone.pop() if len(standalone) == 1 else None


def extract_code(reply: str) -> str:
    """The code block that defines a function, else the whole reply."""
    blocks = _CODE_BLOCK.findall(reply)
    for block in blocks:
        if "def " in block:
            return block
    return blocks[0] if blocks else reply


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = CODE_TIMEOUT_S) -> bool:
    """True when the code passes every assert. Unsafe code is never executed."""
    if not tests or _UNSAFE.search(code):
        return False
    program = "\n\n".join([setup, code, "\n".join(tests)])
    with tempfile.TemporaryDirectory() as folder:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=folder, capture_output=True, timeout=timeout, env={},
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def grade(question: Question, reply: str) -> bool:
    """True when the reply's final answer matches the gold answer.

    Open-ended questions have no gold answer; they are judged later (judge.py).
    """
    if question.judged:
        raise ValueError(f"{question.id} is open-ended and is graded by the judge")
    if question.kind == "code":
        return run_tests(extract_code(reply), question.tests, question.test_setup)
    if question.kind == "math":
        predicted = extract_number(reply)
        gold = _to_float(question.answer)
        return predicted is not None and gold is not None and abs(predicted - gold) < 1e-6
    return extract_choice(reply, len(question.choices)) == question.answer.upper()
