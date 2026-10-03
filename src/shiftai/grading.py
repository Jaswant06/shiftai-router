"""Deterministic grading: pull the final answer out of a reply and compare it.

Small models do not always follow the requested format, so each extractor
tries the strict format first and then falls back to looser patterns. A reply
that yields no answer at all is graded wrong, not skipped.
"""

from __future__ import annotations

import re

from .tasks import LETTERS, Question

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
    # Letters are matched case-sensitively so the article "a" never counts as A.
    patterns = [
        rf"^\(?([{valid}])\)?(?:[.):\s]|$)",                         # "B", "B.", "(B) ..."
        rf"(?i:answer)\s*(?i:is)?\s*[:=]?\s*\(?\**([{valid}])\b",   # "Answer: B", "the answer is B"
        rf"(?i:option)\s*\(?([{valid}])\b",                          # "option B"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)
    standalone = set(re.findall(rf"\b([{valid}])\b", text))
    return standalone.pop() if len(standalone) == 1 else None


def grade(question: Question, reply: str) -> bool:
    """True when the reply's final answer matches the gold answer."""
    if question.kind == "math":
        predicted = extract_number(reply)
        gold = _to_float(question.answer)
        return predicted is not None and gold is not None and abs(predicted - gold) < 1e-6
    return extract_choice(reply, len(question.choices)) == question.answer.upper()
