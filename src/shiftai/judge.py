"""Judge open-ended replies, on their own (pointwise) or against the largest model (pairwise).

Pointwise judging is what the router trains on. Each reply is read on its own,
next to Dolly's human-written reference answer for fact checking, and marked
acceptable when a typical user would be satisfied with it, so every
model, the largest included, gets a real quality score, just like accuracy on
the benchmark questions. Showing one answer at a time also removes position
bias by construction.

Pairwise judging (below) was the first design and is kept for comparison.
It defined quality as "at least as good as the largest model's reply", which
gives the largest model 100% by construction and marks a slightly less
polished but perfectly fine reply as a failure. Mixed with benchmark accuracy,
that mismatch made the router send every open-ended prompt to the largest
model and miss its quality targets.

Open-ended requests ("write a short poem", "summarise this paragraph") have no
single right answer, so quality is defined relative to the reference model,
which matches ShiftAI's relative quality target: a smaller model's reply is
acceptable when a judge rates it at least as good as the largest model's.

Known judge biases and what is done about them:
  * Position bias: every pair is judged twice with the answers swapped, and the
    two verdicts are averaged. The judge also explains its comparison in a
    few sentences before giving a verdict. On the pilot, a bare one-word
    verdict picked whichever answer came first about 90% of the time and agreed
    with itself after a swap only 18% of the time; brief reasoning raised that
    to 69% with no first-answer preference.
  * Self-preference: the judge comes from a different model family than the
    models being compared (Gemma judging Qwen by default).
  * Length bias: the rubric says length alone is not quality.
  * Overall reliability: a sample of pairs is labelled by a human
    (scripts/label_pairs.py) and agreement with the judge is reported.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .ollama import OllamaClient
from .profiler import load_records
from .tasks import Question

DEFAULT_JUDGE = "gemma4:e4b"
MAX_REQUEST_CHARS = 6000
JUDGE_MAX_TOKENS = 256

JUDGE_PROMPT = """You are an impartial judge. Compare two AI assistant answers to the same user request.

Judge which answer better serves the user, considering, in order:
1. Correctness: no factual or logical errors.
2. Instruction following: does what was asked, in the requested form.
3. Completeness: covers what the request needs.
4. Relevance and clarity: no padding or off-topic content.
Length alone is not quality. A shorter answer that fully does the job is as good as a longer one.
If both answers are about equally good, or equally bad, say TIE.

[User request]
{request}

[Answer A]
{answer_a}

[Answer B]
{answer_b}

First explain your comparison in at most three short sentences.
Then on the last line write exactly one of: Verdict: A, Verdict: B, Verdict: TIE"""

POINTWISE_PROMPT = """You are a strict evaluator. Read a user request, a reference answer written by a person, and one AI assistant answer.

Decide whether the AI answer is ACCEPTABLE: a typical user who asked this would be satisfied with it.
An acceptable answer is correct (no significant factual or logical errors), does what was asked in the requested form, and covers what the request needs.
Use the reference to check facts: if the AI answer contradicts the reference on a key fact, it is UNACCEPTABLE.
The AI answer does not need to match the reference's wording, length or examples. For creative, opinion or brainstorming requests the reference is just one possible good answer, so judge those on whether the request is fulfilled.
Minor style issues are fine. Fabricated facts, wrong answers, ignoring the request, or being cut off before the essential content make it UNACCEPTABLE.
Length alone is not quality.

[User request]
{request}

[Reference answer]
{reference}

[AI answer]
{answer}

First explain your judgement in at most three short sentences.
Then on the last line write exactly one of: Verdict: ACCEPTABLE, Verdict: UNACCEPTABLE"""


def parse_acceptability(raw: str) -> bool | None:
    """True / False from the pointwise judge's reply, or None if unreadable."""
    tagged = re.findall(r"verdict\s*[:=]?\s*\**\s*(UNACCEPTABLE|ACCEPTABLE)\b", raw, re.IGNORECASE)
    if tagged:
        return tagged[-1].upper() == "ACCEPTABLE"
    return None


def judge_answer(
    client: OllamaClient, judge: str, request: str, answer: str, reference: str = ""
) -> tuple[bool | None, str]:
    """Ask the judge whether one reply is acceptable. Returns (verdict, raw reply).

    A small judge cannot check facts from memory (it accepted "a baker's dozen
    is 12" without one), so a human-written reference answer is included.
    """
    prompt = POINTWISE_PROMPT.format(
        request=request[:MAX_REQUEST_CHARS], reference=reference[:MAX_REQUEST_CHARS] or "(none)", answer=answer
    )
    raw = client.chat(judge, [{"role": "user", "content": prompt}], max_tokens=JUDGE_MAX_TOKENS).text
    return parse_acceptability(raw), raw


def run_pointwise(
    run_dir: str | Path,
    questions: dict[str, Question],
    models: list[str],
    judge: str = DEFAULT_JUDGE,
    client: OllamaClient | None = None,
    on_verdict: Callable[[dict, int, int], None] | None = None,
) -> Path:
    """Judge every model's open-ended reply on its own. Resumable; writes pointwise.jsonl."""
    client = client or OllamaClient()
    run_dir = Path(run_dir)
    out_path = run_dir / "pointwise.jsonl"
    replies = {(r["model"], r["qid"]): r["reply"] for r in load_records(run_dir / "records.jsonl")}
    done = {(v["model"], v["qid"]) for v in load_pointwise(run_dir)}
    jobs = sorted(
        (m, q) for (m, q) in replies if m in models and q in questions and questions[q].judged
    )
    total, count = len(jobs), len(done)
    with out_path.open("a") as out:
        for model, qid in jobs:
            if (model, qid) in done:
                continue
            t0 = time.monotonic()
            q = questions[qid]
            verdict, raw = judge_answer(client, judge, q.prompt(), replies[(model, qid)], reference=q.answer)
            row = {"qid": qid, "model": model, "acceptable": verdict, "raw": raw, "judge": judge,
                   "seconds": time.monotonic() - t0}
            out.write(json.dumps(row) + "\n")
            out.flush()
            count += 1
            if on_verdict:
                on_verdict(row, count, total)
    return out_path


def load_pointwise(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "pointwise.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pointwise_labels(run_dir: str | Path) -> dict[tuple[str, str], bool]:
    """(model, qid) -> acceptable, skipping unreadable verdicts."""
    return {(v["model"], v["qid"]): v["acceptable"] for v in load_pointwise(run_dir) if v["acceptable"] is not None}


# Verdict from the candidate's point of view: better, tie or worse than the reference.
SCORE = {"better": 1.0, "tie": 0.5, "worse": 0.0}


@dataclass
class Judgment:
    """One judge call on one (question, candidate) pair in one answer order."""

    qid: str
    candidate: str
    reference: str
    candidate_first: bool
    verdict: str  # "better", "tie", "worse" or "invalid" (from the candidate's side)
    raw: str
    judge: str
    seconds: float


def parse_verdict(raw: str) -> str | None:
    """'A', 'B' or 'TIE' from the judge's reply, or None if it is unreadable.

    The last "Verdict: X" line wins; a bare one-word reply is also accepted.
    """
    tagged = re.findall(r"verdict\s*[:=]?\s*\**\s*(TIE|A|B)\b", raw, re.IGNORECASE)
    if tagged:
        return tagged[-1].upper()
    text = raw.strip().upper()
    match = re.fullmatch(r"\W*(TIE|A|B)\W*", text)
    return match.group(1) if match else None


def judge_pair(
    client: OllamaClient,
    judge: str,
    request: str,
    candidate_reply: str,
    reference_reply: str,
    candidate_first: bool,
) -> tuple[str, str]:
    """Ask the judge once. Returns (verdict from the candidate's side, raw reply)."""
    a, b = (candidate_reply, reference_reply) if candidate_first else (reference_reply, candidate_reply)
    prompt = JUDGE_PROMPT.format(request=request[:MAX_REQUEST_CHARS], answer_a=a, answer_b=b)
    raw = client.chat(judge, [{"role": "user", "content": prompt}], max_tokens=JUDGE_MAX_TOKENS).text
    choice = parse_verdict(raw)
    if choice is None:
        return "invalid", raw
    if choice == "TIE":
        return "tie", raw
    candidate_letter = "A" if candidate_first else "B"
    return ("better" if choice == candidate_letter else "worse"), raw


def run_judging(
    run_dir: str | Path,
    questions: dict[str, Question],
    reference: str,
    candidates: list[str],
    judge: str = DEFAULT_JUDGE,
    client: OllamaClient | None = None,
    on_judgment: Callable[[Judgment, int, int], None] | None = None,
) -> Path:
    """Judge every open-ended reply in a run against the reference model's reply.

    Results go to judgments.jsonl in the run folder, one line per judge call,
    and the run resumes where it stopped.
    """
    client = client or OllamaClient()
    run_dir = Path(run_dir)
    out_path = run_dir / "judgments.jsonl"
    replies = {(r["model"], r["qid"]): r["reply"] for r in load_records(run_dir / "records.jsonl")}
    open_qids = sorted(q for q in {qid for _, qid in replies} if q in questions and questions[q].judged)

    done = {(j["qid"], j["candidate"], j["candidate_first"]) for j in load_judgments(run_dir)}
    jobs = [
        (qid, cand, first)
        for qid in open_qids
        for cand in candidates
        for first in (True, False)
        if (cand, qid) in replies and (reference, qid) in replies
    ]
    total, count = len(jobs), len(done)
    with out_path.open("a") as out:
        for qid, cand, first in jobs:
            if (qid, cand, first) in done:
                continue
            t0 = time.monotonic()
            verdict, raw = judge_pair(
                client, judge, questions[qid].prompt(), replies[(cand, qid)], replies[(reference, qid)], first
            )
            judgment = Judgment(qid, cand, reference, first, verdict, raw, judge, time.monotonic() - t0)
            out.write(json.dumps(asdict(judgment)) + "\n")
            out.flush()
            count += 1
            if on_judgment:
                on_judgment(judgment, count, total)
    return out_path


def load_judgments(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "judgments.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_scores(judgments: list[dict]) -> dict[tuple[str, str], float]:
    """Average score per (candidate, qid) over both answer orders.

    1.0 = better than the reference in both orders, 0.5 = tie (or a split
    decision), 0.0 = worse in both. Invalid verdicts are ignored.
    """
    collected: dict[tuple[str, str], list[float]] = {}
    for j in judgments:
        if j["verdict"] in SCORE:
            collected.setdefault((j["candidate"], j["qid"]), []).append(SCORE[j["verdict"]])
    return {key: sum(v) / len(v) for key, v in collected.items()}


def acceptable(score: float) -> bool:
    """A reply is good enough when it is judged at least as good as the reference."""
    return score >= 0.5
