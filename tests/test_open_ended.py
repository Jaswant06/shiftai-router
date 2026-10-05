"""Code grading, judge verdict parsing and judge-based labels."""

import json

from shiftai.evaluate import build_matrix
from shiftai.grading import extract_code, grade, run_tests
from shiftai.judge import judge_pair, pair_scores, parse_verdict
from shiftai.ollama import ChatResult
from shiftai.tasks import Question

TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]
CODE_Q = Question(id="mbpp-1", task="mbpp", question="Write a function to add two numbers.",
                  answer=json.dumps({"tests": TESTS, "setup": ""}))


def test_code_prompt_shows_first_test_only():
    prompt = CODE_Q.prompt()
    assert TESTS[0] in prompt and TESTS[1] not in prompt
    assert CODE_Q.kind == "code" and not CODE_Q.judged


def test_code_passes_and_fails_by_running_tests():
    good = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert grade(CODE_Q, good)
    assert not grade(CODE_Q, bad)
    assert not grade(CODE_Q, "I cannot help with that.")


def test_infinite_loop_times_out():
    assert not run_tests("def add(a, b):\n    while True: pass", TESTS, timeout=1)


def test_unsafe_code_is_never_run():
    sneaky = "import os\ndef add(a, b):\n    os.system('echo hi')\n    return a + b"
    assert not run_tests(sneaky, TESTS)
    assert not run_tests("def add(a, b):\n    open('x', 'w')\n    return a + b", TESTS)


def test_extract_code_prefers_block_with_function():
    reply = "```python\nprint('example')\n```\n```python\ndef add(a, b):\n    return a + b\n```"
    assert "def add" in extract_code(reply)


def test_parse_verdict():
    assert parse_verdict("Answer A misses the date. B is complete.\nVerdict: B") == "B"
    assert parse_verdict("Both are fine.\n**Verdict:** TIE") == "TIE"
    assert parse_verdict("A is said first, but in the end Verdict: A") == "A"
    assert parse_verdict("A") == "A"
    assert parse_verdict(" tie.") == "TIE"
    assert parse_verdict("Answer A is better than B") is None
    assert parse_verdict("") is None


class FixedJudge:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, model, messages, max_tokens=4, **kwargs):
        return ChatResult(model, self.reply, 0, 0, 0, 0, 0, 0, "stop")


def test_judge_pair_maps_letters_to_candidate_side():
    assert judge_pair(FixedJudge("A is right.\nVerdict: A"), "j", "req", "cand", "ref", candidate_first=True)[0] == "better"
    assert judge_pair(FixedJudge("Verdict: A"), "j", "req", "cand", "ref", candidate_first=False)[0] == "worse"
    assert judge_pair(FixedJudge("Verdict: TIE"), "j", "req", "cand", "ref", candidate_first=True)[0] == "tie"
    assert judge_pair(FixedJudge("???"), "j", "req", "cand", "ref", candidate_first=True)[0] == "invalid"


def test_pair_scores_average_both_orders():
    judgments = [
        {"qid": "q", "candidate": "small", "verdict": "better"},
        {"qid": "q", "candidate": "small", "verdict": "worse"},
        {"qid": "q", "candidate": "mid", "verdict": "tie"},
        {"qid": "q", "candidate": "mid", "verdict": "invalid"},
    ]
    assert pair_scores(judgments) == {("small", "q"): 0.5, ("mid", "q"): 0.5}


def _record(model, qid, task, correct=None):
    return {"model": model, "qid": qid, "task": task, "correct": correct, "reply": "x",
            "total_s": 1.0, "load_s": 0.0, "energy_j": None, "output_tokens": 10}


def test_matrix_uses_judge_labels_and_skips_unjudged():
    open_q = Question(id="d1", task="dolly_brainstorming", question="Ideas?", answer="")
    unjudged = Question(id="d2", task="dolly_open_qa", question="Why?", answer="")
    records = [_record(m, q, "dolly") for m in ("small", "large") for q in ("d1", "d2")]
    matrix = build_matrix(records, ["small", "large"], {"d1": open_q, "d2": unjudged}, {("small", "d1"): 0.0})
    assert matrix.qids == ["d1"]
    assert matrix.correct.tolist() == [[False, True]]


def test_parse_acceptability():
    from shiftai.judge import parse_acceptability
    assert parse_acceptability("It contradicts the reference.\nVerdict: UNACCEPTABLE") is False
    assert parse_acceptability("Correct and complete.\n**Verdict:** ACCEPTABLE") is True
    assert parse_acceptability("Looks fine to me.") is None


def test_pointwise_labels_replace_pairwise_and_score_the_largest_model_too():
    open_q = Question(id="d1", task="dolly_open_qa", question="Why?", answer="Because.")
    records = [_record(m, "d1", "dolly") for m in ("small", "large")]
    absolute = {("small", "d1"): True, ("large", "d1"): False}
    matrix = build_matrix(records, ["small", "large"], {"d1": open_q}, judged={}, absolute=absolute)
    assert matrix.correct.tolist() == [[True, False]]
