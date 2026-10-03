"""Answer extraction and grading, including the messy replies small models give."""

from shiftai.grading import extract_choice, extract_number, grade
from shiftai.tasks import Question


def test_number_prefers_answer_line():
    reply = "April: 48, May: 24.\nTotal = 48 + 24 = 72\n\nAnswer: 72"
    assert extract_number(reply) == 72


def test_number_handles_commas_dollars_and_decimals():
    assert extract_number("Answer: $1,250") == 1250
    assert extract_number("Answer: 3.5") == 3.5
    assert extract_number("**Answer:** -4") == -4


def test_number_falls_back_to_last_number():
    assert extract_number("She has 5 apples, then buys 7, so 12") == 12
    assert extract_number("I am not sure.") is None


def test_choice_formats():
    assert extract_choice("B", 4) == "B"
    assert extract_choice("(C) Selenocysteine", 4) == "C"
    assert extract_choice("The answer is D.", 4) == "D"
    assert extract_choice("Answer: A", 4) == "A"


def test_choice_ignores_article_a_and_unoffered_letters():
    assert extract_choice("a plant needs sunlight", 4) is None
    assert extract_choice("E", 4) is None


def test_choice_ignores_a_reply_that_repeats_the_question():
    # Seen from qwen3.5:0.8b in the real profiling run.
    assert extract_choice("A stem-boring beetle consumes wood by eating", 4) is None
    assert extract_choice("A. stem-boring beetle", 4) == "A"


def test_grade_math_and_choice():
    math = Question(id="m", task="gsm8k", question="?", answer="72")
    assert grade(math, "so the answer is\nAnswer: 72")
    assert not grade(math, "Answer: 70")

    choice = Question(id="c", task="mmlu", question="?", answer="B", choices=["w", "x", "y", "z"])
    assert grade(choice, "B")
    assert not grade(choice, "C")
    assert not grade(choice, "")


def test_prompts_list_choices_with_letters():
    q = Question(id="c", task="arc_easy", question="Pick one", answer="A", choices=["red", "blue"])
    prompt = q.prompt()
    assert "A. red" in prompt and "B. blue" in prompt
    assert q.max_tokens == 16
