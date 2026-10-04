# Question sets

`questions.jsonl` (3,000 questions) and `questions_pilot.jsonl` (60) are
sampled with a fixed seed by `scripts/build_question_set.py` from:

| Source | License | Link |
|---|---|---|
| GSM8K | MIT | https://huggingface.co/datasets/openai/gsm8k |
| MMLU | MIT | https://huggingface.co/datasets/cais/mmlu |
| ARC (Easy and Challenge) | CC BY-SA 4.0 | https://huggingface.co/datasets/allenai/ai2_arc |

Each line holds the question, any answer options, the gold answer and the
source task. The original datasets belong to their authors.
