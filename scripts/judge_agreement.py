"""How often does the pointwise judge agree with an independent set of labels?

    python scripts/judge_agreement.py --run runs/qwen35-open --labels reference_pointwise.jsonl

The labels file holds one line per judged reply: {"qid", "model", "acceptable"}.
Reports raw agreement, Cohen's kappa (agreement beyond chance), and the
acceptance rate per model according to each source.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from shiftai.judge import pointwise_labels


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return (observed - chance) / (1 - chance) if chance < 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", default="reference_pointwise.jsonl")
    args = parser.parse_args()

    judge = pointwise_labels(args.run)
    with (args.run / args.labels).open() as f:
        reference = [json.loads(line) for line in f if line.strip()]
    pairs = [(r["model"], r["acceptable"], judge[(r["model"], r["qid"])]) for r in reference if (r["model"], r["qid"]) in judge]
    if not pairs:
        print("None of the labelled replies have been judged yet.")
        return

    ref = [p[1] for p in pairs]
    jud = [p[2] for p in pairs]
    print(f"{len(pairs)} replies judged and labelled")
    print(f"  agreement {sum(x == y for x, y in zip(ref, jud)) / len(pairs):.1%}   Cohen's kappa {kappa(ref, jud):.2f}")
    print(f"  acceptable: labels {sum(ref) / len(ref):.1%}, judge {sum(jud) / len(jud):.1%}")
    by_model = defaultdict(list)
    for model, r, j in pairs:
        by_model[model].append((r, j))
    for model, rows in sorted(by_model.items()):
        print(f"  {model:14} n={len(rows):3}  labels {sum(r for r, _ in rows) / len(rows):5.1%}  "
              f"judge {sum(j for _, j in rows) / len(rows):5.1%}  agree {sum(r == j for r, j in rows) / len(rows):5.1%}")


if __name__ == "__main__":
    main()
