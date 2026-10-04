"""Run every model on every question and log quality, latency and energy.

The output is one JSON line per (model, question), written as it goes, so an
overnight run that is interrupted resumes where it stopped instead of starting
over. Models run one at a time over all questions, so measurements reflect a
model that is already loaded; cold-load cost is measured separately during
setup calibration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .energy import PowerSampler
from .grading import grade
from .ollama import OllamaClient
from .system import machine_info, resource_state
from .tasks import Question


@dataclass
class Record:
    """One model's attempt at one question."""

    qid: str
    task: str
    model: str
    correct: bool | None  # None for open-ended questions, which are judged later
    reply: str
    wall_s: float
    total_s: float
    load_s: float
    ttft_s: float
    prompt_tokens: int
    output_tokens: int
    output_s: float
    truncated: bool
    energy_j: float | None
    mean_watts: float | None
    energy_source: str


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _done_keys(path: Path) -> set[tuple[str, str]]:
    return {(r["model"], r["qid"]) for r in load_records(path)}


def run_profile(
    models: list[str],
    questions: list[Question],
    out_dir: str | Path,
    client: OllamaClient | None = None,
    sampler: PowerSampler | None = None,
    seed: int = 0,
    on_record: Callable[[Record, int, int], None] | None = None,
) -> Path:
    """Profile each model on each question; returns the records file path."""
    client = client or OllamaClient()
    sampler = sampler or PowerSampler()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        meta = {
            "machine": machine_info(),
            "resources_at_start": resource_state().as_dict(),
            "energy_source": sampler.source,
            "seed": seed,
            "models": models,
            "n_questions": len(questions),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    done = _done_keys(records_path)
    total = len(models) * len(questions)
    count = len(done)

    with sampler, records_path.open("a") as out:
        for model in models:
            for q in questions:
                if (model, q.id) in done:
                    continue
                t0 = time.monotonic()
                result = client.chat(model, q.messages(), max_tokens=q.max_tokens, seed=seed)
                t1 = time.monotonic()
                record = Record(
                    qid=q.id,
                    task=q.task,
                    model=model,
                    correct=None if q.judged else grade(q, result.text),
                    reply=result.text,
                    wall_s=t1 - t0,
                    total_s=result.total_s,
                    load_s=result.load_s,
                    ttft_s=result.ttft_s,
                    prompt_tokens=result.prompt_tokens,
                    output_tokens=result.output_tokens,
                    output_s=result.output_s,
                    truncated=result.truncated,
                    energy_j=sampler.energy_j(t0, t1),
                    mean_watts=sampler.mean_watts(t0, t1),
                    energy_source=sampler.source,
                )
                out.write(json.dumps(record.__dict__) + "\n")
                out.flush()
                count += 1
                if on_record:
                    on_record(record, count, total)
    return records_path
