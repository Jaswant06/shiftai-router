"""Command-line interface.

    shiftai models                      list installed models and what is loaded
    shiftai setup                       measure each model's cost on this machine
    shiftai ask "prompt" [--explain]    route a prompt and print the answer
    shiftai bench --models a b c        profile models on the benchmark questions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .ollama import OllamaUnavailable


def _sampler(enabled: bool):
    from .energy import PowerSampler, best_sampler

    if not enabled:
        return PowerSampler()
    sampler = best_sampler()
    if sampler.source == "none":
        print("Energy telemetry not available; energy will be reported as not measured.")
        if sys.platform == "darwin":
            print("  On a Mac, run `sudo -v` first to let ShiftAI read powermetrics.")
    else:
        print(f"Measuring energy with {sampler.source}.")
    return sampler


def cmd_models(args) -> None:
    from .discovery import list_models, loaded_models

    loaded = loaded_models()
    for m in list_models():
        params = f"{m.params_b:g}B" if m.params_b is not None else "?"
        role = "chat" if m.can_chat else ",".join(m.capabilities) or "?"
        mark = "  [loaded]" if m.name in loaded else ""
        print(f"{m.name:32} {params:>7} {m.quantization:>8} {m.size_bytes / 1e9:6.1f} GB  {role}{mark}")


def cmd_setup(args) -> None:
    from .calibrate import PROFILE_PATH, calibrate

    sampler = _sampler(not args.no_energy)
    print("Calibrating each model (cold load, then a few timed generations)...")

    def show(cost):
        watts = f"{cost.watts:.1f} W" if cost.watts is not None else "n/a"
        print(
            f"  {cost.name:28} load {cost.load_s:5.1f}s   prompt {cost.prompt_tps:7.0f} tok/s   "
            f"generate {cost.gen_tps:5.1f} tok/s   power {watts}"
        )

    profile = calibrate(sampler=sampler, models=args.models, on_model=show)
    path = profile.save(Path(args.out) if args.out else PROFILE_PATH)
    print(f"Saved hardware profile to {path}")


def cmd_ask(args) -> None:
    from .calibrate import HardwareProfile
    from .router import Router, RouterArtifact, explain

    artifact = RouterArtifact.load(args.artifact) if args.artifact else None
    profile = HardwareProfile.load(Path(args.profile)) if args.profile else None
    router = Router(artifact=artifact, profile=profile)
    if args.dry_run:
        print(explain(router.decide(args.prompt, args.quality)))
        return
    decision, result = router.ask(args.prompt, args.quality)
    if args.explain:
        print(explain(decision))
        print()
    print(result.text)
    if not args.explain:
        print(f"\n[{decision.model} · {result.total_s:.1f}s]", file=sys.stderr)
        if decision.warning:
            print(f"[{decision.warning}]", file=sys.stderr)


def cmd_bench(args) -> None:
    from .profiler import run_profile
    from .tasks import load_questions

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    sampler = _sampler(not args.no_energy)

    def show(record, i, n):
        mark = "·" if record.correct is None else "✓" if record.correct else "✗"
        print(f"[{i}/{n}] {record.model:20} {record.task:14} {mark} {record.total_s:5.1f}s", flush=True)

    path = run_profile(args.models, questions, args.out, sampler=sampler, seed=args.seed, on_record=show)
    print(f"Records in {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shiftai",
        description="ShiftAI: send each prompt to the smallest local model that can answer it.",
    )
    parser.add_argument("--version", action="version", version=f"shiftai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list installed Ollama models")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("setup", help="measure each model's speed, load time and power on this machine")
    p.add_argument("--models", nargs="+", help="only calibrate these models")
    p.add_argument("--out", help="where to save the profile (default ~/.shiftai/hardware.json)")
    p.add_argument("--no-energy", action="store_true", help="skip power telemetry")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("ask", help="route a prompt to the right-sized model")
    p.add_argument("prompt")
    p.add_argument("-q", "--quality", type=float, default=95, help="target %% of the largest model's quality (default 95)")
    p.add_argument("--explain", action="store_true", help="show why the model was chosen")
    p.add_argument("--dry-run", action="store_true", help="show the decision without generating")
    p.add_argument("--artifact", help="use a router artifact file instead of the bundled one")
    p.add_argument("--profile", help="use a hardware profile file instead of ~/.shiftai/hardware.json")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("bench", help="profile models on the benchmark question set")
    p.add_argument("--models", nargs="+", required=True, help="models to profile, smallest first")
    p.add_argument("--questions", default="data/questions.jsonl")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-energy", action="store_true", help="skip power telemetry")
    p.set_defaults(func=cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except OllamaUnavailable as exc:
        sys.exit(f"Error: {exc}")
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
