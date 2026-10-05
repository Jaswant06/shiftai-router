"""Build the data for the browser demo (run from the repo root, with Ollama running).

    python space/build_data.py

Writes space/data.json: the trained router (cluster centroids, quality tables,
tuned targets), the measured Apple M5 cost profile, and held-out test prompts
with every model's real answer and the real router's pick at each target. Also
writes the monochrome results charts, and space/.expected.json, the real
router's decisions on all test prompts, used to check the JavaScript port.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from shiftai.calibrate import PROFILE_PATH
from shiftai.evaluate import load_matrix, merge_matrices, split_indices
from shiftai.tasks import load_questions

RUNS = [Path("runs/qwen35"), Path("runs/qwen35-open")]
FAMILY = {"math": "Math", "choice": "Multiple choice", "code": "Coding", "open": "Open-ended"}
PER_FAMILY = {"Math": 12, "Multiple choice": 12, "Coding": 10, "Open-ended": 24}
MAX_CHARS = 1800


def main() -> None:
    from markdown_it import MarkdownIt

    from shiftai.calibrate import HardwareProfile
    from shiftai.ollama import OllamaClient
    from shiftai.router import Router, RouterArtifact
    from shiftai.system import ResourceState

    md = MarkdownIt("commonmark", {"html": False}).enable("table")  # model output is always escaped
    questions = {q.id: q for f in ("data/questions.jsonl", "data/open.jsonl") for q in load_questions(f)}
    matrix = merge_matrices([load_matrix(r, questions=questions) for r in RUNS])
    records = {}
    for run in RUNS:
        for line in (run / "records.jsonl").read_text().splitlines():
            r = json.loads(line)
            records[(r["model"], r["qid"])] = r

    # The measured Apple M5 profile, restricted to the ladder.
    profile = json.loads(PROFILE_PATH.read_text())
    profile["models"] = {m: v for m, v in profile["models"].items() if m in matrix.models}
    profile_path = Path("space/.m5_profile.json")
    profile_path.write_text(json.dumps(profile))

    # The real router, with Ollama embeddings, nothing loaded in memory and a plugged-in machine,
    # exactly as the browser version assumes.
    class NothingLoaded(OllamaClient):
        def ps(self):
            return []

    router = Router(RouterArtifact.load(), HardwareProfile.load(profile_path), NothingLoaded())
    state = ResourceState(on_battery=False, battery_percent=None, cpu_percent=5.0,
                          available_memory_gb=16.0, total_memory_gb=24.0)
    targets = list(range(80, 100))

    _, _, test = split_indices(len(matrix.qids), seed=0)  # the same held-out split as training
    rng = random.Random(7)
    by_family: dict[str, list[int]] = {}
    for i in test:
        by_family.setdefault(FAMILY[questions[matrix.qids[i]].kind], []).append(int(i))

    items = []
    for family, n in PER_FAMILY.items():
        for i in sorted(rng.sample(by_family[family], min(n, len(by_family[family])))):
            q = questions[matrix.qids[i]]
            prompt = q.feature_text()
            answers = []
            for j, model in enumerate(matrix.models):
                r = records[(model, q.id)]
                reply = r["reply"] if len(r["reply"]) <= MAX_CHARS else r["reply"][:MAX_CHARS] + " [...]"
                answers.append({
                    "model": model,
                    "html": md.render(reply),
                    "quality": round(float(matrix.correct[i, j]), 2),
                    "seconds": round(float(matrix.latency[i, j]), 2),
                    "joules": None if np.isnan(matrix.energy[i, j]) else round(float(matrix.energy[i, j]), 1),
                })
            items.append({
                "id": q.id,
                "family": family,
                "prompt": prompt,
                "reference": q.answer if q.kind in ("math", "choice") else (q.answer[:600] if q.judged else ""),
                "answers": answers,
                "picks": {t: router.decide(prompt, t, state=state).model for t in targets},
            })

    # Expected decisions for checking the browser port: every test prompt at five targets.
    check_prompts = [questions[matrix.qids[i]].feature_text() for i in test]
    expected = []
    for prompt in check_prompts:
        row = {"prompt": prompt}
        for t in (80, 85, 90, 95, 99):
            d = router.decide(prompt, t, state=state)
            row[str(t)] = d.model
        expected.append(row)
    Path("space/.expected.json").write_text(json.dumps(expected))

    artifact = json.loads((Path("src/shiftai/artifacts/router.json")).read_text())
    data = {
        "ladder": artifact["ladder"],
        "predictors": artifact["predictors"],
        "targets": artifact["targets"],
        "expected_tokens": artifact["expected_tokens"],
        "profile": {m: {k: v[k] for k in ("load_s", "prompt_tps", "gen_tps")} for m, v in profile["models"].items()},
        "replay": items,
    }
    Path("space/data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote space/data.json with {len(items)} replay prompts; {len(expected)} expected decisions for checking")


def draw_charts() -> None:
    """Monochrome quality-versus-energy charts for the demo, light and dark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads(Path("results/metrics.json").read_text())["results"]
    pick = lambda prefix: sorted((r for r in metrics if r["policy"].startswith(prefix)), key=lambda r: r["mean_energy_j"])
    singles, routers, mixes = pick("only:"), pick("shiftai@"), pick("random-mix@")
    oracle = [r for r in metrics if r["policy"] == "oracle"][0]

    for name, bg, ink, muted, faint in (("light", "#f6f5f1", "#1c1b19", "#6b6862", "#d9d5cc"),
                                         ("dark", "#121211", "#ecebe7", "#9c9a94", "#33322f")):
        plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 9})
        fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.plot([r["mean_energy_j"] for r in mixes], [r["accuracy"] for r in mixes], color=muted, lw=1,
                ls=(0, (3, 3)), label="random split of models")
        ax.plot([r["mean_energy_j"] for r in routers], [r["accuracy"] for r in routers], color=ink, lw=1.6,
                marker="o", ms=5, mfc=bg, mew=1.4, label="shiftai")
        for r in routers:
            ax.annotate(r["policy"].split("@")[1] + "%", (r["mean_energy_j"], r["accuracy"]), color=ink,
                        xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.scatter([r["mean_energy_j"] for r in singles], [r["accuracy"] for r in singles], s=22, color=muted,
                   zorder=3, label="single model")
        for r in singles:
            ax.annotate(r["policy"].split(":")[-1].upper(), (r["mean_energy_j"], r["accuracy"]), color=muted,
                        xytext=(6, -12), textcoords="offset points", fontsize=8)
        ax.scatter([oracle["mean_energy_j"]], [oracle["accuracy"]], s=60, marker="D", facecolor=bg,
                   edgecolor=ink, lw=1.2, zorder=3, label="oracle (upper bound)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(faint)
        ax.tick_params(colors=muted, length=0, pad=6)
        ax.grid(color=faint, lw=0.6, alpha=0.7)
        ax.set_xlabel("energy per prompt (joules)", color=muted, labelpad=8)
        ax.set_ylabel("quality on held-out prompts", color=muted, labelpad=8)
        legend = ax.legend(frameon=False, loc="lower right", fontsize=8)
        for text in legend.get_texts():
            text.set_color(muted)
        fig.tight_layout()
        fig.savefig(f"space/results_{name}.png", facecolor=bg)
        plt.close(fig)
    print("Wrote space/results_light.png and space/results_dark.png")


if __name__ == "__main__":
    main()
    draw_charts()
