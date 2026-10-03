# ShiftAI: Adaptive LLM Routing for Resource-Efficient Local AI

ShiftAI is an installable router for local language models. For every prompt it
predicts how well each model on your computer would answer, then sends the
prompt to the **cheapest model that keeps the quality you asked for**, instead
of running everything through the largest model. Easy prompts go to a small,
fast model; hard prompts go to a big one.

> Ask a question -> ShiftAI predicts each model's quality and cost on *your*
> machine -> the smallest model that is good enough answers.

```text
$ shiftai ask "What is the capital of France?" --explain
Quality target: 95% of the largest model (latency optimised)
model                    pred. quality  relative  est. time  est. energy  loaded
qwen3.5:0.8b                      0.93      97%      0.41s          n/a
qwen3.5:2b                        0.95      99%      0.62s          n/a
qwen3.5:4b                        0.96     100%      1.10s          n/a
qwen3.5:9b                        0.96     100%      2.30s          n/a  yes
Selected: qwen3.5:0.8b  (cheapest model predicted to keep at least 95% of the largest model's quality)
```
*(Illustrative output; real numbers come from your own `shiftai setup`.)*

**Status:** v0.1 in development. Results below will be filled in from the full
profiling run.

## Why route at all

Local models are a ladder of trade-offs: a 0.8B model answers in a fraction of a
second, a 9B model is much smarter but slower and uses far more energy. Most
apps pick one model and send it everything, so either easy prompts waste time
and battery on a big model, or hard prompts get weak answers from a small one.
ShiftAI makes that choice per prompt.

## How it works

ShiftAI splits the decision into two questions, because they depend on
different things:

1. **Which models can answer this well?** This depends only on the prompt, so
   it is learned once, offline, and shipped with the package.
2. **What does each model cost on this machine, right now?** This depends on
   the hardware and its current state, so it is measured locally.

```text
prompt -> embedding + length -> capability predictor -> P(good answer) per model
                                                               |
local hardware profile + loaded models + battery ----> estimated cost per model
                                                               |
                         cheapest model meeting the quality target -> answer
```

- **Capability predictor:** one logistic regression per model on a
  `nomic-embed-text` embedding of the prompt, calibrated with isotonic
  regression on held-out data so a predicted 0.8 means right about 80% of the
  time (`src/shiftai/predictor.py`).
- **Relative quality target:** `--quality 95` means "keep at least 95% of what
  the largest model would score". The router takes the cheapest model predicted
  to reach at least (1 - delta) of the largest model's quality, and delta is
  tuned on validation data so the routed accuracy over the whole workload meets
  the target (`src/shiftai/policy.py`).
- **Cost model:** `shiftai setup` measures each model's cold load time, prompt
  speed, generation speed, memory and power on your machine
  (`src/shiftai/calibrate.py`). At decision time ShiftAI checks which models are
  already loaded (loading a model can cost seconds) and whether you are on
  battery, in which case it ranks models by energy instead of time.
- **When unsure, spend more:** prompts unlike anything in the training data are
  routed more conservatively, and very unfamiliar ones go to the largest model.
  If even the largest model is predicted to struggle, ShiftAI still answers and
  shows a warning instead of refusing.
- **Honest overhead:** the router's own cost (embedding plus prediction) is
  measured and included in every reported result.

## Training data and evaluation

The router is trained on 3,000 questions with answers a script can check,
sampled once with a fixed seed (`scripts/build_question_set.py`):

| Source | Questions | What it tests |
|---|---:|---|
| GSM8K | 1,000 | multi-step math word problems |
| MMLU | 1,000 | multiple choice across 57 subjects |
| ARC-Easy | 500 | grade-school science |
| ARC-Challenge | 500 | harder grade-school science |

Every model in the ladder answers every question with temperature 0, a fixed
seed and thinking turned off, and each answer is graded deterministically
(`src/shiftai/grading.py`). Questions are split 60/20/20 into train,
validation and test. Because every model answered every question, any routing
policy can be scored exactly on the test set and compared with:

- each single model on its own (including always-smallest and always-largest)
- an **oracle** that always picks the cheapest model that was actually right,
  which shows the most any router could save

Energy is measured with `powermetrics` on Apple Silicon, `nvidia-smi` on NVIDIA
GPUs and RAPL counters on Linux. Where no telemetry exists it is reported as
not measured, never guessed.

## Results

*Coming after the full profiling run: accuracy versus latency and energy
(Pareto chart), savings at 80/90/95/99/100% quality targets, and the gap to the
oracle.*

## Tech stack

Python, Ollama, NumPy, scikit-learn (training only), nomic-embed-text, psutil,
httpx, Matplotlib, pytest.

## Run it locally

You need [Ollama](https://ollama.com) running and at least two chat models of
different sizes. The bundled router is trained on the Qwen 3.5 ladder:

```bash
ollama pull qwen3.5:0.8b
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

Install ShiftAI and measure your machine:

```bash
pip install -e .
shiftai models     # what is installed and loaded
shiftai setup      # measures each model on this machine (a few minutes)
```

On a Mac, run `sudo -v` before `shiftai setup` to include power measurements.

Ask something:

```bash
shiftai ask "Rewrite this sentence to sound more formal: gonna be late, sorry"
shiftai ask "A train leaves at 3pm going 80 km/h..." --quality 99 --explain
shiftai ask "What is 12 x 12?" --dry-run   # show the decision only
```

## Reproduce the research

```bash
pip install -e ".[bench,dev]"
python scripts/build_question_set.py                       # 3,000 questions
shiftai bench --models qwen3.5:0.8b qwen3.5:2b qwen3.5:4b qwen3.5:9b --out runs/qwen35
python scripts/train_router.py --run runs/qwen35           # artifact, metrics, charts
pytest
```

Profiling is resumable: if it stops, run the same command again and it
continues where it left off.

## Project structure

```text
shiftai-router/
├── src/shiftai/
│   ├── ollama.py        # Ollama API client with server-side timings
│   ├── discovery.py     # find installed models and what is loaded
│   ├── tasks.py         # question format and prompts
│   ├── grading.py       # deterministic answer extraction and grading
│   ├── energy.py        # powermetrics / nvidia-smi / RAPL power sampling
│   ├── system.py        # battery, CPU and memory state
│   ├── profiler.py      # run every model on every question (resumable)
│   ├── calibrate.py     # per-machine cost profile (shiftai setup)
│   ├── features.py      # prompt embedding and length features
│   ├── predictor.py     # calibrated per-model capability predictor
│   ├── policy.py        # routing rule and delta tuning per quality target
│   ├── router.py        # runtime router and decision explanation
│   ├── evaluate.py      # offline scoring of policies, oracle
│   └── cli.py           # shiftai command
├── scripts/
│   ├── build_question_set.py
│   └── train_router.py
├── data/                # sampled question sets (JSONL)
├── tests/               # unit tests (no Ollama needed)
└── pyproject.toml
```

## Limitations

- **Trained on benchmark questions.** Real prompts (emails, code, summaries) do
  not have exact answers. Unfamiliar prompts are routed conservatively, and an
  open-ended evaluation set judged against the largest model is planned for the
  next version.
- **Bundled router knows one model family.** Models it was not trained on are
  not routed yet; support via model metadata plus a short calibration quiz is
  planned.
- **Output length is estimated** from typical lengths per model, so time and
  energy estimates for very long answers are rough.
- **Energy telemetry varies by platform** and is unavailable on most Windows
  machines.

## Roadmap

- v0.2: OpenAI-compatible local server with streaming, so existing apps can use
  ShiftAI by changing one URL; live dashboard; open-ended evaluation set.
- v0.3: unknown-model support, smarter model loading and unloading,
  cross-hardware transfer study, technical write-up.

## License

MIT, see [LICENSE](LICENSE).

---

Keywords: llm router, local llm, ollama, model routing, inference optimization,
energy efficient ai, on-device ai, adaptive inference, small language models,
Python.
