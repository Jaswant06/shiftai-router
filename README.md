# ShiftAI: Adaptive LLM Routing for Resource-Efficient Local AI

ShiftAI is an installable router for local language models. For every prompt it
predicts how well each model on your computer would answer, then sends the
prompt to the **cheapest model that keeps the quality you asked for**, instead
of running everything through the largest model. Easy prompts go to a small,
fast model; hard prompts go to a big one.

> Ask a question -> ShiftAI predicts each model's quality and cost on *your*
> machine -> the smallest model that is good enough answers.

```text
$ shiftai ask "A bakery sells 24 muffins per tray. On Monday it baked 7 trays
  and sold all but 13 muffins. How many muffins were sold?" --quality 99 --explain
Quality target: 99% of the largest model (latency optimised)
model                    pred. quality  relative  est. time  est. energy  loaded
qwen3.5:0.8b                      0.53       57%      3.82s          n/a
qwen3.5:2b                        0.72       77%      3.39s          n/a  yes
qwen3.5:4b                        0.92       99%      7.84s          n/a
qwen3.5:9b                        0.93      100%      8.71s          n/a  yes
Selected: qwen3.5:4b  (cheapest model predicted close enough to the largest for your quality target)
Prompt familiarity: high   Router overhead: 34 ms
```

**Headline result (v0.1, 600 held-out questions, Apple M5):** asked to keep 99%
of the 9B model's quality, ShiftAI kept **99.6%** while answering **29% faster**
with **26% less energy**. The best random split between fixed models saved only
12% and 11% at the same quality. [Full results below.](#results)

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
prompt -> type + embedding -> capability predictor -> P(good answer) per model
                                                               |
local hardware profile + loaded models + battery ----> estimated cost per model
                                                               |
                         cheapest model meeting the quality target -> answer
```

- **Capability predictor:** prompts are split by type (multiple choice or
  open-ended) and grouped into clusters of similar prompts using a
  `nomic-embed-text` embedding. Each cluster stores every model's measured
  accuracy, shrunk toward the type average so small clusters stay sensible. A
  new prompt gets the estimates of its nearest cluster
  (`src/shiftai/predictor.py`). A per-prompt logistic regression was tried and
  rejected; see [what did not work](#what-did-not-work).
- **Relative quality target:** `--quality 95` means "keep at least 95% of what
  the largest model would score". The router takes the cheapest model predicted
  to reach at least (1 - delta) of the largest model's quality. For each target,
  the number of clusters and delta are tuned on out-of-fold predictions so the
  routed accuracy over the whole workload meets the target
  (`src/shiftai/policy.py`). `--quality 100` simply uses the largest model.
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
(`src/shiftai/grading.py`). 20% of the questions (600) are held out as a test
set that nothing touches until the final evaluation. On the other 80%, 5-fold
out-of-fold predictions are used to choose the number of clusters and tune
delta for each target. Because every model answered every question, any routing
policy can be scored exactly on the test set and compared with:

- each single model on its own (including always-smallest and always-largest)
- a **random split**: send each prompt at random to one of two models, in the
  proportion that meets the target on the development data. This is the best
  you can do without looking at the prompt, so it is the baseline to beat.
- a **per-type router**: the same router with one cluster per prompt type
- an **oracle** that always picks the cheapest model that was actually right,
  which shows the most any router could save

Energy is measured with `powermetrics` on Apple Silicon, `nvidia-smi` on NVIDIA
GPUs and RAPL counters on Linux. Where no telemetry exists it is reported as
not measured, never guessed.

## Results

Qwen 3.5 ladder (0.8B, 2B, 4B, 9B) on an Apple M5 with 24 GB of RAM, energy
measured with `powermetrics`. Savings are relative to always using the 9B model,
on the 600 held-out test questions, with the router's own overhead (about 10 ms
per prompt) included.

![Accuracy versus energy per prompt](results/pareto_energy.png)

| Quality target | ShiftAI quality kept | ShiftAI faster | ShiftAI less energy | Random split faster | Random split less energy |
|---|---:|---:|---:|---:|---:|
| 80% | 82.3% | 59.9% | 68.1% | 62.3% | 69.1% |
| 85% | 89.3% | 47.7% | 51.6% | 52.9% | 58.4% |
| 90% | 91.7% | 40.5% | 41.0% | 44.4% | 47.2% |
| 95% | 96.4% | 32.9% | 30.6% | 35.6% | 34.4% |
| **99%** | **99.6%** | **29.3%** | **25.6%** | 11.6% | 10.7% |

What the results show:

- **ShiftAI met every quality target on unseen questions**, from 80% to 99%.
- **Its clear win is at the strict end.** At 99% it keeps 99.6% of the 9B's
  quality while saving 2.5 times more time and energy than the best random
  split. The reason is visible in the data: the 4B model is nearly as good as
  the 9B on math (92% versus 93%) but much faster on long answers, so math
  prompts go to the 4B and the rest stay on the 9B.
- **At looser targets it is not yet better than a well-tuned random split.**
  Between 80% and 95% ShiftAI meets the target but overshoots on quality (89%
  when asked for 85%), so it saves slightly less. Being honest about this is the
  point of including the baseline.
- **There is a lot of room left.** The oracle keeps 106% of the 9B's quality
  (several models together get more questions right than the 9B alone) at 63%
  less latency. The bottleneck is how much can be predicted from the prompt
  before running any model.
- **The 0.8B model was never chosen.** It is about as expensive as the 2B on
  this machine but much less accurate.

Every number above can be regenerated with `scripts/train_router.py`; the full
metrics, including p95 latency and 95% confidence intervals (about plus or
minus 3 points on 600 questions), are in [`results/`](results/).

## What did not work

Both are reproducible with `scripts/ablations.py`.

- **Per-prompt logistic regression** over the full embedding ranked right and
  wrong answers only moderately well (0.64 to 0.72 AUC). Tuned and routed the
  same way, it matched the cluster router between 80% and 95% but **missed the
  99% target** (98.5%) and saved 18% instead of 29% there.
- **Agreement cascades** (run two cheaper models, accept the answer if they
  agree, otherwise escalate) kept quality high but were **20% to 50% slower and
  used more energy** than always running the 9B, because running models one
  after another adds up. Deciding before running anything is the better design
  on a single machine.

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
│   ├── features.py      # prompt type and embedding
│   ├── predictor.py     # cluster-based capability predictor
│   ├── policy.py        # routing rule and delta tuning per quality target
│   ├── router.py        # runtime router and decision explanation
│   ├── evaluate.py      # offline scoring of policies, oracle
│   ├── artifacts/       # the trained router shipped with the package
│   └── cli.py           # shiftai command
├── scripts/
│   ├── build_question_set.py
│   ├── train_router.py  # train, evaluate, write results/
│   └── ablations.py     # the designs that were rejected, with numbers
├── data/                # sampled question sets (JSONL)
├── results/             # metrics, summary table, Pareto charts
├── tests/               # unit tests (no Ollama needed)
└── pyproject.toml
```

## Limitations

- **Trained on benchmark questions, so savings today come from math and
  multiple-choice prompts.** Open-ended prompts such as emails, code or a
  factual question without options look unlike the training data, so ShiftAI
  sends them to the largest model rather than guess. An open-ended dataset,
  judged against the largest model's answer, is the main goal of the next
  version.
- **One machine, one model family.** Results are from an Apple M5 and the Qwen
  3.5 ladder. Other hardware gets its own cost profile from `shiftai setup`,
  but the quality estimates were measured on this ladder only.
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
