---
title: ShiftAI
emoji: ⚙️
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Send each prompt to the smallest LLM that can answer it
---

# ShiftAI demo

Interactive demo of [ShiftAI](https://github.com/Jaswant06/shiftai-router), an installable router that sends
each prompt to the smallest local language model that can answer it well.

- **Route a prompt:** type anything and see which model the real router would choose, why, and each model's
  predicted quality and estimated time on an Apple M5. The router runs entirely in your browser: the same
  embedding model (nomic-embed-text, via Transformers.js) and the same trained router as the Python package.
- **Replay real answers:** held-out test prompts with the answers all four models actually gave, how they were
  graded, and the measured time and energy.
- **Results:** quality versus energy on 760 held-out prompts.

The language models themselves do not run here. Installed locally, ShiftAI routes to models running in Ollama.
`space/build_data.py` in the GitHub repo regenerates the demo data from the profiling runs.
