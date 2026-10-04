"""Turn a prompt into what the capability predictor reads.

Only information available before any expensive model runs is used: the
prompt's type (multiple choice or open-ended) and an embedding from a small
local embedding model. The embedding model is tiny (137M parameters), so this
costs milliseconds, and that cost is measured and reported as router overhead.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .ollama import OllamaClient

EMBED_MODEL = "nomic-embed-text"
# nomic-embed-text expects a task prefix; "classification" suits routing.
EMBED_PREFIX = "classification: "
MAX_CHARS = 4000

KINDS = ("choice", "open")
# Two or more lines that start like "A. ..." or "(B) ..." mean the prompt lists options.
_OPTION_LINE = re.compile(r"^\s*\(?[A-H][.)]\s+\S", re.MULTILINE)


def prompt_kind(text: str) -> str:
    """'choice' when the prompt lists lettered options, otherwise 'open'."""
    return "choice" if len(_OPTION_LINE.findall(text)) >= 2 else "open"


def embed(texts: list[str], client: OllamaClient, batch_size: int = 32) -> np.ndarray:
    """L2-normalised embeddings, one row per text."""
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = [EMBED_PREFIX + t[:MAX_CHARS] for t in texts[start:start + batch_size]]
        rows.extend(client.embed(EMBED_MODEL, batch))
    vectors = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def cached_embeddings(texts: list[str], client: OllamaClient, folder: str | Path) -> np.ndarray:
    """Embed once and reuse. The cache file is keyed by a hash of the exact
    texts, so changing the inputs can never silently reuse stale vectors."""
    digest = hashlib.sha1("\x00".join([EMBED_MODEL, EMBED_PREFIX, *texts]).encode()).hexdigest()[:12]
    path = Path(folder) / f"embeddings-{digest}.npy"
    if path.exists():
        return np.load(path)
    vectors = embed(texts, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
