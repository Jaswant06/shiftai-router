"""Turn a prompt into the feature vector the capability predictor reads.

Only information available before any expensive model runs is used: an
embedding from a small local embedding model and the prompt's length. The
embedding model is tiny (137M parameters), so this costs milliseconds, and
that cost is measured and reported as router overhead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ollama import OllamaClient

EMBED_MODEL = "nomic-embed-text"
# nomic-embed-text expects a task prefix; "classification" suits routing.
EMBED_PREFIX = "classification: "
MAX_CHARS = 4000


def embed(texts: list[str], client: OllamaClient, batch_size: int = 32) -> np.ndarray:
    """L2-normalised embeddings, one row per text."""
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = [EMBED_PREFIX + t[:MAX_CHARS] for t in texts[start:start + batch_size]]
        rows.extend(client.embed(EMBED_MODEL, batch))
    vectors = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def length_feature(texts: list[str]) -> np.ndarray:
    return np.log1p([len(t) for t in texts]).astype(np.float32)[:, None]


def featurize(embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
    """Concatenate the embedding with the log prompt length."""
    return np.hstack([embeddings, length_feature(texts)])


def cached_embeddings(texts: list[str], client: OllamaClient, path: str | Path) -> np.ndarray:
    """Embed once and reuse: profiling sets are embedded many times while iterating."""
    path = Path(path)
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    vectors = embed(texts, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
