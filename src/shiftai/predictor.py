"""Capability predictor: how likely is each model to answer this prompt well?

Prompts are first split by type (multiple choice versus open-ended), then
grouped into clusters of similar prompts by their embeddings. For every
cluster we store each model's measured accuracy, shrunk toward the accuracy on
that prompt type so small clusters do not produce extreme estimates. A new
prompt gets the estimates of its nearest cluster.

Why clusters and not a per-prompt classifier: a logistic regression over the
full embedding ranks answers only moderately well (0.64 to 0.72 AUC). Routed
with the same tuning, it matched the cluster estimates at looser targets but
missed the 99% target, where it also saved far less (see scripts/ablations.py).
Group-level estimates are less noisy, met every target, and are easy to
inspect.

Distance to the nearest cluster doubles as a novelty score: prompts unlike
anything seen in training are routed more conservatively.

Training needs scikit-learn (the `bench` extra). Prediction is plain NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import KINDS, prompt_kind


def _round(a, digits: int = 6) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


@dataclass
class KindClusters:
    """Clusters for one prompt type and each model's accuracy per cluster."""

    centroids: np.ndarray  # (k, dim), unit length
    accuracy: np.ndarray   # (k, n_models)
    medium: float          # similarity below this: somewhat unfamiliar
    low: float             # similarity below this: very unfamiliar

    def nearest(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sims = embeddings @ self.centroids.T
        return sims.argmax(axis=1), sims.max(axis=1)


class CapabilityPredictor:
    """Predicts P(model gives an acceptable answer | prompt) for every model."""

    def __init__(self, models: list[str], kinds: dict[str, KindClusters]):
        self.models = models
        self.kinds = kinds

    def predict(self, embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
        """(n_prompts, n_models) probabilities, columns in `self.models` order."""
        probs = np.zeros((len(texts), len(self.models)))
        for kind, clusters in self.kinds.items():
            rows = [i for i, t in enumerate(texts) if prompt_kind(t) == kind]
            if rows:
                nearest, _ = clusters.nearest(embeddings[rows])
                probs[rows] = clusters.accuracy[nearest]
        return probs

    def novelty(self, embeddings: np.ndarray, texts: list[str]) -> list[str]:
        levels = []
        for vec, text in zip(embeddings, texts):
            clusters = self.kinds.get(prompt_kind(text))
            if clusters is None:
                levels.append("low")
                continue
            _, sim = clusters.nearest(vec[None, :])
            levels.append("low" if sim[0] < clusters.low else "medium" if sim[0] < clusters.medium else "high")
        return levels

    def to_dict(self) -> dict:
        return {
            "models": self.models,
            "kinds": {
                kind: {
                    "centroids": _round(c.centroids, 5),
                    "accuracy": _round(c.accuracy, 4),
                    "medium": c.medium,
                    "low": c.low,
                }
                for kind, c in self.kinds.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityPredictor":
        kinds = {
            kind: KindClusters(np.asarray(c["centroids"]), np.asarray(c["accuracy"]), c["medium"], c["low"])
            for kind, c in data["kinds"].items()
        }
        return cls(data["models"], kinds)


def train_predictor(
    embeddings: np.ndarray,
    texts: list[str],
    correct: np.ndarray,
    models: list[str],
    clusters_per_kind: int = 8,
    prior: float = 10.0,
    seed: int = 0,
) -> CapabilityPredictor:
    """Cluster each prompt type and estimate every model's accuracy per cluster.

    `correct` is (n_prompts, n_models). `prior` is how many pseudo-examples of
    the prompt-type average each cluster is shrunk toward.
    """
    from sklearn.cluster import KMeans

    kinds = np.array([prompt_kind(t) for t in texts])
    fitted = {}
    for kind in KINDS:
        rows = np.where(kinds == kind)[0]
        if len(rows) == 0:
            continue
        k = max(1, min(clusters_per_kind, len(rows) // 20))
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(embeddings[rows])
        centroids = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        base = correct[rows].mean(axis=0)
        accuracy = np.zeros((k, len(models)))
        for c in range(k):
            members = rows[km.labels_ == c]
            accuracy[c] = (correct[members].sum(axis=0) + prior * base) / (len(members) + prior)
        sims = (embeddings[rows] @ centroids.T).max(axis=1)
        fitted[kind] = KindClusters(
            centroids, accuracy,
            medium=float(np.percentile(sims, 5)),
            low=float(np.percentile(sims, 1)),
        )
    return CapabilityPredictor(models, fitted)
