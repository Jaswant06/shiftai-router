"""Capability predictor: how likely is each model to answer this prompt well?

One logistic regression per model reads the prompt features and outputs a
probability; an isotonic map fitted on held-out data then calibrates it, so a
predicted 0.8 really means "right about 80% of the time".

A novelty score flags prompts unlike anything seen in training. The router
uses it to spend more compute when it cannot trust its own prediction.

Training needs scikit-learn (the `bench` extra). Prediction is plain NumPy,
so the installed package stays light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _round(a: np.ndarray, digits: int = 6) -> list:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


@dataclass
class ModelHead:
    """Logistic-regression weights plus the isotonic calibration curve."""

    coef: np.ndarray
    intercept: float
    iso_x: np.ndarray
    iso_y: np.ndarray

    def predict(self, z: np.ndarray) -> np.ndarray:
        raw = _sigmoid(z @ self.coef + self.intercept)
        return np.interp(raw, self.iso_x, self.iso_y)


@dataclass
class Novelty:
    """Similarity to the training distribution, with two alert levels."""

    centroids: np.ndarray
    medium: float
    low: float

    def similarity(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings @ self.centroids.T).max(axis=1)

    def level(self, embeddings: np.ndarray) -> list[str]:
        sims = self.similarity(embeddings)
        return ["low" if s < self.low else "medium" if s < self.medium else "high" for s in sims]


class CapabilityPredictor:
    """Predicts P(model gives an acceptable answer | prompt) for every model."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, heads: dict[str, ModelHead], novelty: Novelty):
        self.mean = mean
        self.std = std
        self.heads = heads
        self.novelty = novelty

    @property
    def models(self) -> list[str]:
        return list(self.heads)

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        z = (features - self.mean) / self.std
        return {name: head.predict(z) for name, head in self.heads.items()}

    def to_dict(self) -> dict:
        return {
            "mean": _round(self.mean),
            "std": _round(self.std),
            "heads": {
                name: {
                    "coef": _round(h.coef),
                    "intercept": float(h.intercept),
                    "iso_x": _round(h.iso_x),
                    "iso_y": _round(h.iso_y),
                }
                for name, h in self.heads.items()
            },
            "novelty": {
                "centroids": _round(self.novelty.centroids, 5),
                "medium": self.novelty.medium,
                "low": self.novelty.low,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityPredictor":
        heads = {
            name: ModelHead(
                coef=np.asarray(h["coef"]),
                intercept=h["intercept"],
                iso_x=np.asarray(h["iso_x"]),
                iso_y=np.asarray(h["iso_y"]),
            )
            for name, h in data["heads"].items()
        }
        nov = data["novelty"]
        novelty = Novelty(np.asarray(nov["centroids"]), nov["medium"], nov["low"])
        return cls(np.asarray(data["mean"]), np.asarray(data["std"]), heads, novelty)


def train_predictor(
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    emb_train: np.ndarray,
    emb_val: np.ndarray,
    c: float = 0.5,
    n_clusters: int = 32,
    seed: int = 0,
) -> CapabilityPredictor:
    """Fit one calibrated head per model and the novelty detector."""
    from sklearn.cluster import KMeans
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    z_train = (x_train - mean) / std
    z_val = (x_val - mean) / std

    heads = {}
    for name, labels in y_train.items():
        if labels.min() == labels.max():
            # A model that is always right (or always wrong) on the training set.
            p = float(labels[0])
            heads[name] = ModelHead(np.zeros(z_train.shape[1]), 0.0, np.array([0.0, 1.0]), np.array([p, p]))
            continue
        lr = LogisticRegression(C=c, max_iter=5000, random_state=seed).fit(z_train, labels)
        raw_val = lr.predict_proba(z_val)[:, 1]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_val, y_val[name])
        heads[name] = ModelHead(lr.coef_[0], float(lr.intercept_[0]), iso.X_thresholds_, iso.y_thresholds_)

    k = min(n_clusters, len(emb_train))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(emb_train)
    centroids = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    val_sims = (emb_val @ centroids.T).max(axis=1)
    novelty = Novelty(centroids, medium=float(np.percentile(val_sims, 10)), low=float(np.percentile(val_sims, 2)))
    return CapabilityPredictor(mean, std, heads, novelty)
