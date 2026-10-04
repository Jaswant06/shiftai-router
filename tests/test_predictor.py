"""Cluster predictor: learns which prompt groups each model handles well."""

import numpy as np

from shiftai.predictor import CapabilityPredictor, train_predictor


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # Two topics in embedding space; the small model only handles topic 0.
    topic = rng.integers(0, 2, n)
    emb = np.where(topic[:, None] == 0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) + rng.normal(0, 0.05, (n, 3))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    texts = [f"open question {i}" for i in range(n)]
    small_right = np.where(topic == 0, rng.uniform(size=n) < 0.95, rng.uniform(size=n) < 0.1)
    large_right = rng.uniform(size=n) < 0.95
    return emb, texts, np.column_stack([small_right, large_right]), topic


def test_learns_where_the_small_model_fails():
    emb, texts, correct, topic = _data()
    predictor = train_predictor(emb, texts, correct, ["small", "large"], clusters_per_kind=2)
    probs = predictor.predict(emb, texts)
    assert probs[topic == 0, 0].mean() > 0.8
    assert probs[topic == 1, 0].mean() < 0.3
    assert np.all(probs[:, 1] > 0.8)


def test_round_trip_and_novelty():
    emb, texts, correct, _ = _data()
    predictor = train_predictor(emb, texts, correct, ["small", "large"], clusters_per_kind=2)
    restored = CapabilityPredictor.from_dict(predictor.to_dict())
    assert np.allclose(restored.predict(emb, texts), predictor.predict(emb, texts), atol=1e-3)
    assert restored.novelty(np.array([[0.0, 0.0, 1.0]]), ["new"]) == ["low"]
    assert restored.novelty(emb[:1], texts[:1]) == ["high"]
