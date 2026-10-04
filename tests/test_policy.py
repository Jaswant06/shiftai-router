"""Routing rule, delta tuning and offline scoring on small hand-made data."""

import numpy as np

from shiftai.evaluate import ProfileMatrix, oracle_choice, score_policy
from shiftai.policy import choose, pick_target, simulate, tune_deltas


def test_choose_takes_cheapest_model_close_enough_to_largest():
    probs = {"small": 0.70, "mid": 0.86, "large": 0.90}
    costs = {"small": 1.0, "mid": 2.0, "large": 5.0}
    assert choose(probs, costs, "large", delta=0.05)[0] == "mid"
    assert choose(probs, costs, "large", delta=0.25)[0] == "small"
    assert choose(probs, costs, "large", delta=0.0)[0] == "large"


def test_unfamiliar_prompts_get_more_compute():
    probs = {"small": 0.80, "large": 0.90}
    costs = {"small": 1.0, "large": 5.0}
    assert choose(probs, costs, "large", delta=0.15, novelty="high")[0] == "small"
    assert choose(probs, costs, "large", delta=0.15, novelty="medium")[0] == "large"
    assert choose(probs, costs, "large", delta=1.0, novelty="low")[0] == "large"


def test_cost_order_follows_costs_not_size():
    # The large model is already loaded, so it is cheaper than cold-loading the mid one.
    probs = {"small": 0.5, "mid": 0.9, "large": 0.9}
    costs = {"small": 1.0, "mid": 6.0, "large": 3.0}
    assert choose(probs, costs, "large", delta=0.05)[0] == "large"


def test_simulate_matches_choose():
    probs = np.array([[0.70, 0.86, 0.90], [0.95, 0.96, 0.97]])
    chosen = simulate(probs, order=[0, 1, 2], largest=2, delta=0.05)
    assert chosen.tolist() == [1, 0]


def test_tuned_delta_meets_target_on_validation():
    rng = np.random.default_rng(0)
    n = 400
    p_large = rng.uniform(0.5, 1.0, n)
    p_small = p_large - rng.uniform(0.0, 0.5, n)
    probs = np.column_stack([p_small, p_large])
    correct = rng.uniform(size=(n, 2)) < probs
    table = tune_deltas(probs, correct, order=[0, 1], largest=1, taus=(0.9, 1.0))
    reference = correct[:, 1].mean()
    for tau, delta in table.items():
        chosen = simulate(probs, [0, 1], 1, delta)
        assert correct[np.arange(n), chosen].mean() >= tau * reference - 1e-12
    assert table[0.9] >= table[1.0]


def test_pick_target_rounds_up_and_caps_at_strictest():
    table = {0.9: 0.2, 0.95: 0.1, 0.99: 0.05}
    assert pick_target(table, 95) == 0.1
    assert pick_target(table, 92) == 0.1
    assert pick_target(table, 100) == 0.05


def test_oracle_and_scoring():
    correct = np.array([[True, True], [False, True], [False, False]])
    latency = np.array([[1.0, 4.0], [1.0, 4.0], [1.0, 4.0]])
    matrix = ProfileMatrix(
        qids=["a", "b", "c"], tasks=["t"] * 3, models=["s", "l"],
        correct=correct, latency=latency, energy=np.full((3, 2), np.nan),
        output_tokens=np.ones((3, 2)),
    )
    oracle = oracle_choice(correct, latency)
    assert oracle.tolist() == [0, 1, 0]
    rows = np.arange(3)
    result = score_policy("oracle", oracle, matrix, rows)
    assert result["relative_quality"] == 1.0
    assert result["mean_latency_s"] == 2.0
    assert result["latency_saving"] == 0.5
    assert "mean_energy_j" not in result
