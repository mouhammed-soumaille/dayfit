"""The conformal guarantee on simulated scores where we know the truth."""
import numpy as np

from evaluation.conformal import relative_q, repeated_splits, threshold_tau


def fake_scores(n_queries=300, n_catalog=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 0.1, (n_queries, n_catalog))
    true_idx = rng.integers(0, n_catalog, n_queries)
    scores[np.arange(n_queries), true_idx] += rng.normal(0.35, 0.15, n_queries)
    return scores, true_idx


def test_coverage_matches_target():
    scores, true_idx = fake_scores()
    for alpha in (0.2, 0.1):
        for method in ("threshold", "relative"):
            r = repeated_splits(scores, true_idx, alpha, method, n_trials=200)
            assert abs(r["coverage"] - (1 - alpha)) < 0.03


def test_too_few_calibration_points_returns_everything():
    assert threshold_tau(np.array([1.0, 2.0, 3.0]), 0.1) == -np.inf
    assert relative_q(np.array([1.0, 2.0, 3.0]), 0.1) == np.inf
