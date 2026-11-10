"""Confidence intervals and significance tests by bootstrapping.

The idea: we only have ~300 queries, so a score like 79% is a noisy estimate.
To see how noisy, we pretend our queries are the whole population:
  1. draw 300 queries WITH replacement from our 300 (some appear twice, some not at all)
  2. recompute the score on that resample
  3. repeat 2000 times
The middle 95% of those 2000 scores (2.5th to 97.5th percentile) is the
95% confidence interval.

For comparing two methods we use a PAIRED bootstrap: both methods were run on
the same queries, so we resample queries and look at the DIFFERENCE between
the two methods on the same resampled queries. This is much more sensitive
than comparing two separate intervals, because query difficulty cancels out.
"""
import numpy as np

N_BOOT = 2000


def bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    """values: one number per query (e.g. 1/0 hits). Returns (low, high) 95% CI of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=0):
    """a, b: per-query scores of two methods on the SAME queries, same order.
    Returns (mean difference a-b, CI low, CI high, two-sided p-value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    # p-value: how often the resampled difference lands on the other side of 0
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(diff.mean()), float(low), float(high), float(min(p, 1.0))
