"""Sanity checks for the bootstrap helpers, on cases with an obvious answer."""
from evaluation.stats import bootstrap_ci, paired_bootstrap


def test_ci_contains_the_mean():
    low, high = bootstrap_ci([1] * 80 + [0] * 20)  # true mean 0.8
    assert low < 0.8 < high
    assert 0.65 < low and high < 0.95


def test_ci_of_constant_is_a_point():
    assert bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0)


def test_paired_identical_methods_not_significant():
    diff, low, high, p = paired_bootstrap([1, 0] * 50, [1, 0] * 50)
    assert diff == 0 and p == 1.0


def test_paired_clearly_better_method_is_significant():
    diff, low, high, p = paired_bootstrap([1] * 50, [0] * 50)
    assert diff == 1 and p < 0.05
