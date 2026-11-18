"""Paired cluster bootstrap on the LookBench TEST half for the methods chosen on dev
(pre-registered in report/results_log.md). Difference = method - finetuned_random.
Clusters = the split clusters (queries sharing a crop, street photo or item), resampled whole.
Holm correction over the 4 methods, separately for R@1 and for group P@5.

Sanity check: "base" (a = w = beta = 0) must equal the saved finetuned_random test values.

Run from the repo root (after fusion_lookbench.py --side test --save NAME for every method):
    python -m evaluation.fusion_test_stats
"""
import numpy as np
import pandas as pd

from evaluation.cluster_bootstrap import cluster_bootstrap

LB = "data/lookbench/real_streetlook"
METHODS = ["colour", "dinov2", "groupbonus", "combined"]


def holm(pvalues):
    """Holm step-down: adjusted p-values in the original order."""
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def main():
    split = pd.read_csv(f"{LB}/split.csv").set_index("query_id")
    base = pd.read_csv("results/perquery/base_lookbench_test.csv").set_index("query_id")
    saved = pd.read_csv("results/perquery/finetuned_random_lookbench_real_streetlook_image_dedup.csv")
    saved = saved.set_index("query_id").loc[base.index]
    assert np.allclose(base["hit1"], saved["hit1"]), "base does not reproduce the saved test R@1"
    print(f"check OK: base test R@1 {base['hit1'].mean():.3f} = saved")
    clusters = split.loc[base.index, "cluster"].to_numpy()

    rows = []
    for metric in ["hit1", "gp5"]:
        found = []
        for m in METHODS:
            other = pd.read_csv(f"results/perquery/{m}_lookbench_test.csv").set_index("query_id").loc[base.index]
            diff, low, high, p = cluster_bootstrap(other[metric], base[metric], clusters)
            found.append(dict(metric=metric, method=m, base=base[metric].mean(), value=other[metric].mean(),
                              diff=diff, ci_low=low, ci_high=high, p=p))
        adjusted = holm(np.array([f["p"] for f in found]))
        for f, p_holm in zip(found, adjusted):
            rows.append({**f, "p_holm": p_holm})
    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    table.to_csv("results/fusion_test.csv", index=False)


if __name__ == "__main__":
    main()
