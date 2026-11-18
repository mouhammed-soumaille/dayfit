"""Cluster bootstrap for the LookBench paired tests (queries that share an identical crop).

Finding 20: only 700 of the 981 LookBench query crops are distinct; 503 queries fall into groups
that use the SAME image (each asking for a different item). The model gives the same ranking for
the same image, so these queries do not fail independently. Our paired bootstrap resamples
QUERIES as if they were independent, which makes confidence intervals too narrow.

Cluster bootstrap: resample whole CROPS (clusters) with replacement and keep each crop's queries
together. With per-query differences d_i, cluster sums S_c and cluster sizes n_c, one resample
draws 700 clusters and computes sum(S) / sum(n), i.e. the mean difference over the drawn queries.

Sanity check: if every query is its own cluster, the cluster bootstrap must reproduce the
existing paired bootstrap (evaluation/stats.py) exactly: same seed, same numbers.

Metrics (dedup protocol): Recall@1, category P@5 (exact names), group P@5 (Finding 19).

Run from the repo root:
    python -m evaluation.cluster_bootstrap
"""
import hashlib

import numpy as np
import pandas as pd

from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY, p5_per_query
from evaluation.stats import N_BOOT, paired_bootstrap

LB = "data/lookbench/real_streetlook"
COMPARISONS = [("fashion_siglip", "finetuned_random"),
               ("finetuned", "finetuned_random"),
               ("ensemble_mv", "finetuned_random")]


def cluster_bootstrap(a, b, clusters, n_boot=N_BOOT, seed=0):
    """Like stats.paired_bootstrap (difference a - b), but resampling whole clusters.
    clusters: one cluster label per query. Returns (mean difference, CI low, CI high, p-value)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    labels, first = np.unique(clusters, return_index=True)
    order = labels[np.argsort(first)]                 # clusters in order of first appearance
    position = {c: k for k, c in enumerate(order)}
    ids = np.array([position[c] for c in clusters])
    sums = np.bincount(ids, weights=diff)              # S_c
    sizes = np.bincount(ids).astype(float)             # n_c
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(order), size=(n_boot, len(order)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(diff.mean()), float(low), float(high), float(min(p, 1.0))


def main():
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)
    queries["crop"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in queries["image"]]
    print(f"  {len(queries)} queries, {queries['crop'].nunique()} distinct crops")

    # group P@5 per query needs each gallery picture's groups
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    pic_groups = gallery.groupby("picture")["group"].apply(set)

    per_method = {}
    for m in sorted({x for pair in COMPARISONS for x in pair}):
        pq = pd.read_csv(f"results/perquery/{m}_lookbench_real_streetlook_image_dedup.csv")
        ranks = pd.read_csv(f"results/rankings/{m}_lookbench_real_streetlook_image_dedup.csv")
        group = p5_per_query(ranks, queries[["query_id", "category"]], pic_groups,
                             lambda c: GROUP_OF[c.lower()])
        pq["group_p5"] = pq["query_id"].map(group)
        assert (pq["query_id"].values == queries["query_id"].values).all(), "query order differs"
        per_method[m] = pq

    crops = queries["crop"].values
    singletons = queries["query_id"].values                # every query its own cluster
    rows = []
    for a, b in COMPARISONS:
        for metric, col in [("Recall@1", "hit1"), ("CategoryP@5", "p5"), ("GroupP@5", "group_p5")]:
            x, y = per_method[b][col].values, per_method[a][col].values
            naive = paired_bootstrap(x, y)
            check = cluster_bootstrap(x, y, singletons)
            if not np.allclose(naive, check):
                raise SystemExit(f"sanity check failed for {a} -> {b} {metric}: {naive} vs {check}")
            clus = cluster_bootstrap(x, y, crops)
            widen = (clus[2] - clus[1]) / (naive[2] - naive[1])
            rows.append(dict(change=f"{a} -> {b}", metric=metric, difference=naive[0],
                             naive_lo=naive[1], naive_hi=naive[2], naive_p=naive[3],
                             cluster_lo=clus[1], cluster_hi=clus[2], cluster_p=clus[3],
                             ci_widening=widen))
            print(f"  {a} -> {b}  {metric:12s} {naive[0]:+.3f}  naive [{naive[1]:+.3f}, {naive[2]:+.3f}] "
                  f"p={naive[3]:.4f}   cluster [{clus[1]:+.3f}, {clus[2]:+.3f}] p={clus[3]:.4f}  "
                  f"(CI x{widen:.2f})")
    print("  check: with one query per cluster the cluster bootstrap reproduced every naive result")

    pd.DataFrame(rows).to_csv("results/cluster_bootstrap.csv", index=False)
    print("saved results/cluster_bootstrap.csv")


if __name__ == "__main__":
    main()