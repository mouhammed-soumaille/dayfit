"""Freeze a dev/test split of the LookBench real_streetlook queries (done ONCE, Nov 18-19, before
any new method is tried).

Why: LookBench was already used to choose the recommended model. From now on every new method is
tuned on the DEV half only, and the TEST half is used once per final method. Own photos (D9) stay
the last, untouched check.

Queries that are not independent must land on the same side, otherwise dev leaks into test:
  - queries that use the same street photo (raw_image file content),
  - queries that use the same crop (image file content),
  - queries that ask for the same gallery item (label).
We join these with union-find into clusters, shuffle the clusters with a fixed seed and give
clusters to "dev" until it holds about half of the queries; the rest is "test".

Sanity checks: every query gets exactly one side; no street photo, crop or label is on both sides;
the per-category split is printed so we can see it is balanced.

Run from the repo root:
    python -m evaluation.lookbench_split
Output: data/lookbench/real_streetlook/split.csv (query_id, side, cluster)
"""
import hashlib

import numpy as np
import pandas as pd

LB = "data/lookbench/real_streetlook"
SEED = 490


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def main():
    q = pd.read_csv(f"{LB}/queries.csv")
    q["crop"] = [file_hash(p) for p in q["image"]]
    q["photo"] = [file_hash(p) for p in q["raw_image"]]

    # union-find over queries: join queries that share a crop, a street photo or a target item
    parent = list(range(len(q)))
    for column in ["crop", "photo", "label"]:
        for _, rows in q.groupby(column).groups.items():
            rows = list(rows)
            for r in rows[1:]:
                parent[find(parent, r)] = find(parent, rows[0])
    q["cluster"] = [find(parent, r) for r in range(len(q))]

    # shuffle clusters with a fixed seed; fill dev up to half of the queries
    clusters = q["cluster"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    size = q["cluster"].value_counts()
    dev, n_dev = set(), 0
    for c in clusters:
        if n_dev + size[c] <= len(q) / 2 + 10:
            dev.add(c)
            n_dev += size[c]
    q["side"] = np.where(q["cluster"].isin(dev), "dev", "test")

    # sanity checks
    for column in ["crop", "photo", "label"]:
        sides = q.groupby(column)["side"].nunique()
        assert (sides == 1).all(), f"a {column} is on both sides"
    print(f"{q['cluster'].nunique()} clusters, largest {size.max()} queries")
    print(q["side"].value_counts().to_dict())
    print(pd.crosstab(q["category"], q["side"]))

    q[["query_id", "side", "cluster"]].to_csv(f"{LB}/split.csv", index=False)
    print(f"saved {LB}/split.csv")


if __name__ == "__main__":
    main()
