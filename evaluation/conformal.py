"""Conformal retrieval: return a SET of results that contains the right item
with a guaranteed probability, instead of always returning exactly 5.

The idea (split conformal prediction)
-------------------------------------
For every query we have a similarity score with every catalog item. Split the
labelled phone queries at random into two halves:
  * calibration half: used to choose a threshold
  * test half: used to check the guarantee

Method "threshold" (same cut-off for every query):
  On the calibration half, look at s_i = similarity between query i and its
  TRUE item. Choose tau = the k-th smallest s_i, with k = floor((n+1) * alpha).
  For a new query, return every catalog item with similarity >= tau.

Method "relative" (cut-off moves with each query's best match):
  Use r_i = (best similarity of query i) - (similarity to its true item) >= 0,
  i.e. how far below the top result the true item sits. Choose q = the k-th
  smallest r_i with k = ceil((n+1) * (1 - alpha)). For a new query, return
  every item whose similarity is within q of that query's best similarity.
  Easy queries (clear winner) get small sets; ambiguous ones get bigger sets.

The guarantee (for either method)
---------------------------------
If calibration and test queries are exchangeable (e.g. drawn the same way),
then for a new query
        P( true item is in the returned set ) >= 1 - alpha.
Proof sketch: the new query's score is equally likely to fall in any of the
n+1 positions when sorted together with the n calibration scores, so it lands
beyond the k-th one with probability at most alpha. No assumption about the
model or the data distribution is needed - only exchangeability.

What we measure: coverage (should be >= 1 - alpha), average set size (smaller
is more useful), and how often the set has exactly one item (a "confident"
answer) and how accurate those single answers are. We repeat the random split
many times and average. We also break exchangeability on purpose (calibrate on
mild photos, test on strong ones) to show what happens when the assumption fails.

Run from the repo root:
    python -m evaluation.conformal --method ensemble_mv
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
from PIL import Image

ALPHAS = [0.2, 0.1, 0.05]


# ---------------------------------------------------------------- the maths (pure numpy)

def threshold_tau(true_scores, alpha):
    """Similarity cut-off from calibration true-item scores. -inf means 'return everything'."""
    n = len(true_scores)
    k = math.floor((n + 1) * alpha)
    if k < 1:
        return -np.inf
    return np.sort(true_scores)[k - 1]


def relative_q(gaps, alpha):
    """Allowed gap below the top score, from calibration gaps. inf means 'return everything'."""
    n = len(gaps)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.inf
    return np.sort(gaps)[k - 1]


def evaluate_sets(scores, true_idx, cal, test, alpha, method):
    """scores: (n_queries, n_catalog) similarities. true_idx: catalog index of each query's item.
    cal, test: arrays of query indices. Returns coverage, set sizes, singleton stats on test."""
    rows = np.arange(len(scores))
    true_scores = scores[rows, true_idx]
    top = scores.max(axis=1)

    if method == "threshold":
        tau = threshold_tau(true_scores[cal], alpha)
        cutoff = np.full(len(test), tau)
    else:
        q = relative_q(top[cal] - true_scores[cal], alpha)
        cutoff = top[test] - q

    in_set = scores[test] >= cutoff[:, None]          # (n_test, n_catalog) booleans
    covered = in_set[np.arange(len(test)), true_idx[test]]
    sizes = in_set.sum(axis=1)
    single = sizes == 1
    return dict(coverage=covered.mean(), mean_size=sizes.mean(), median_size=np.median(sizes),
                singleton_rate=single.mean(),
                singleton_accuracy=covered[single].mean() if single.any() else np.nan)


def repeated_splits(scores, true_idx, alpha, method, n_trials=200, seed=0):
    """Average evaluate_sets over many random 50/50 calibration/test splits."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    results = []
    for _ in range(n_trials):
        order = rng.permutation(n)
        results.append(evaluate_sets(scores, true_idx, order[:n // 2], order[n // 2:], alpha, method))
    return {k: float(np.nanmean([r[k] for r in results])) for k in results[0]}


def shifted(scores_cal, true_cal, scores_test, true_test, alpha, method):
    """Calibrate on one kind of query, test on another (breaks exchangeability on purpose)."""
    scores = np.vstack([scores_cal, scores_test])
    true_idx = np.concatenate([true_cal, true_test])
    cal = np.arange(len(scores_cal))
    test = np.arange(len(scores_cal), len(scores))
    return evaluate_sets(scores, true_idx, cal, test, alpha, method)


# ---------------------------------------------------------------- running a real model

def similarity_matrix(retriever, catalog, cond_queries):
    """Embed every query photo and compare with every catalog item."""
    position = {item_id: i for i, item_id in enumerate(catalog["item_id"])}
    vectors = []
    for n, q in enumerate(cond_queries.itertuples(), start=1):
        vectors.append(retriever.embed_query(Image.open(q.image_path).convert("RGB")))
        print(f"  {n}/{len(cond_queries)}", end="\r")
    scores = np.stack(vectors) @ retriever.features.T
    true_idx = np.array([position[i] for i in cond_queries["source_item_id"]])
    return scores, true_idx


def main():
    from evaluation.run_eval import RETRIEVERS  # imported here so the maths above runs without torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="ensemble_mv", choices=list(RETRIEVERS))
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--out", default="results/conformal.csv")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    queries = pd.read_csv(args.queries, dtype={"source_item_id": str})
    retriever = RETRIEVERS[args.method]()
    if not hasattr(retriever, "embed_query"):
        raise SystemExit("conformal sets need an embedding model (not random / color_hist)")
    retriever.build_index(catalog)

    conditions = ["phone_mild", "phone_medium", "phone_strong"]
    data = {}
    for condition in conditions:
        print(f"embedding {condition} queries ...")
        data[condition] = similarity_matrix(retriever, catalog,
                                            queries[queries["condition"] == condition])

    rows = []
    for condition in conditions:
        scores, true_idx = data[condition]
        for alpha in ALPHAS:
            for method in ["threshold", "relative"]:
                r = repeated_splits(scores, true_idx, alpha, method, args.trials)
                rows.append(dict(model=args.method, setting=condition, alpha=alpha,
                                 target=1 - alpha, method=method, **r))

    # all three severities mixed together (still exchangeable: split at random)
    scores = np.vstack([data[c][0] for c in conditions])
    true_idx = np.concatenate([data[c][1] for c in conditions])
    for alpha in ALPHAS:
        for method in ["threshold", "relative"]:
            r = repeated_splits(scores, true_idx, alpha, method, args.trials)
            rows.append(dict(model=args.method, setting="all_severities", alpha=alpha,
                             target=1 - alpha, method=method, **r))

    # distribution shift: calibrate on mild photos, test on strong ones
    for alpha in ALPHAS:
        for method in ["threshold", "relative"]:
            r = shifted(*data["phone_mild"], *data["phone_strong"], alpha, method)
            rows.append(dict(model=args.method, setting="shift_mild_to_strong", alpha=alpha,
                             target=1 - alpha, method=method, **r))

    table = pd.DataFrame(rows)
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        table = pd.concat([old[old["model"] != args.method], table], ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)

    show = table[(table["model"] == args.method) & (table["alpha"] == 0.1)]
    pd.set_option("display.width", 160)
    print("\nTarget coverage 90% (alpha = 0.1):\n")
    print(show[["setting", "method", "coverage", "mean_size", "median_size",
                "singleton_rate", "singleton_accuracy"]].round(3).to_string(index=False))
    print(f"\nall alphas saved to {args.out}")


if __name__ == "__main__":
    main()
