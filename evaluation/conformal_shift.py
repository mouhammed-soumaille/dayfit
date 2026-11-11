"""T1: does the conformal guarantee survive the move from synthetic to REAL photos?

Two ways to calibrate, both tested on the same random halves of LookBench (real street photos):
  synthetic_to_real : calibrate on our 882 synthetic phone photos (all severities),
                      i.e. what we could do without any real labelled photos
  real_to_real      : calibrate on the OTHER half of LookBench (exchangeable, so the
                      guarantee applies again)

LookBench can have several gallery images of the same item, so a set "covers" a
query if it contains ANY gallery image with the query's item label. That only
changes what the "true score" is: the best similarity to any correct image.

Theory check (T1): if calibration scores come from distribution P and test scores
from Q, then coverage on Q >= 1 - alpha - TV(P, Q), where TV is the total variation
distance between the two 1-D score distributions. We estimate TV from histograms
(several bin counts, because the estimate depends on the bins) and compare the
predicted lower bound with the measured coverage.

Run from the repo root:
    python -m evaluation.conformal_shift --method finetuned_random
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.conformal import ALPHAS, relative_q, similarity_matrix, threshold_tau


def cutoffs(true_cal, top_cal, top_test, alpha, method):
    """Similarity cut-off for each test query, learned from the calibration queries."""
    if method == "threshold":
        return np.full(len(top_test), threshold_tau(true_cal, alpha))
    return top_test - relative_q(top_cal - true_cal, alpha)


def set_stats(scores_test, true_test, cutoff):
    """Coverage and set sizes on the test queries."""
    covered = true_test >= cutoff                       # a correct image made it into the set
    sizes = (scores_test >= cutoff[:, None]).sum(axis=1)
    return dict(coverage=covered.mean(), mean_size=sizes.mean(), median_size=np.median(sizes))


def histogram_tv(a, b, bins):
    """Total variation distance between two samples, using the same histogram bins for both."""
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    p = np.histogram(a, edges)[0] / len(a)
    q = np.histogram(b, edges)[0] / len(b)
    return 0.5 * np.abs(p - q).sum()


def synthetic_scores(retriever, catalog, queries):
    """Similarities of all synthetic phone photos (3 severities) with our catalog."""
    phone = queries[queries["condition"].isin(["phone_mild", "phone_medium", "phone_strong"])]
    scores, true_idx = similarity_matrix(retriever, catalog, phone)
    return scores, scores[np.arange(len(scores)), true_idx]


def lookbench_scores(retriever, gallery, lb_queries):
    """Similarities of the LookBench street photos with the LookBench gallery."""
    vectors = []
    for n, q in enumerate(lb_queries.itertuples(), start=1):
        vectors.append(retriever.embed_query(Image.open(q.image).convert("RGB")))
        print(f"  {n}/{len(lb_queries)}", end="\r")
    scores = np.stack(vectors) @ retriever.features.T
    labels = gallery["label"].to_numpy()
    # true score = best similarity to ANY gallery image of the same item
    true = np.array([scores[i, labels == lab].max() for i, lab in enumerate(lb_queries["label"])])
    return scores, true


def save(table, path, model):
    """Replace this model's old rows in the CSV with the new ones."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        table = pd.concat([old[old["model"] != model], table], ignore_index=True)
    table.to_csv(path, index=False)


def main():
    from evaluation.run_eval import RETRIEVERS
    from evaluation.run_lookbench import prepare, use_index_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="finetuned_random", choices=list(RETRIEVERS))
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    gallery, lb_queries = prepare("real_streetlook")
    lb_queries = lb_queries[lb_queries["label"].isin(gallery["label"])].reset_index(drop=True)

    # 1. scores on our synthetic photos (catalog gallery)
    retriever = RETRIEVERS[args.method]()
    retriever.build_index(catalog)
    print("embedding synthetic phone photos ...")
    syn_scores, syn_true = synthetic_scores(retriever, catalog, queries)

    # 2. scores on LookBench (switch the same model to the LookBench gallery)
    use_index_dir(retriever, os.path.join("models", "index_lookbench", "real_streetlook"))
    retriever.build_index(gallery)
    print("\nembedding LookBench street photos ...")
    lb_scores, lb_true = lookbench_scores(retriever, gallery, lb_queries)

    syn_top, lb_top = syn_scores.max(axis=1), lb_scores.max(axis=1)
    rng = np.random.default_rng(0)
    n = len(lb_scores)

    # 3. coverage and set size, both calibrations tested on the same LookBench halves
    rows = []
    for alpha in ALPHAS:
        for method in ["threshold", "relative"]:
            runs = {"synthetic_to_real": [], "real_to_real": []}
            for _ in range(args.trials):
                order = rng.permutation(n)
                cal, test = order[:n // 2], order[n // 2:]
                c = cutoffs(syn_true, syn_top, lb_top[test], alpha, method)
                runs["synthetic_to_real"].append(set_stats(lb_scores[test], lb_true[test], c))
                c = cutoffs(lb_true[cal], lb_top[cal], lb_top[test], alpha, method)
                runs["real_to_real"].append(set_stats(lb_scores[test], lb_true[test], c))
            for setting, res in runs.items():
                rows.append(dict(model=args.method, setting=setting, alpha=alpha,
                                 target=1 - alpha, method=method,
                                 **{k: float(np.mean([r[k] for r in res])) for k in res[0]}))

    # 4. T1: TV distance between the synthetic and real score distributions
    samples = {"threshold": (syn_true, lb_true),                    # score = similarity to true item
               "relative": (syn_top - syn_true, lb_top - lb_true)}  # score = gap below the top result
    tv_rows = []
    for method, (p_sample, q_sample) in samples.items():
        for bins in [10, 20, 50]:
            tv = histogram_tv(p_sample, q_sample, bins)
            tv_rows.append(dict(model=args.method, method=method, bins=bins,
                                tv=tv, bound_at_90=0.9 - tv))

    table, tv_table = pd.DataFrame(rows), pd.DataFrame(tv_rows)
    save(table, "results/conformal_shift.csv", args.method)
    save(tv_table, "results/conformal_tv.csv", args.method)

    pd.set_option("display.width", 160)
    print("\n\nTarget coverage 90% (alpha = 0.1), tested on real LookBench photos:\n")
    show = table[table["alpha"] == 0.1]
    print(show[["setting", "method", "coverage", "mean_size", "median_size"]].round(3).to_string(index=False))
    print("\nT1 bound: coverage >= 0.9 - TV (TV between synthetic and real scores):\n")
    print(tv_table[["method", "bins", "tv", "bound_at_90"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()