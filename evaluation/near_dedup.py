"""D8c: sensitivity of the LookBench results to near-duplicate gallery pictures.

D8a found gallery pictures that are the same photo saved differently (md5 differs, so
the dedup protocol of Finding 15 does not merge them). Rule fixed BEFORE this analysis:
two pictures are near-duplicates if their PRETRAINED FashionSigLIP cosine is >= 0.97;
0.95 and 0.98 are sensitivity checks.

"near-dedup" protocol: near-duplicate pictures are joined into groups (if A~B and B~C,
then A, B, C are one group: connected components, found with union-find). A query's
top 5 = the first 5 distinct GROUPS in its ranking (saved top-20 distinct pictures in
results/rankings/). A group is correct if ANY of its pictures shows the query's item,
and counts for category P@5 if any of its pictures has the query's category; exactly
the dedup rule, with "same file" replaced by "same photo".

Sanity check: with no near-duplicate merging (threshold None), the script must reproduce
the saved dedup per-query hit1 / hit5 / p5 of every model exactly.

Outputs:
  results/near_dedup_summary.csv  R@1, R@5, P@5 (95% CI) per model and threshold
  results/near_dedup_tests.csv    paired bootstrap of the Finding 7 comparisons per threshold

Run from the repo root:
    python -m evaluation.near_dedup
"""
import hashlib

import numpy as np
import pandas as pd

from evaluation.stats import bootstrap_ci, paired_bootstrap

LB = "data/lookbench/real_streetlook"
INDEX = "models/index_lookbench/real_streetlook/fashion_siglip"
METHODS = ["fashion_siglip", "ensemble_mv", "finetuned", "finetuned_random"]
THRESHOLDS = [None, 0.98, 0.97, 0.95]       # None = plain dedup (sanity check); 0.97 = main
COMPARISONS = [("fashion_siglip", "finetuned_random"),
               ("finetuned", "finetuned_random"),
               ("ensemble_mv", "finetuned_random")]


def find(parent, x):
    """Union-find: the representative of x's group (with path halving)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def picture_groups(pics, emb, threshold):
    """Map each distinct picture (md5) to a group id; pictures with cosine >= threshold are joined."""
    parent = list(range(len(pics)))
    if threshold is not None:
        sim = emb @ emb.T
        for a, b in zip(*np.where(np.triu(sim, k=1) >= threshold)):
            parent[find(parent, a)] = find(parent, b)
    return {pic: find(parent, k) for k, pic in enumerate(pics)}


def evaluate(ranks, queries, group_of, labels, cats):
    """Per-query hit1, hit5, p5 using the first 5 distinct groups of each query's ranking."""
    group_labels, group_cats = {}, {}
    for pic, g in group_of.items():
        group_labels.setdefault(g, set()).update(labels[pic])
        group_cats.setdefault(g, set()).update(cats[pic])
    rows, short = [], 0
    for q in queries.itertuples():
        groups = []
        for pic in ranks.get(q.query_id, []):
            if group_of[pic] not in groups:
                groups.append(group_of[pic])
        short += len(groups) < 5
        top = groups[:5]
        rows.append(dict(query_id=q.query_id,
                         hit1=float(q.label in group_labels[top[0]]),
                         hit5=float(any(q.label in group_labels[g] for g in top)),
                         p5=np.mean([q.category in group_cats[g] for g in top])))
    return pd.DataFrame(rows), short


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    labels = gallery.groupby("picture")["label"].apply(set)
    cats = gallery.groupby("picture")["category"].apply(set)

    # one pretrained embedding per distinct picture (copies of a file have identical embeddings)
    emb = np.load(f"{INDEX}.npy")
    with open(f"{INDEX}_paths.txt") as f:
        assert f.read().split("\n")[1:] == list(gallery["image_path"]), "embeddings not in gallery order"
    first_row = gallery.reset_index().groupby("picture")["index"].first()
    pics = list(first_row.index)
    pic_emb = emb[first_row.values]

    # each model's ranking: query_id -> list of distinct pictures, best first
    rankings = {}
    for m in METHODS:
        r = pd.read_csv(f"results/rankings/{m}_lookbench_real_streetlook_image_dedup.csv")
        r = r.sort_values(["query_id", "rank"])
        rankings[m] = r.groupby("query_id")["picture"].apply(list).to_dict()

    summary, tests, perquery = [], [], {}
    for t in THRESHOLDS:
        group_of = picture_groups(pics, pic_emb, t)
        n_groups = len(set(group_of.values()))
        name = "dedup" if t is None else f"near_{t:.2f}"
        print(f"\n{name}: {len(pics)} pictures -> {n_groups} groups")
        for m in METHODS:
            pq, short = evaluate(rankings[m], queries, group_of, labels, cats)
            perquery[(name, m)] = pq
            if t is None:                                    # sanity check against the saved files
                saved = pd.read_csv(f"results/perquery/{m}_lookbench_real_streetlook_image_dedup.csv")
                same = all(np.allclose(pq[c], saved[c]) for c in ["hit1", "hit5", "p5"])
                print(f"  check {m}: reproduces saved dedup per-query results -> {'OK' if same else 'MISMATCH'}")
                if not same:
                    raise SystemExit("sanity check failed, not saving")
            row = dict(protocol=name, threshold=t, n_groups=n_groups, method=m,
                       queries_with_fewer_than_5_groups=short)
            for metric, col in [("Recall@1", "hit1"), ("Recall@5", "hit5"), ("CategoryP@5", "p5")]:
                row[metric] = pq[col].mean()
                row[f"{metric}_lo"], row[f"{metric}_hi"] = bootstrap_ci(pq[col])
            summary.append(row)
            print(f"  {m:17s} R@1 {row['Recall@1']:.3f}  R@5 {row['Recall@5']:.3f}  "
                  f"P@5 {row['CategoryP@5']:.3f}  (queries with < 5 groups in top 20: {short})")
        for a, b in COMPARISONS:
            for metric, col in [("Recall@1", "hit1"), ("CategoryP@5", "p5")]:
                diff, lo, hi, p = paired_bootstrap(perquery[(name, b)][col], perquery[(name, a)][col])
                tests.append(dict(protocol=name, metric=metric, change=f"{a} -> {b}",
                                  difference=diff, lo=lo, hi=hi, p_value=p))
                print(f"  {a} -> {b}  {metric}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")

    pd.DataFrame(summary).to_csv("results/near_dedup_summary.csv", index=False)
    pd.DataFrame(tests).to_csv("results/near_dedup_tests.csv", index=False)
    print("\nsaved results/near_dedup_summary.csv and results/near_dedup_tests.csv")


if __name__ == "__main__":
    main()