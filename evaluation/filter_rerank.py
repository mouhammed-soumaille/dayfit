"""C2: menu filter + visual re-ranking ("does the model help if the shopper knows the type?").

The shopper filters by garment type (and colour on synthetic data), and the filtered items are
sorted by the model's similarity to the photo instead of a random order (C1).
Rank of the shopper's item = 1 + number of allowed, non-matching items that score higher than
the item's best picture. If the item is filtered out, it can never be found (rank = inf).
Filter "none" is the plain model and must reproduce run_eval / LookBench Recall@1 (built-in check).
Filtering only removes competitors, so whenever the item survives the filter its rank can only go
down (improve). On LookBench, a few items are filtered out by inconsistent category labels.

Run from the repo root:
    python -m evaluation.filter_rerank --method finetuned_random
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.conformal_shift import lookbench_scores, save
from evaluation.run_eval import RETRIEVERS, has_source


def first_rank(s, relevant, allowed):
    """Rank of the best relevant item among the allowed items (inf if none is allowed)."""
    ok = relevant & allowed
    if not ok.any():
        return np.inf
    best = s[ok].max()
    return 1 + int(np.sum(allowed & ~relevant & (s > best)))


def summarize(data, condition, filter_name, ranks):
    ranks = np.array(ranks, dtype=float)
    found = ranks[np.isfinite(ranks)]
    return dict(data=data, condition=condition, filter=filter_name, n_queries=len(ranks),
                filtered_out=int(np.isinf(ranks).sum()),
                recall1=np.mean(ranks <= 1), recall5=np.mean(ranks <= 5),
                median_rank=np.median(found), mean_rank=np.mean(found))


def synthetic_rows(retriever):
    catalog = pd.read_csv("data/catalog.csv", dtype=str).fillna({"color": "unknown"})
    queries = pd.read_csv("data/queries.csv", dtype=str).fillna({"color": "unknown"})
    queries = queries[queries["source_item_id"].apply(has_source)]
    retriever.build_index(catalog)
    ids = catalog["item_id"].to_numpy()          # index rows are in catalogue order
    types = catalog["article_type"].to_numpy()
    colors = catalog["color"].to_numpy()

    rows = []
    for condition, cond_queries in queries.groupby("condition"):
        ranks = {"none": [], "type": [], "type + colour": []}
        for n, q in enumerate(cond_queries.itertuples(), start=1):
            image = Image.open(q.image_path).convert("RGB")
            s = retriever.features @ retriever.embed_query(image)
            relevant = ids == q.source_item_id
            same_type = types == q.article_type
            ranks["none"].append(first_rank(s, relevant, np.ones(len(ids), bool)))
            ranks["type"].append(first_rank(s, relevant, same_type))
            ranks["type + colour"].append(first_rank(s, relevant, same_type & (colors == q.color)))
            print(f"  {condition} {n}/{len(cond_queries)}", end="\r")
        for name, r in ranks.items():
            rows.append(summarize("synthetic", condition, name, r))
    return rows


def lookbench_rows(retriever):
    from evaluation.run_lookbench import prepare, use_index_dir
    gallery, lb = prepare("real_streetlook")
    lb = lb[lb["label"].isin(gallery["label"])].reset_index(drop=True)
    use_index_dir(retriever, os.path.join("models", "index_lookbench", "real_streetlook"))
    retriever.build_index(gallery)
    print("\nembedding LookBench street photos ...")
    scores, _ = lookbench_scores(retriever, gallery, lb)
    labels = gallery["label"].to_numpy()
    cats = gallery["category"].to_numpy()

    ranks = {"none": [], "type": []}
    for i, q in enumerate(lb.itertuples()):
        relevant = labels == q.label
        ranks["none"].append(first_rank(scores[i], relevant, np.ones(len(labels), bool)))
        ranks["type"].append(first_rank(scores[i], relevant, cats == q.category))
    return [summarize("lookbench_real", "real_streetlook", name, r) for name, r in ranks.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="finetuned_random", choices=list(RETRIEVERS))
    args = parser.parse_args()

    retriever = RETRIEVERS[args.method]()
    table = pd.DataFrame(synthetic_rows(retriever) + lookbench_rows(retriever))
    table.insert(0, "model", args.method)
    save(table, "results/filter_rerank.csv", args.method)
    pd.set_option("display.width", 160)
    print("\n")
    print(table.drop(columns="model").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
