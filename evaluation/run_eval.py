"""Evaluate any Retriever on data/queries.csv.

For each method and condition (clean / phone) it writes:
  results/<method>_<condition>.csv   one row per (query, rank): query_id, rank, item_id, score, latency_ms
  results/summary.csv                one row per method x condition with the averaged metrics

Metrics:
  Recall@1, Recall@5 : is the exact source item in the top 1 / 5?
  Precision@5        : source item removed, fraction of top 5 with the same article_type
  chance_P@5         : what a random retriever would get on average (sanity check)

Run from the repo root:
    python -m evaluation.run_eval --method random
"""
import argparse
import os
import time

import pandas as pd
from PIL import Image

from evaluation.metrics import precision_at_k, recall_at_k
from evaluation.random_retriever import RandomRetriever
from retrieval.color_hist import ColorHistRetriever


def make_clip():
    # imported here so random / color_hist still run without torch installed
    from retrieval.clip_retriever import OpenClipRetriever
    # the original OpenAI CLIP ViT-B/32 (the OpenAI weights need the "quickgelu" variant)
    return OpenClipRetriever("clip", "ViT-B-32-quickgelu", pretrained="openai")


# Add new methods here as you write them (name -> function that creates the retriever)
RETRIEVERS = {
    "random": RandomRetriever,
    "color_hist": ColorHistRetriever,
    "clip": make_clip,
}


def evaluate(retriever, catalog, queries, k=5, out_dir="results"):
    retriever.build_index(catalog)
    label_of = dict(zip(catalog["item_id"], catalog["article_type"]))
    type_share = catalog["article_type"].value_counts(normalize=True)

    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    for condition, cond_queries in queries.groupby("condition"):
        result_rows = []
        r1, r5, p5, chance, latencies = [], [], [], [], []

        for q in cond_queries.itertuples():
            image = Image.open(q.image_path).convert("RGB")

            start = time.perf_counter()
            results = retriever.search(image, k=k + 1)  # +1 so we still have k after removing the source
            latency_ms = (time.perf_counter() - start) * 1000

            ranked_ids = [item_id for item_id, _ in results]
            r1.append(recall_at_k(ranked_ids, q.source_item_id, 1))
            r5.append(recall_at_k(ranked_ids, q.source_item_id, k))

            others = [i for i in ranked_ids if i != q.source_item_id][:k]
            p5.append(precision_at_k([label_of[i] for i in others], q.article_type, k))
            chance.append(type_share[q.article_type])
            latencies.append(latency_ms)

            for rank, (item_id, score) in enumerate(results[:k], start=1):
                result_rows.append(dict(query_id=q.query_id, rank=rank, item_id=item_id,
                                        score=score, latency_ms=latency_ms))

        pd.DataFrame(result_rows).to_csv(
            os.path.join(out_dir, f"{retriever.name}_{condition}.csv"), index=False)

        n = len(cond_queries)
        summary_rows.append({
            "method": retriever.name,
            "condition": condition,
            "n_queries": n,
            "Recall@1": sum(r1) / n,
            "Recall@5": sum(r5) / n,
            "Precision@5": sum(p5) / n,
            "chance_P@5": sum(chance) / n,
            "mean_latency_ms": sum(latencies) / n,
        })

    return pd.DataFrame(summary_rows)


def update_summary(new_rows, path):
    """Replace this method's old rows in summary.csv (if any) with the new ones."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["method"].isin(new_rows["method"])]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)
    return new_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="random", choices=list(RETRIEVERS))
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    queries = pd.read_csv(args.queries, dtype={"source_item_id": str})

    retriever = RETRIEVERS[args.method]()
    summary = evaluate(retriever, catalog, queries, out_dir=args.out_dir)
    full = update_summary(summary, os.path.join(args.out_dir, "summary.csv"))

    pd.set_option("display.width", 120)
    print(full.round(4).to_string(index=False))


if __name__ == "__main__":
    main()