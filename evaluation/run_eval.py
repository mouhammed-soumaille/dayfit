"""Evaluate any Retriever on data/queries.csv.

For each method and condition (clean / phone_mild / phone_medium / phone_strong / real)
it writes:
  results/<method>_<condition>.csv           top-5 results: query_id, rank, item_id, score, latency_ms
  results/perquery/<method>_<condition>.csv  one row per query: hit@1, hit@5, P@5 (for paired tests)
  results/summary.csv                        one row per method x condition, with 95% CIs

Metrics:
  Recall@1, Recall@5 : is the exact source item in the top 1 / 5?
                       (not defined for "real" photos of your own clothes -> left empty)
  Precision@5        : source item removed, fraction of top 5 with the same article_type
  chance_P@5         : what a random retriever would get on average (sanity check)
  *_lo, *_hi         : 95% bootstrap confidence interval (see evaluation/stats.py)

Run from the repo root:
    python -m evaluation.run_eval --method fashion_siglip
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.metrics import precision_at_k, recall_at_k
from evaluation.random_retriever import RandomRetriever
from evaluation.stats import bootstrap_ci
from retrieval.color_hist import ColorHistRetriever

SIGLIP = "hf-hub:Marqo/marqo-fashionSigLIP"
FCLIP = "hf-hub:Marqo/marqo-fashionCLIP"


def open_clip_model(name, model_name, pretrained=None, **options):
    """Returns a function that builds an OpenClipRetriever when called.
    (torch is imported only then, so random / color_hist run without it.)
    options: crop_queries, crop_catalog, multiview, index_name, checkpoint_path"""
    def make():
        from retrieval.clip_retriever import OpenClipRetriever
        return OpenClipRetriever(name, model_name, pretrained=pretrained, **options)
    return make


def ensemble(name, *part_makers):
    def make():
        from retrieval.ensemble import EnsembleRetriever
        return EnsembleRetriever(name, [m() for m in part_makers])
    return make


# Add new methods here as you write them (name -> function that creates the retriever).
# index_name = which saved catalog embeddings to use. Variants that only change the
# queries share an index; variants that change the catalog get their own.
RETRIEVERS = {
    # ---- ladder rows 0-2 ----
    "random": RandomRetriever,
    "color_hist": ColorHistRetriever,
    "clip": open_clip_model("clip", "ViT-B-32-quickgelu", pretrained="openai"),
    "fashion_siglip": open_clip_model("fashion_siglip", SIGLIP),
    "fashion_clip": open_clip_model("fashion_clip", FCLIP),

    # ---- row 3: garment cropping ----
    # crop the query only
    "color_hist_crop": lambda: ColorHistRetriever(crop_queries=True),
    "fashion_clip_crop": open_clip_model("fashion_clip_crop", FCLIP,
                                         crop_queries=True, index_name="fashion_clip"),
    "fashion_siglip_crop": open_clip_model("fashion_siglip_crop", SIGLIP,
                                           crop_queries=True, index_name="fashion_siglip"),
    # crop the query AND every catalog image, so both are framed the same way
    "fashion_siglip_cropboth": open_clip_model("fashion_siglip_cropboth", SIGLIP, crop_queries=True,
                                               crop_catalog=True, index_name="fashion_siglip_cropcat"),

    # ---- row 4: multi-view catalog embeddings ----
    "fashion_siglip_mv": open_clip_model("fashion_siglip_mv", SIGLIP, multiview=True),
    "fashion_clip_mv": open_clip_model("fashion_clip_mv", FCLIP, multiview=True),

    # ---- row 5: ensemble of the two best models ----
    "ensemble": ensemble("ensemble",
                         open_clip_model("fashion_siglip", SIGLIP),
                         open_clip_model("fashion_clip", FCLIP)),
    "ensemble_mv": ensemble("ensemble_mv",
                            open_clip_model("fashion_siglip_mv", SIGLIP, multiview=True),
                            open_clip_model("fashion_clip_mv", FCLIP, multiview=True)),

    # ---- row 7: our fine-tuned model (once models/finetuned.pt exists) ----
    "finetuned": open_clip_model("finetuned", SIGLIP, checkpoint_path="models/finetuned.pt"),
}


def has_source(source_item_id):
    """Real photos of your own clothes have no source item in the catalog."""
    return isinstance(source_item_id, str) and source_item_id not in ("", "nan")


def evaluate(retriever, catalog, queries, k=5, out_dir="results"):
    retriever.build_index(catalog)
    label_of = dict(zip(catalog["item_id"], catalog["article_type"]))
    type_share = catalog["article_type"].value_counts(normalize=True)

    os.makedirs(os.path.join(out_dir, "perquery"), exist_ok=True)
    summary_rows = []

    for condition, cond_queries in queries.groupby("condition"):
        result_rows, per_query = [], []

        for q in cond_queries.itertuples():
            image = Image.open(q.image_path).convert("RGB")

            start = time.perf_counter()
            results = retriever.search(image, k=k + 1)  # +1 so we still have k after removing the source
            latency_ms = (time.perf_counter() - start) * 1000

            ranked_ids = [item_id for item_id, _ in results]
            if has_source(q.source_item_id):
                hit1 = recall_at_k(ranked_ids, q.source_item_id, 1)
                hit5 = recall_at_k(ranked_ids, q.source_item_id, k)
            else:
                hit1 = hit5 = np.nan  # exact-item recall is undefined here

            others = [i for i in ranked_ids if i != q.source_item_id][:k]
            p5 = precision_at_k([label_of[i] for i in others], q.article_type, k)
            per_query.append(dict(query_id=q.query_id, hit1=hit1, hit5=hit5, p5=p5,
                                  chance=type_share.get(q.article_type, 0.0),
                                  latency_ms=latency_ms))

            for rank, (item_id, score) in enumerate(results[:k], start=1):
                result_rows.append(dict(query_id=q.query_id, rank=rank, item_id=item_id,
                                        score=score, latency_ms=latency_ms))

        pd.DataFrame(result_rows).to_csv(
            os.path.join(out_dir, f"{retriever.name}_{condition}.csv"), index=False)
        pq = pd.DataFrame(per_query)
        pq.to_csv(os.path.join(out_dir, "perquery", f"{retriever.name}_{condition}.csv"), index=False)

        row = {"method": retriever.name, "condition": condition, "n_queries": len(pq)}
        for metric, column in [("Recall@1", "hit1"), ("Recall@5", "hit5"), ("Precision@5", "p5")]:
            row[metric] = pq[column].mean()  # pandas mean skips empty (NaN) values
            row[f"{metric}_lo"], row[f"{metric}_hi"] = bootstrap_ci(pq[column])
        row["chance_P@5"] = pq["chance"].mean()
        row["mean_latency_ms"] = pq["latency_ms"].mean()
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def update_summary(new_rows, path):
    """Replace this method's old rows (for these conditions) in summary.csv with the new ones."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        replaced = (old["method"].isin(new_rows["method"])
                    & old["condition"].isin(new_rows["condition"]))
        new_rows = pd.concat([old[~replaced], new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)
    return new_rows


def pretty(summary):
    """Short table for the terminal: value [CI low, CI high]."""
    def fmt(r, m):
        if pd.isna(r[m]):
            return "-"
        return f"{r[m]:.3f} [{r[m + '_lo']:.2f},{r[m + '_hi']:.2f}]"
    return pd.DataFrame({
        "method": summary["method"], "condition": summary["condition"],
        "Recall@1": [fmt(r, "Recall@1") for _, r in summary.iterrows()],
        "Recall@5": [fmt(r, "Recall@5") for _, r in summary.iterrows()],
        "Precision@5": [fmt(r, "Precision@5") for _, r in summary.iterrows()],
        "ms": summary["mean_latency_ms"].round(1),
    }).to_string(index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="random", choices=list(RETRIEVERS))
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--conditions", default="",
                        help="comma-separated subset, e.g. 'real' (default: all)")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    queries = pd.read_csv(args.queries, dtype={"source_item_id": str})
    if args.conditions:
        queries = queries[queries["condition"].isin(args.conditions.split(","))]

    retriever = RETRIEVERS[args.method]()
    summary = evaluate(retriever, catalog, queries, out_dir=args.out_dir)
    update_summary(summary, os.path.join(args.out_dir, "summary.csv"))

    pd.set_option("display.width", 160)
    print(pretty(summary))


if __name__ == "__main__":
    main()
