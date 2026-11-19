"""LookBench real_streetlook against the OFFICIAL corpus: its 3,278-item gallery + 58,275 noise images.

Problem: all our LookBench numbers search only the 3,278 gallery items. The benchmark's official corpus
for this subset has 61,553 entries (Table 1 of the paper = 3,278 + the 58,275 shared "noise" images, as
version_info.json on Hugging Face confirms). The app searches 54k items, so the bigger corpus is much
closer to the app. Noise images are other fashion products with no category label, so only exact-item
metrics (R@1, R@5) are defined here; group P@5 stays a 3,278-gallery metric.

What it does, for finetuned_random and the pretrained FashionSigLIP, for the crop and the whole photo:
  - downloads the noise images once (data/lookbench/noise/, about 1.4 GB) and embeds them once per model
    (models/index_lookbench/noise/); the 3,278 gallery embeddings are the saved ones;
  - ranks all 61,553 entries with the dedup protocol (identical image files merged, Finding 15);
  - R@1 and R@5 with 95% CIs, and the share of noise images in each top 5;
  - paired cluster bootstrap (split clusters) finetuned_random - fashion_siglip, all 981 and TEST half.
Sanity check (stops if it fails): ranking only the 3,278 gallery entries reproduces the saved per-query
R@1 of every query (results/perquery/..._dedup.csv).

Run from the repo root (download + embedding: roughly 30-45 minutes on MPS the first time):
    python -m evaluation.lookbench_full
Writes results/lookbench_full.csv, results/perquery/*_full.csv, results/rankings/*_full.csv.
"""
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.run_eval import RETRIEVERS
from evaluation.run_lookbench import SAVE_DEPTH, prepare, use_index_dir
from evaluation.stats import bootstrap_ci

LB = "data/lookbench/real_streetlook"
NOISE = "data/lookbench/noise"
MODELS = ["finetuned_random", "fashion_siglip"]
VIEWS = ["image", "raw_image"]            # LookBench crop, whole photo
CANDIDATES = 300                          # scores looked at per query; enough for 20 distinct pictures


def prepare_noise():
    """Download the noise images once and save them as files + a CSV (like run_lookbench.prepare)."""
    csv = os.path.join(NOISE, "gallery.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv, dtype=str)
    from datasets import load_dataset
    print("downloading the LookBench noise images (about 1.4 GB) ...")
    data = load_dataset("srpone/look-bench", "noise", split="gallery")
    os.makedirs(os.path.join(NOISE, "gallery"), exist_ok=True)
    rows = []
    for i, row in enumerate(data):
        path = os.path.join(NOISE, "gallery", f"{i}.jpg")
        row["image"].convert("RGB").save(path, quality=92)
        rows.append(dict(item_id=f"n{i}", image_path=path, label=str(row["item_ID"]), category="noise"))
        if i % 1000 == 0:
            print(f"  {i}/{len(data)}", end="\r")
    noise = pd.DataFrame(rows)
    noise.to_csv(csv, index=False)
    print(f"\nsaved {len(noise)} noise images to {NOISE}/")
    return noise


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def distinct_pictures(order, picture, depth):
    """The first `depth` distinct pictures (and the entry that first showed each) of a ranked list."""
    seen, entries = [], []
    for i in order:
        if picture[i] not in seen:
            seen.append(picture[i])
            entries.append(i)
            if len(seen) == depth:
                break
    return seen, entries


def top(scores, n):
    """Indices of the n highest scores, best first."""
    n = min(n, len(scores))
    best = np.argpartition(-scores, n - 1)[:n]
    return best[np.argsort(-scores[best])]


def evaluate(method, gallery, noise, queries):
    """Rank the full corpus for every query and view; check the 3,278-gallery ranking against saved R@1."""
    retriever = RETRIEVERS[method]()
    use_index_dir(retriever, "models/index_lookbench/real_streetlook")
    retriever.build_index(gallery)                     # the saved gallery embeddings (no re-embedding)
    real = retriever.features
    use_index_dir(retriever, "models/index_lookbench/noise")
    retriever.build_index(noise)                       # noise embeddings, made once and saved
    features = np.concatenate([real, retriever.features])
    corpus = pd.concat([gallery, noise], ignore_index=True)
    picture = corpus["picture"].to_numpy()
    labels = corpus.groupby("picture")["label"].apply(set).to_dict()
    is_noise = corpus.groupby("picture")["category"].apply(lambda c: set(c) == {"noise"}).to_dict()
    n_real = len(gallery)

    rows = []
    for view in VIEWS:
        saved = pd.read_csv(f"results/perquery/{method}_lookbench_real_streetlook_{view}_dedup.csv")
        saved_hit1 = dict(zip(saved["query_id"], saved["hit1"]))
        per_query, ranking_rows, mismatches = [], [], 0
        for n, q in enumerate(queries.itertuples(), start=1):
            query = retriever.embed_query(Image.open(getattr(q, view)).convert("RGB"))
            scores = features @ query
            # check: the 3,278 gallery alone must give the saved rank-1 result
            gallery_pics, _ = distinct_pictures(top(scores[:n_real], CANDIDATES), picture, 1)
            mismatches += float(q.label in labels[gallery_pics[0]]) != saved_hit1[q.query_id]
            # full corpus
            pics, entries = distinct_pictures(top(scores, CANDIDATES), picture, SAVE_DEPTH)
            per_query.append(dict(query_id=q.query_id, hit1=float(q.label in labels[pics[0]]),
                                  hit5=float(any(q.label in labels[p] for p in pics[:5])),
                                  noise_in_top5=float(np.mean([is_noise[p] for p in pics[:5]]))))
            for rank, (p, i) in enumerate(zip(pics, entries), start=1):
                ranking_rows.append(dict(query_id=q.query_id, rank=rank, item_id=corpus.loc[i, "item_id"],
                                         picture=p, score=float(scores[i]), noise=is_noise[p]))
            print(f"  {method} {view} {n}/{len(queries)}", end="\r")
        print(f"\n  check {method} {view}: gallery-only rank 1 differs from the saved result for "
              f"{mismatches} of {len(queries)} queries")
        if mismatches:
            raise SystemExit("PROBLEM: the gallery-only ranking does not reproduce the saved R@1; stopping.")

        per_query = pd.DataFrame(per_query)
        per_query.to_csv(f"results/perquery/{method}_lookbench_real_streetlook_{view}_full.csv", index=False)
        pd.DataFrame(ranking_rows).to_csv(f"results/rankings/{method}_lookbench_real_streetlook_{view}_full.csv",
                                          index=False)
        row = dict(method=method, view=view, corpus_entries=len(corpus), n_queries=len(per_query),
                   gallery_R1=float(np.mean(list(saved_hit1.values()))),
                   noise_share_top5=per_query["noise_in_top5"].mean())
        for name, column in [("R1", "hit1"), ("R5", "hit5")]:
            row[name] = per_query[column].mean()
            row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(per_query[column])
        rows.append(row)
    return rows


def compare(queries):
    """Paired cluster bootstrap finetuned_random - fashion_siglip on the full corpus."""
    rows = []
    for view in VIEWS:
        a = pd.read_csv(f"results/perquery/finetuned_random_lookbench_real_streetlook_{view}_full.csv")
        b = pd.read_csv(f"results/perquery/fashion_siglip_lookbench_real_streetlook_{view}_full.csv")
        a, b = a.set_index("query_id").loc[queries["query_id"]], b.set_index("query_id").loc[queries["query_id"]]
        for side in ["all", "test"]:
            keep = (queries["side"] == "test").to_numpy() if side == "test" else np.ones(len(queries), bool)
            for metric in ["hit1", "hit5"]:
                diff, lo, hi, p = cluster_bootstrap(a[metric][keep], b[metric][keep], queries.loc[keep, "cluster"].to_numpy())
                rows.append(dict(method="finetuned_random - fashion_siglip", view=view, side=side, metric=metric,
                                 difference=diff, lo=lo, hi=hi, p=p))
    return rows


def main():
    gallery, queries = prepare("real_streetlook")
    queries = queries.merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    noise = prepare_noise()
    print(f"corpus: {len(gallery)} gallery + {len(noise)} noise = {len(gallery) + len(noise)} entries")
    gallery["picture"] = [md5(p) for p in gallery["image_path"]]
    noise["picture"] = [md5(p) for p in noise["image_path"]]
    print(f"  noise images that are byte-identical to a gallery picture: "
          f"{int(noise['picture'].isin(set(gallery['picture'])).sum())}")

    rows = []
    for method in MODELS:
        rows += evaluate(method, gallery, noise, queries)
    table = pd.DataFrame(rows + compare(queries))
    table.to_csv("results/lookbench_full.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print("saved results/lookbench_full.csv")


if __name__ == "__main__":
    main()
