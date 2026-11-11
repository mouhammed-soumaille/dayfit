"""Evaluate our retrievers on LookBench (real product and street photos).

LookBench (Gao et al., 2026, arXiv:2601.14706) is an open fashion-retrieval
benchmark on Hugging Face (srpone/look-bench). We use two subsets:
  real_studio_flat : product photos -> product photos             (~1k queries, ~4k gallery)
  real_streetlook  : real street-style photos -> product photos   (~1k queries, ~3.3k gallery)

Following LookBench's own evaluation code, a retrieved gallery image counts as
correct if it has the same item_ID as the query ("official" protocol).

Duplicates: in real_streetlook, 3,278 gallery entries are only 2,537 distinct image
files; 633 files (outfit photos) are listed 2-5 times, once per item they show, with
identical embeddings. The official protocol keeps whichever copy the sort puts first,
so retrieving the right PICTURE can count as a miss. We therefore also report a
"dedup" protocol: merge copies of the same file (same md5) before taking the top 5;
a picture is correct if ANY of its copies has the query's item_ID, and counts for
category P@5 if any copy has the query's category. Both protocols are saved.

For street photos we can choose what the query image is (--query_image):
  image      : LookBench's own crop of the garment (made from a hand-drawn box)
  raw_image  : the full original photo (person, background, other clothes)
Running a crop_queries method on raw_image tests OUR automatic cropping on real
photos, between those two extremes.

The first run downloads the subset (a few hundred MB) and saves the images to
data/lookbench/<subset>/. Catalog embeddings go to models/index_lookbench/<subset>/.

Run from the repo root:
    python -m evaluation.run_lookbench --subset real_streetlook --method fashion_siglip
    python -m evaluation.run_lookbench --subset real_streetlook --method fashion_siglip --query_image raw_image
    python -m evaluation.run_lookbench --subset real_streetlook --method fashion_siglip_crop --query_image raw_image
"""
import argparse
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.run_eval import RETRIEVERS
from evaluation.stats import bootstrap_ci


def prepare(subset, root="data/lookbench"):
    """Download one LookBench subset once and save it as image files + 2 CSVs."""
    folder = os.path.join(root, subset)
    gallery_csv = os.path.join(folder, "gallery.csv")
    query_csv = os.path.join(folder, "queries.csv")
    if os.path.exists(gallery_csv) and os.path.exists(query_csv):
        return pd.read_csv(gallery_csv, dtype=str), pd.read_csv(query_csv, dtype=str)

    from datasets import load_dataset  # pip install datasets
    print(f"downloading LookBench '{subset}' ...")
    data = load_dataset("srpone/look-bench", subset)

    os.makedirs(os.path.join(folder, "gallery"), exist_ok=True)
    gallery_rows = []
    for i, row in enumerate(data["gallery"]):
        path = os.path.join(folder, "gallery", f"{i}.jpg")
        row["image"].convert("RGB").save(path, quality=92)
        gallery_rows.append(dict(item_id=f"g{i}", image_path=path,
                                 label=str(row["item_ID"]), category=row["category"]))

    os.makedirs(os.path.join(folder, "query"), exist_ok=True)
    os.makedirs(os.path.join(folder, "query_raw"), exist_ok=True)
    query_rows = []
    for i, row in enumerate(data["query"]):
        path = os.path.join(folder, "query", f"{i}.jpg")
        raw_path = os.path.join(folder, "query_raw", f"{i}.jpg")
        row["image"].convert("RGB").save(path, quality=92)
        (row["raw_image"] or row["image"]).convert("RGB").save(raw_path, quality=92)
        query_rows.append(dict(query_id=f"q{i}", image=path, raw_image=raw_path,
                               label=str(row["item_ID"]), category=row["category"]))

    gallery, queries = pd.DataFrame(gallery_rows), pd.DataFrame(query_rows)
    gallery.to_csv(gallery_csv, index=False)
    queries.to_csv(query_csv, index=False)
    print(f"saved {len(gallery)} gallery and {len(queries)} query images to {folder}/")
    return gallery, queries


def picture_groups(gallery):
    """Map each gallery entry to its image file's md5, plus the labels/categories of every picture."""
    picture = {i: hashlib.md5(open(p, "rb").read()).hexdigest()
               for i, p in zip(gallery["item_id"], gallery["image_path"])}
    labels, cats = defaultdict(set), defaultdict(set)
    for i, lab, cat in zip(gallery["item_id"], gallery["label"], gallery["category"]):
        labels[picture[i]].add(lab)
        cats[picture[i]].add(cat)
    return picture, labels, cats


def use_index_dir(retriever, index_dir):
    """Keep LookBench embeddings separate from our main catalog's embeddings."""
    for part in getattr(retriever, "parts", [retriever]):
        part.index_dir = index_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="real_streetlook",
                        choices=["real_studio_flat", "real_streetlook"])
    parser.add_argument("--method", default="fashion_siglip", choices=list(RETRIEVERS))
    parser.add_argument("--query_image", default="image", choices=["image", "raw_image"])
    parser.add_argument("--max_queries", type=int, default=0, help="0 = all")
    parser.add_argument("--out", default="results/lookbench_summary.csv")
    args = parser.parse_args()

    gallery, queries = prepare(args.subset)
    if args.max_queries:
        queries = queries.sample(n=min(args.max_queries, len(queries)), random_state=42)

    retriever = RETRIEVERS[args.method]()
    use_index_dir(retriever, os.path.join("models", "index_lookbench", args.subset))
    retriever.build_index(gallery)
    label_of = dict(zip(gallery["item_id"], gallery["label"]))
    category_of = dict(zip(gallery["item_id"], gallery["category"]))
    picture, pic_labels, pic_cats = picture_groups(gallery)
    max_copies = max(pd.Series(list(picture.values())).value_counts())
    k_search = 5 * max_copies               # enough results to always find 5 distinct pictures

    found = {p: dict(hit1=[], hit5=[], p5=[]) for p in ["official", "dedup"]}
    for n, q in enumerate(queries.itertuples(), start=1):
        image = Image.open(getattr(q, args.query_image)).convert("RGB")
        ranked = [item_id for item_id, _ in retriever.search(image, k=k_search)]

        top = ranked[:5]                    # official: gallery entries as they come
        labels = [label_of[i] for i in top]
        found["official"]["hit1"].append(float(labels[0] == q.label))
        found["official"]["hit5"].append(float(q.label in labels))
        found["official"]["p5"].append(np.mean([category_of[i] == q.category for i in top]))

        pictures = []                       # dedup: first 5 DISTINCT pictures
        for i in ranked:
            if picture[i] not in pictures:
                pictures.append(picture[i])
        pictures = pictures[:5]
        found["dedup"]["hit1"].append(float(q.label in pic_labels[pictures[0]]))
        found["dedup"]["hit5"].append(float(any(q.label in pic_labels[p] for p in pictures)))
        found["dedup"]["p5"].append(np.mean([q.category in pic_cats[p] for p in pictures]))
        print(f"  {n}/{len(queries)}", end="\r")

    # save each query's result so compare.py can run a paired test
    # (p5 here is category Precision@5; same column name as run_eval so compare.py can read it)
    os.makedirs("results/perquery", exist_ok=True)
    rows = []
    for protocol, values in found.items():
        condition = f"lookbench_{args.subset}_{args.query_image}"
        if protocol == "dedup":
            condition += "_dedup"
        perquery = pd.DataFrame(dict(query_id=queries["query_id"].values, **values))
        perquery.to_csv(f"results/perquery/{args.method}_{condition}.csv", index=False)

        row = dict(subset=args.subset, method=args.method, query_image=args.query_image,
                   protocol=protocol, n_queries=len(queries))
        for name, column in [("Recall@1", "hit1"), ("Recall@5", "hit5"), ("CategoryP@5", "p5")]:
            row[name] = float(np.mean(values[column]))
            row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(values[column])
        rows.append(row)

    new = pd.DataFrame(rows)
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        if "protocol" not in old.columns:   # rows saved before the dedup protocol existed
            old.insert(3, "protocol", "official")
        same = ((old["subset"] == args.subset) & (old["method"] == args.method)
                & (old["query_image"] == args.query_image))
        new = pd.concat([old[~same], new], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new.to_csv(args.out, index=False)

    print()
    for row in rows:
        print(row["protocol"])
        for name in ["Recall@1", "Recall@5", "CategoryP@5"]:
            print(f"  {name:12s} {row[name]:.3f}  [{row[name + '_lo']:.3f}, {row[name + '_hi']:.3f}]")


if __name__ == "__main__":
    main()
