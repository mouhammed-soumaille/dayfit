"""E1: the search engine behind the app.

Loads the recommended model (finetuned_random) and its precomputed catalogue embeddings
ONCE, then answers searches: photo in, top-k catalogue items out (with type, colour,
image path and similarity score). No photo is ever written anywhere: the image only
lives in memory while it is embedded.

This is a thin layer over the retriever used in all experiments (RETRIEVERS in
evaluation/run_eval.py), so the app searches exactly like the evaluated model.

Two catalogues (chosen with the environment variable DAYFIT_CATALOG):
  evaluation (default): data/catalog.csv, the 6,197 items every reported number uses
  app                 : data/app_catalog.csv, ~33k items (data_prep/app_catalog.py), for the demo only
Each catalogue has its own saved embeddings in models/index/, so they never overwrite each other.

Sanity checks (run this file):
  evaluation: for 10 saved synthetic test queries, the engine returns the same top 5 items
              as the saved evaluation results, with the same scores.
  app       : the first 6,197 rows of the app catalogue ARE the evaluation catalogue, so their
              new embeddings must equal the saved evaluation embeddings; and since the app
              catalogue contains the evaluation one, each query's best score can only go up.

Run from the repo root:
    python -m serving.engine
    python -m serving.engine --catalog app        (the first run embeds ~33k images, several minutes)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.run_eval import RETRIEVERS

METHOD = "finetuned_random"
CATALOGS = {"evaluation": "data/catalog.csv", "app": "data/app_catalog.csv"}


class SearchEngine:
    def __init__(self, method=METHOD, catalogue=None):
        self.catalogue = catalogue or os.environ.get("DAYFIT_CATALOG", "evaluation")
        self.catalog = pd.read_csv(CATALOGS[self.catalogue], dtype={"item_id": str})
        self.retriever = RETRIEVERS[method]()
        if self.catalogue != "evaluation":                # own embeddings file, e.g. finetuned_random_app.npy
            self.retriever.index_name = f"{method}_{self.catalogue}"
        self.retriever.build_index(self.catalog)          # loads the saved embeddings (or makes them once)
        self.item = self.catalog.set_index("item_id")
        self.method = method

    def search(self, image, k=5):
        """image: PIL image. Returns a list of dicts, best match first, plus search time in ms."""
        start = time.perf_counter()
        results = self.retriever.search(image.convert("RGB"), k=k)
        ms = (time.perf_counter() - start) * 1000
        found = []
        for rank, (item_id, score) in enumerate(results, start=1):
            row = self.item.loc[item_id]
            found.append(dict(rank=rank, item_id=item_id, score=float(score),
                              article_type=row["article_type"], color=row["color"],
                              image_path=row["image_path"]))
        return found, ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="evaluation", choices=list(CATALOGS))
    args = parser.parse_args()

    engine = SearchEngine(catalogue=args.catalog)
    saved = pd.read_csv(f"results/{METHOD}_phone_medium.csv", dtype={"item_id": str})
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str}).set_index("query_id")

    if args.catalog == "app":
        # check 1: the evaluation items got the same embeddings as in the saved evaluation index
        old = np.load(f"models/index/{METHOD}.npy")
        new = engine.retriever.features[:len(old)]
        same_order = list(engine.catalog["item_id"][:len(old)]) == list(
            pd.read_csv(CATALOGS["evaluation"], dtype={"item_id": str})["item_id"])
        diff = float(np.abs(new - old).max())
        print(f"  check 1: first {len(old)} items in the same order -> {same_order}; "
              f"largest embedding difference {diff:.1e}")
        if not same_order or diff > 1e-3:
            raise SystemExit("sanity check failed: app index does not contain the evaluation index")
        # check 2: a bigger catalogue can only raise each query's best score
        for query_id in saved["query_id"].unique()[:10]:
            old_best = saved[saved["query_id"] == query_id]["score"].max()
            found, ms = engine.search(Image.open(queries.loc[query_id, "image_path"]))
            ok = found[0]["score"] >= old_best - 1e-3
            print(f"  {query_id}: best score {old_best:.3f} -> {found[0]['score']:.3f} "
                  f"({found[0]['article_type']})  {'OK' if ok else 'LOWER'}  ({ms:.0f} ms)")
            if not ok:
                raise SystemExit("sanity check failed: a best score went down")
        print(f"  check 2: 10 queries, no best score went down; catalogue size {len(engine.item)}")
        return

    worst = 0.0
    for query_id in saved["query_id"].unique()[:10]:
        expected = saved[saved["query_id"] == query_id].sort_values("rank")
        found, ms = engine.search(Image.open(queries.loc[query_id, "image_path"]))
        same_items = [f["item_id"] for f in found] == list(expected["item_id"])
        worst = max(worst, max(abs(f["score"] - s) for f, s in zip(found, expected["score"])))
        print(f"  {query_id}: same top 5 as saved -> {'OK' if same_items else 'DIFFERENT'}  ({ms:.0f} ms)")
        if not same_items:
            raise SystemExit("sanity check failed: engine does not reproduce the saved results")
    print(f"  check: 10 queries reproduce the saved top 5; largest score difference {worst:.1e}")


if __name__ == "__main__":
    main()
