"""E1: the search engine behind the app.

Loads the recommended model (finetuned_random) and its precomputed catalogue embeddings
ONCE, then answers searches: photo in, top-k catalogue items out (with type, colour,
image path and similarity score). No photo is ever written anywhere: the image only
lives in memory while it is embedded.

This is a thin layer over the retriever used in all experiments (RETRIEVERS in
evaluation/run_eval.py), so the app searches exactly like the evaluated model.

Sanity check (run this file): for a few saved synthetic test queries, the engine must
return the same top 5 items as the saved evaluation results, with the same scores
(up to tiny floating-point differences).

Run from the repo root:
    python -m serving.engine
"""
import time

import pandas as pd
from PIL import Image

from evaluation.run_eval import RETRIEVERS

METHOD = "finetuned_random"
CATALOG = "data/catalog.csv"


class SearchEngine:
    def __init__(self, method=METHOD, catalog_path=CATALOG):
        self.catalog = pd.read_csv(catalog_path, dtype={"item_id": str})
        self.retriever = RETRIEVERS[method]()
        self.retriever.build_index(self.catalog)          # loads the saved embeddings
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
    engine = SearchEngine()
    saved = pd.read_csv(f"results/{METHOD}_phone_medium.csv", dtype={"item_id": str})
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str}).set_index("query_id")
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