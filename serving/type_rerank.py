"""Type-aware ranking for the APP catalogue: does a small bonus for "the type the photo looks like"
fix cases like a flat photo of long jeans getting kids' shorts first, without hurting the search?

Why: the app catalogue photographs adult jeans on a person, and only kids' denim lies flat on white.
For a flat photo of jeans the image similarity prefers flat kids' shorts (same photo style) over
worn adult jeans (same garment). The pretrained zero-shot type predictor (serving/badges.py, app
vocabulary) does not care about photo style: it gives P(type) for each of the 72 app types.

New score of a catalogue item = similarity + w * P(the item's type | the photo).
The similarity shown in the app stays the plain similarity; "not sure" and the shortlist also stay
on the plain similarity (they were calibrated on it). Only the ORDER of the results changes.

Plan (fixed BEFORE running):
  queries : synthetic phone photos (medium, strong) of the 6,197 evaluation items; every one of them
            is also in the app catalogue, so the exact item is known
  search  : the full app catalogue (33,141 items), finetuned_random
  metrics : R@1 (exact item first) and type P@5 (share of the top 5 with the query's article type)
  w       : 0 (= today's app), 0.02, 0.05, 0.1, 0.2
  rule    : keep the w with the highest mean type P@5 (medium and strong) among those whose R@1 is at
            most 1 percentage point below w = 0 on BOTH medium and strong. Saved to serving/rerank.json.
  Your own photos in my_photos/ are only printed (top 5 types before and after); they do not choose w.

Sanity check: with w = 0 the top 5 equal the engine's top 5 (today's app) for the first 20 queries.

Run from the repo root (embeds ~900 photos twice, a few minutes):
    python -m serving.type_rerank
"""
import glob
import json

import numpy as np
import pandas as pd
from PIL import Image

from serving.badges import TypePredictor
from serving.engine import SearchEngine

WEIGHTS = [0.0, 0.02, 0.05, 0.1, 0.2]
MAX_R1_DROP = 0.01
OUT_CSV = "results/type_rerank.csv"
OUT_JSON = "serving/rerank.json"


def top5(similarity, bonus, w):
    """Indices of the 5 best catalogue items for the score similarity + w * bonus."""
    score = similarity + w * bonus
    best = np.argpartition(-score, 5)[:5]
    return best[np.argsort(-score[best])]


def main():
    engine = SearchEngine(catalogue="app")
    predictor = TypePredictor("app")
    retriever = engine.retriever
    item_types = np.array([engine.item.loc[i, "article_type"] for i in retriever.item_ids])
    type_index = np.array([predictor.types.index(t) for t in item_types])   # item -> position in P(type)
    item_ids = np.array(retriever.item_ids)

    def scores(image):
        similarity = retriever.features @ retriever.embed_query(image)
        bonus = predictor.analyse(image)["type_probs"][type_index]          # P(item's type | photo)
        return similarity, bonus

    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    queries = queries[queries["condition"].isin(["phone_medium", "phone_strong"])]
    rows = []
    for n, q in enumerate(queries.itertuples()):
        image = Image.open(q.image_path).convert("RGB")
        similarity, bonus = scores(image)
        if n < 20:                                                          # sanity check
            found, _ = engine.search(image, k=5)
            assert list(item_ids[top5(similarity, bonus, 0.0)]) == [f["item_id"] for f in found], \
                "w = 0 does not reproduce the app's top 5"
        for w in WEIGHTS:
            best = top5(similarity, bonus, w)
            rows.append(dict(condition=q.condition, w=w, r1=float(item_ids[best[0]] == q.source_item_id),
                             type_p5=float(np.mean(item_types[best] == q.article_type))))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(queries)} photos")
    print("  check: with w = 0 the top 5 equal the app's top 5 (20 photos)")

    d = pd.DataFrame(rows)
    d.to_csv(OUT_CSV, index=False)
    table = d.groupby(["condition", "w"])[["r1", "type_p5"]].mean().unstack("condition")
    print(table.round(3).to_string())

    base = table.loc[0.0]
    allowed = [w for w in WEIGHTS
               if all(table.loc[w, ("r1", c)] >= base[("r1", c)] - MAX_R1_DROP for c in ["phone_medium", "phone_strong"])]
    chosen = max(allowed, key=lambda w: table.loc[w, "type_p5"].mean())
    print(f"  chosen w (rule fixed before running): {chosen}")
    with open(OUT_JSON, "w") as f:
        json.dump(dict(w=chosen, rule=f"highest mean type P@5 with R@1 drop <= {MAX_R1_DROP} on medium and strong",
                       table={f"{m}_{c}_w{w}": float(table.loc[w, (m, c)]) for w in WEIGHTS
                              for m in ["r1", "type_p5"] for c in ["phone_medium", "phone_strong"]}), f, indent=2)
    print(f"saved {OUT_JSON} and {OUT_CSV}")

    for path in sorted(glob.glob("my_photos/*.jpg")):                       # your photos: shown only
        similarity, bonus = scores(Image.open(path).convert("RGB"))
        before = ", ".join(item_types[top5(similarity, bonus, 0.0)])
        after = ", ".join(item_types[top5(similarity, bonus, chosen)])
        print(f"\n  {path}\n    today : {before}\n    w={chosen}: {after}")


if __name__ == "__main__":
    main()
