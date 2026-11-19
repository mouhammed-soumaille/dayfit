"""Add the H&M demo items (data/hm/catalog.csv, data_prep/hm_catalog.py) to the APP catalogue.

Why: the app catalogue (Fashion Product Images, 2011-12) photographs adult jeans only on a person and
has no wide-leg cuts (Findings 29-30). H&M product photos (2018-20) are mostly flat or on a plain
background and include modern cuts. App only: every evaluation number stays on data/catalog.csv.

Rules (fixed before running):
  - every H&M product type is mapped to an EXISTING app type (TYPE_MAP below), so the badge prompts,
    the out-of-catalogue threshold and the group bonus need no recalibration
  - gender from the H&M section: Ladieswear / Divided -> Women, Menswear -> Men, Sport -> Unisex
  - only the NEW images are embedded (fine-tuned model); they are appended to the existing app index,
    in the same order as the new catalogue rows, with the same first line (weights)
  - the old app files are kept as *_fpi copies, so the merge can be undone

Sanity checks: the old 33,141 rows and embeddings are unchanged; every H&M type has a mapping;
re-embedding 20 random H&M images gives the saved vectors (cosine > 0.999); the search engine then
loads the merged index WITHOUT re-embedding (the cache key matches).

Run from the repo root (Mac, .venv; embeds the H&M images on MPS, a few minutes):
    python -m data_prep.app_add_hm
"""
import os
import shutil

import numpy as np
import pandas as pd

from evaluation.run_eval import RETRIEVERS

APP_CSV = "data/app_catalog.csv"
INDEX = "models/index/finetuned_random_app.npy"
PATHS = "models/index/finetuned_random_app_paths.txt"
HM_CSV = "data/hm/catalog.csv"
HM_ARTICLES = "data/hm/articles.csv.zip"

# H&M product type -> existing app catalogue type (same garment group)
TYPE_MAP = {"Trousers": "Trousers", "Dress": "Dresses", "Sweater": "Sweaters", "T-shirt": "Tshirts",
            "Top": "Tops", "Blouse": "Tops", "Jacket": "Jackets", "Shorts": "Shorts", "Shirt": "Shirts",
            "Vest top": "Tops", "Skirt": "Skirts", "Hoodie": "Sweatshirts", "Cardigan": "Sweaters",
            "Jumpsuit/Playsuit": "Jumpsuit", "Blazer": "Blazers", "Coat": "Jackets", "Polo shirt": "Tshirts",
            "Leggings/Tights": "Leggings", "Sneakers": "Casual Shoes", "Boots": "Casual Shoes",
            "Sandals": "Sandals", "Ballerinas": "Flats", "Bag": "Handbags", "Hat/beanie": "Hat",
            "Scarf": "Scarves", "Belt": "Belts", "Sunglasses": "Sunglasses"}
GENDER = {"Ladieswear": "Women", "Divided": "Women", "Menswear": "Men", "Sport": "Unisex"}


def main():
    app = pd.read_csv(APP_CSV, dtype={"item_id": str})
    assert (app["source"] != "hm").all(), "H&M items are already in the app catalogue"
    old_features = np.load(INDEX)
    with open(PATHS) as f:
        old_key = f.read().split("\n")
    assert old_key[1:] == list(app["image_path"]), "the app index does not match the app catalogue"

    # 1. H&M rows in the app catalogue's columns
    hm = pd.read_csv(HM_CSV, dtype={"item_id": str})
    unmapped = set(hm["article_type"]) - set(TYPE_MAP)
    assert not unmapped, f"H&M types without a mapping: {unmapped}"
    articles = pd.read_csv(HM_ARTICLES, dtype={"article_id": str})
    section = dict(zip("hm" + articles["article_id"], articles["index_group_name"]))
    hm = hm.assign(hm_type=hm["article_type"], article_type=hm["article_type"].map(TYPE_MAP),
                   source="hm", gender=hm["item_id"].map(section).map(GENDER))
    assert hm["gender"].notna().all(), "an H&M item has no section"
    hm = hm[[c for c in app.columns] + ["hm_type", "description"]]
    print(f"adding {len(hm)} H&M items; per app type:")
    print(hm["article_type"].value_counts().to_string())

    # 2. embed ONLY the new images with the fine-tuned model
    model = RETRIEVERS["finetuned_random"]()
    new_features = model.embed_catalog(list(hm["image_path"]))
    assert old_key[0] == model.weights_tag, "the checkpoint changed since the app index was made"

    # check: re-embedding 20 random H&M images gives the same vectors
    pick = np.random.default_rng(0).choice(len(hm), size=min(20, len(hm)), replace=False)
    again = model.embed_catalog([hm["image_path"].iloc[i] for i in pick])
    agree = (again * new_features[pick]).sum(axis=1).min()
    print(f"check: re-embedded 20 images, lowest cosine with the saved ones {agree:.4f}")
    assert agree > 0.999

    # 3. keep the old files, then write the merged catalogue, index and key (same order everywhere)
    for path in [APP_CSV, INDEX, PATHS]:
        stem, ext = os.path.splitext(path)
        shutil.copy(path, f"{stem}_fpi{ext}")
    merged = pd.concat([app, hm], ignore_index=True)
    merged.to_csv(APP_CSV, index=False)
    np.save(INDEX, np.concatenate([old_features, new_features]))
    with open(PATHS, "w") as f:
        f.write("\n".join(old_key + list(hm["image_path"])))

    # check: the old part is unchanged, and the engine loads the merged index without re-embedding
    assert np.array_equal(np.load(INDEX)[:len(app)], old_features)
    from serving.engine import SearchEngine
    engine = SearchEngine(catalogue="app")
    assert len(engine.retriever.features) == len(merged) == len(engine.item)
    print(f"done: app catalogue {len(app)} -> {len(merged)} items; old files kept as *_fpi")


if __name__ == "__main__":
    main()
