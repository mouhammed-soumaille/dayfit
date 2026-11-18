"""A much bigger catalogue FOR THE APP from the full Fashion Product Images dataset (already on disk).

The evaluation catalogue (data/catalog.csv) uses 6,197 items of 14 types, balanced for fair tests.
The full Kaggle zip has 44,424 items. For the app, more items means a better chance that something
similar exists (shoes, bags, skirts, sandals, ...). Every evaluation number stays on data/catalog.csv.

Selection rules (fixed before building):
  - adult items only (gender Men, Women, Unisex; no Boys/Girls)
  - the wearable types in TYPES: clothes, shoes, bags, and accessories that are worn (sunglasses,
    belts, caps, scarves, ties, ...); no watches, wallets, jewellery, underwear, nightwear,
    swimwear or personal care
  - all 6,197 evaluation items are always included (same item ids, same images)
  - near-duplicate images are removed with the same rule as data_prep/build_catalog.py
    (same detailed dHash AND same coarse average colour)
  - images are read straight from the zip and resized to MAX_SIDE pixels
Output: data/app_catalog.csv (same columns as data/catalog.csv + source + gender) and data/app_images/.
(gender is kept so the app can hide the 267 kids' items that come in with the evaluation catalogue.)

Sanity checks: every evaluation item is in the app catalogue; every image opens; counts per type printed.

Run from the repo root:
    python -m data_prep.app_catalog --dry_run
    python -m data_prep.app_catalog
"""
import argparse
import io
import os
import zipfile

import pandas as pd
from PIL import Image

from data_prep.build_catalog import duplicate_key

ZIP = "data/raw_full/fashion-product-images-dataset.zip"
OUT_CSV = "data/app_catalog.csv"
OUT_DIR = "data/app_images"
MAX_SIDE = 512
GENDERS = ["Men", "Women", "Unisex"]
TYPES = [
    # clothes
    "Tshirts", "Shirts", "Kurtas", "Tops", "Jeans", "Shorts", "Trousers", "Dresses", "Sarees",
    "Track Pants", "Sweatshirts", "Sweaters", "Jackets", "Kurtis", "Tunics", "Leggings", "Capris",
    "Skirts", "Dupatta", "Kurta Sets", "Stoles", "Lounge Pants", "Camisoles", "Patiala", "Jeggings",
    "Lounge Shorts", "Salwar", "Churidar", "Tracksuits", "Rain Jacket", "Jumpsuit", "Waistcoat",
    "Rompers", "Blazers", "Shrug", "Nehru Jackets", "Clothing Set", "Salwar and Dupatta",
    "Lehenga Choli", "Suits", "Rain Trousers", "Lounge Tshirts",
    # shoes
    "Casual Shoes", "Sports Shoes", "Heels", "Flip Flops", "Sandals", "Formal Shoes", "Flats",
    "Sports Sandals", "Booties",
    # bags
    "Handbags", "Backpacks", "Clutches", "Duffel Bag", "Laptop Bag", "Messenger Bag", "Rucksacks",
    "Waist Pouch", "Trolley Bag",
    # accessories that are worn
    "Sunglasses", "Belts", "Caps", "Scarves", "Ties", "Mufflers", "Gloves", "Hat", "Headband",
    "Socks", "Stockings", "Tights",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    evaluation = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    with zipfile.ZipFile(ZIP) as z:
        styles = pd.read_csv(io.BytesIO(z.read("fashion-dataset/styles.csv")), on_bad_lines="skip")
        styles["item_id"] = styles["id"].astype(str)
        chosen = styles[styles["articleType"].isin(TYPES) & styles["gender"].isin(GENDERS)]
        extra = chosen[~chosen["item_id"].isin(evaluation["item_id"])]
        print(f"  {len(chosen)} items of {chosen['articleType'].nunique()} types chosen; "
              f"{len(chosen) - len(extra)} already in the evaluation catalogue, {len(extra)} new")
        if args.dry_run:
            print(chosen["articleType"].value_counts().to_string())
            return

        member_of = {os.path.basename(n): n for n in z.namelist()
                     if n.endswith(".jpg") and "/images/" in n}
        os.makedirs(OUT_DIR, exist_ok=True)
        rows, no_image = [], 0
        for n, item in enumerate(extra.itertuples(), start=1):
            member = member_of.get(f"{item.item_id}.jpg")
            if member is None:
                no_image += 1
                continue
            path = os.path.join(OUT_DIR, f"{item.item_id}.jpg")
            if not os.path.exists(path):
                image = Image.open(io.BytesIO(z.read(member))).convert("RGB")
                image.thumbnail((MAX_SIDE, MAX_SIDE))
                image.save(path, quality=90)
            rows.append(dict(item_id=item.item_id, image_path=path, category=item.subCategory,
                             article_type=item.articleType, color=item.baseColour, source="extra",
                             gender=item.gender))
            if n % 1000 == 0:
                print(f"  {n}/{len(extra)} extracted")

    # near-duplicates: the evaluation items are seen first, so they are always kept
    seen = {duplicate_key(p) for p in evaluation["image_path"]}
    kept, duplicates = [], 0
    for row in rows:
        key = duplicate_key(row["image_path"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(row)

    gender_of = dict(zip(styles["item_id"], styles["gender"]))
    evaluation = evaluation.assign(source="evaluation", gender=evaluation["item_id"].map(gender_of))
    app = pd.concat([evaluation, pd.DataFrame(kept)], ignore_index=True)
    assert set(evaluation["item_id"]) <= set(app["item_id"]), "an evaluation item is missing"
    for p in app["image_path"]:                                   # check: every image opens
        Image.open(p).verify()
    app.to_csv(OUT_CSV, index=False)
    print(f"  check: all {len(evaluation)} evaluation items included; all {len(app)} images open")
    print(f"  {no_image} chosen items had no image; {duplicates} near-duplicates removed")
    print(app["article_type"].value_counts().to_string())
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
