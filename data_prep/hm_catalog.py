"""Build a modern DEMO catalogue from the H&M Kaggle data (academic / non-commercial use).

Why: our evaluation catalogue (Fashion Product Images, 14 types) has no skirts, bags, jumpsuits and
few modern cuts, so the app can only give loose suggestions for them (Finding 25). This catalogue is
for the app only; every reported evaluation number stays on the original catalogue.

Selection rules (fixed before downloading any image):
  - adult sections only: Ladieswear, Divided, Menswear, Sport (no Baby/Children)
  - the product types in TYPES (clothes, shoes, bags, a few accessories; no underwear, swimwear,
    nightwear, jewellery or cosmetics)
  - at most PER_TYPE items per type: first ONE colour variant per product (more different cuts),
    then further variants if the type still has room; random with a fixed seed
  - images are downloaded one file at a time (in parallel threads) and resized to MAX_SIDE pixels
Output: data/hm/catalog.csv with the same columns as data/catalog.csv
(item_id, image_path, category, article_type, color) + description; images in data/hm/images/.
Safe to re-run: images already downloaded are skipped.

Sanity check: every row of the output has an image file that opens, and the per-type counts are
printed against the cap.

Two ways to get the images:
  --from_zip PATH  (recommended) read them from the full competition archive, downloaded once;
                   only the chosen images are read out of it, nothing else is unpacked
  default          one Kaggle request per image; Kaggle blocked this after ~500 requests (Nov 18)

Run from the repo root (needs the Kaggle rules accepted and `kaggle` set up):
    python -m data_prep.hm_catalog --dry_run      # only show what would be downloaded
    python -m data_prep.hm_catalog --from_zip data/hm/h-and-m-personalized-fashion-recommendations.zip
"""
import argparse
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

COMPETITION = "h-and-m-personalized-fashion-recommendations"
SECTIONS = ["Ladieswear", "Divided", "Menswear", "Sport"]
TYPES = ["Trousers", "Dress", "Sweater", "T-shirt", "Top", "Blouse", "Jacket", "Shorts", "Shirt",
         "Vest top", "Skirt", "Hoodie", "Cardigan", "Jumpsuit/Playsuit", "Blazer", "Coat", "Polo shirt",
         "Leggings/Tights", "Sneakers", "Boots", "Sandals", "Ballerinas", "Bag", "Hat/beanie", "Scarf",
         "Belt", "Sunglasses"]
PER_TYPE = 1000
MAX_SIDE = 512
OUT_DIR = "data/hm"


def choose(articles, per_type, seed=0):
    """Apply the selection rules; returns the chosen rows."""
    a = articles[articles["index_group_name"].isin(SECTIONS) & articles["product_type_name"].isin(TYPES)]
    a = a.sample(frac=1, random_state=seed)                       # random order, fixed seed
    a["variant_rank"] = a.groupby("product_code").cumcount()     # 0 = first colour variant of a product
    a = a.sort_values("variant_rank", kind="stable")            # all first variants before second ones
    return a.groupby("product_type_name", group_keys=False).head(per_type)


def image_name(article_id):
    """H&M stores images as images/<first 3 digits>/<article_id>.jpg."""
    return f"images/{article_id[:3]}/{article_id}.jpg"


def download(api, article_id):
    """Download one image, resize it, save it. Returns the saved path, or None if it is missing."""
    path = os.path.join(OUT_DIR, "images", f"{article_id}.jpg")
    if os.path.exists(path):
        return path
    tmp = os.path.join(OUT_DIR, "tmp", article_id)
    os.makedirs(tmp, exist_ok=True)
    try:
        api.competition_download_file(COMPETITION, image_name(article_id), path=tmp, quiet=True)
    except Exception:
        return None                                              # some articles have no image
    files = [os.path.join(d, f) for d, _, names in os.walk(tmp) for f in names]   # Kaggle keeps sub-folders
    if not files:
        return None
    raw = files[0]
    with open(raw, "rb") as f:
        data = f.read()
    if raw.endswith(".zip"):                                     # Kaggle may zip single files
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(z.namelist()[0])
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(path, quality=90)
    os.remove(raw)
    return path


def save_resized(data, path):
    """Image bytes -> resized JPEG on disk."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(path, quality=90)
    return path


def extract_from_zip(zip_path, ids):
    """Read only the chosen images out of the big archive. Returns one path (or None) per id."""
    paths = []
    with zipfile.ZipFile(zip_path) as z:
        inside = set(z.namelist())
        for n, article_id in enumerate(ids, start=1):
            path = os.path.join(OUT_DIR, "images", f"{article_id}.jpg")
            name = image_name(article_id)
            if os.path.exists(path):
                paths.append(path)
            elif name in inside:
                paths.append(save_resized(z.read(name), path))
            else:
                paths.append(None)                               # this article really has no image
            print(f"  {n}/{len(ids)}", end="\r")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_type", type=int, default=PER_TYPE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--from_zip", default=None, help="path of the full competition archive")
    args = parser.parse_args()

    with zipfile.ZipFile(os.path.join(OUT_DIR, "articles.csv.zip")) as z:
        articles = pd.read_csv(z.open("articles.csv"), dtype={"article_id": str})
    chosen = choose(articles, args.per_type)
    print(f"  chosen {len(chosen)} articles of {chosen['product_type_name'].nunique()} types "
          f"({chosen['product_code'].nunique()} different products)")
    print(chosen["product_type_name"].value_counts().to_string())
    if args.dry_run:
        return

    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    ids = list(chosen["article_id"])
    if args.from_zip:
        paths = extract_from_zip(args.from_zip, ids)
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            paths = []
            for n, p in enumerate(pool.map(lambda i: download(api, i), ids), start=1):
                paths.append(p)
                print(f"  {n}/{len(ids)}", end="\r")
    chosen["image_path"] = paths
    missing = chosen["image_path"].isna().sum()
    chosen = chosen.dropna(subset=["image_path"])

    for p in chosen["image_path"]:                               # check: every image opens
        Image.open(p).verify()
    catalog = pd.DataFrame(dict(item_id="hm" + chosen["article_id"], image_path=chosen["image_path"],
                                category=chosen["product_group_name"], article_type=chosen["product_type_name"],
                                color=chosen["colour_group_name"], description=chosen["detail_desc"]))
    catalog.to_csv(os.path.join(OUT_DIR, "catalog.csv"), index=False)
    print(f"\n  check: all {len(catalog)} images open; {missing} articles had no image")
    print(f"saved {OUT_DIR}/catalog.csv")


if __name__ == "__main__":
    main()
