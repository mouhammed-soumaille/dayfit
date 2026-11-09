"""Build data/catalog.csv from the Kaggle "Fashion Product Images (Small)" dataset.

Steps:
  1. read styles.csv (the Kaggle label file)
  2. keep only the chosen article types, and only rows whose image exists
  3. sample up to --per_type images per article type (roughly balanced)
  4. remove near-duplicate images using a difference hash (dHash)
  5. write catalog.csv with columns: item_id, image_path, category, article_type, color

Run from the repo root:
    python -m data_prep.build_catalog
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

# ~14 common clothing/shoe types. Change freely, but then keep it fixed.
DEFAULT_TYPES = [
    "Tshirts", "Shirts", "Tops", "Kurtas", "Sweatshirts", "Jackets", "Dresses",
    "Jeans", "Trousers", "Shorts", "Track Pants",
    "Casual Shoes", "Sports Shoes", "Heels",
]


def dhash(image_path, hash_size=16):
    """Difference hash: shrink to 17x16 grayscale, compare each pixel with its
    right neighbour. Returns 256 booleans (same idea as imagehash.dhash).
    Re-saved or slightly resized copies of an image give the same hash."""
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(img, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def duplicate_key(image_path):
    """Two images count as duplicates if they have the same detailed dHash AND
    roughly the same average colour. The colour part matters: the hash is
    grayscale, so the same T-shirt cut in red and in blue would otherwise
    look like a duplicate even though they are different products."""
    shape = dhash(image_path).tobytes()
    mean_rgb = np.asarray(Image.open(image_path).convert("RGB")).reshape(-1, 3).mean(axis=0)
    colour = tuple((mean_rgb // 16).astype(int))  # round colour to coarse steps
    return shape, colour


def remove_near_duplicates(df):
    """Keep the first image of each group of duplicates."""
    seen = set()
    kept_rows = []
    for i, row in enumerate(df.itertuples()):
        key = duplicate_key(row.image_path)
        if key in seen:
            continue
        seen.add(key)
        kept_rows.append(i)
    return df.iloc[kept_rows].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", default="data/raw/styles.csv")
    parser.add_argument("--images", default="data/raw/images")
    parser.add_argument("--out", default="data/catalog.csv")
    parser.add_argument("--per_type", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # a few rows in styles.csv have extra commas in the product name -> skip them
    styles = pd.read_csv(args.styles, on_bad_lines="skip")
    styles = styles[styles["articleType"].isin(DEFAULT_TYPES)].copy()

    styles["item_id"] = styles["id"].astype(str)
    styles["image_path"] = [os.path.join(args.images, f"{i}.jpg") for i in styles["item_id"]]
    styles = styles[[os.path.exists(p) for p in styles["image_path"]]]

    # balanced sample: up to per_type rows from each article type
    # (shuffle everything, then take the first per_type rows of each type)
    shuffled = styles.sample(frac=1, random_state=args.seed)
    sampled = shuffled.groupby("articleType").head(args.per_type).reset_index(drop=True)
    print(f"sampled {len(sampled)} images before duplicate removal")

    catalog = remove_near_duplicates(sampled)
    print(f"{len(catalog)} images after removing {len(sampled) - len(catalog)} near-duplicates")

    catalog = catalog.rename(columns={
        "subCategory": "category",      # Topwear / Bottomwear / Shoes ...
        "articleType": "article_type",  # Tshirts / Jeans / Sneakers ...
        "baseColour": "color",
    })[["item_id", "image_path", "category", "article_type", "color"]]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    catalog.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(catalog["article_type"].value_counts().to_string())


if __name__ == "__main__":
    main()