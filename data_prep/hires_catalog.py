"""Swap the 60x80 catalog images for high-resolution versions of the SAME items.

Reads the full Kaggle dataset zip (paramaggarwal/fashion-product-images-dataset)
directly, without unzipping all 25 GB: for every item in data/catalog.csv it
pulls out <item_id>.jpg, shrinks it to at most --max_side pixels, and saves it to
data/images_hires/. The catalog keeps exactly the same items, only image_path changes.

The old catalog is kept as data/catalog_small.csv.

Run from the repo root:
    python -m data_prep.hires_catalog --zip data/raw_full/fashion-product-images-dataset.zip
"""
import argparse
import io
import os
import shutil
import zipfile

import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/raw_full/fashion-product-images-dataset.zip")
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--out_dir", default="data/images_hires")
    parser.add_argument("--max_side", type=int, default=512)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    backup = args.catalog.replace(".csv", "_small.csv")
    if not os.path.exists(backup):
        shutil.copy(args.catalog, backup)
        print(f"saved the old catalog as {backup}")

    os.makedirs(args.out_dir, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        # map "12345.jpg" -> its full path inside the zip (only files in an images/ folder)
        member_of = {}
        for name in z.namelist():
            if "/images/" in name or name.startswith("images/"):
                member_of[os.path.basename(name)] = name

        new_paths, missing = [], 0
        for i, row in enumerate(catalog.itertuples(), start=1):
            member = member_of.get(f"{row.item_id}.jpg")
            if member is None:
                new_paths.append(row.image_path)  # keep the small image if no big one exists
                missing += 1
                continue
            img = Image.open(io.BytesIO(z.read(member))).convert("RGB")
            img.thumbnail((args.max_side, args.max_side))  # shrink, keeping the aspect ratio
            path = os.path.join(args.out_dir, f"{row.item_id}.jpg")
            img.save(path, quality=92)
            new_paths.append(path)
            if i % 500 == 0:
                print(f"  {i}/{len(catalog)}")

    catalog["image_path"] = new_paths
    catalog.to_csv(args.catalog, index=False)
    print(f"wrote {args.catalog}: {len(catalog) - missing} high-res images, {missing} kept small")


if __name__ == "__main__":
    main()