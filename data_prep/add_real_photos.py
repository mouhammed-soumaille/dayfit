"""Add real phone photos of your own clothes to data/queries.csv (condition "real").

Put the photos in folders named after the article type, using the exact names
from the catalog, e.g.:
    data/real_photos/Tshirts/IMG_0012.jpg
    data/real_photos/Jeans/IMG_0020.jpg
    data/real_photos/Casual Shoes/IMG_0031.jpg

Your clothes are not in the catalog, so exact-item Recall cannot be measured
for these; run_eval leaves Recall empty and reports Precision@5 (same
article type) only. They are also the best demo and sanity check.

iPhone HEIC photos work if `pillow-heif` is installed; otherwise export as JPEG.

Run from the repo root (safe to re-run; old "real" rows are replaced):
    python -m data_prep.add_real_photos
"""
import argparse
import os

import pandas as pd
from PIL import Image, ImageOps

try:  # lets PIL open iPhone .heic files
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".webp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", default="data/real_photos")
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--out_dir", default="data/queries/real")
    parser.add_argument("--max_side", type=int, default=1024)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    # article_type -> its category, e.g. "Jeans" -> "Bottomwear"
    category_of = catalog.groupby("article_type")["category"].first().to_dict()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for article_type in sorted(os.listdir(args.photos)):
        folder = os.path.join(args.photos, article_type)
        if not os.path.isdir(folder):
            continue
        if article_type not in category_of:
            print(f"skipping folder '{article_type}': not an article type in the catalog "
                  f"(choose from: {', '.join(sorted(category_of))})")
            continue
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(EXTENSIONS):
                continue
            img = Image.open(os.path.join(folder, filename))
            img = ImageOps.exif_transpose(img).convert("RGB")  # respect the phone's rotation flag
            img.thumbnail((args.max_side, args.max_side))
            query_id = f"real_{article_type.replace(' ', '')}_{os.path.splitext(filename)[0]}"
            path = os.path.join(args.out_dir, f"{query_id}.jpg")
            img.save(path, quality=92)
            rows.append(dict(query_id=query_id, image_path=path, source_item_id="",
                             condition="real", category=category_of[article_type],
                             article_type=article_type, color=""))

    queries = pd.read_csv(args.queries, dtype={"source_item_id": str})
    queries = queries[queries["condition"] != "real"]
    queries = pd.concat([queries, pd.DataFrame(rows)], ignore_index=True)
    queries.to_csv(args.queries, index=False)
    print(f"added {len(rows)} real photos to {args.queries}")
    if rows:
        print(pd.DataFrame(rows)["article_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
