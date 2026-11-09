"""Pick ~300 catalog items as evaluation queries and make a phone version of each.

Writes data/queries.csv with two rows per chosen item:
  - condition "clean": the original catalog image
  - condition "phone": a synthetic phone-style version saved under data/queries/phone/

How these are scored (see evaluation/run_eval.py):
  - Precision@5: the source item is removed from the results, then we count how
    many of the top 5 have the same article_type. (Proposal's metric.)
  - Recall@1/5: is the source item itself in the top 1/5? Only meaningful for
    "phone" queries on Kaggle data (a clean query is literally the same image).

Run from the repo root:
    python -m data_prep.make_queries
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from data_prep.phone_aug import make_phone_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--out", default="data/queries.csv")
    parser.add_argument("--phone_dir", default="data/queries/phone")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--severity", default="medium", choices=["mild", "medium", "strong"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    rng = np.random.default_rng(args.seed)

    # same number of queries from every article type
    n_types = catalog["article_type"].nunique()
    per_type = max(1, args.n // n_types)
    shuffled = catalog.sample(frac=1, random_state=args.seed)
    chosen = shuffled.groupby("article_type").head(per_type).reset_index(drop=True)

    os.makedirs(args.phone_dir, exist_ok=True)
    rows = []
    for item in chosen.itertuples():
        labels = dict(category=item.category, article_type=item.article_type, color=item.color)

        rows.append(dict(query_id=f"{item.item_id}_clean", image_path=item.image_path,
                         source_item_id=item.item_id, condition="clean", **labels))

        phone = make_phone_version(Image.open(item.image_path), rng, args.severity)
        phone_path = os.path.join(args.phone_dir, f"{item.item_id}.jpg")
        phone.save(phone_path, quality=95)
        rows.append(dict(query_id=f"{item.item_id}_phone", image_path=phone_path,
                         source_item_id=item.item_id, condition="phone", **labels))

    queries = pd.DataFrame(rows)
    queries.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(chosen)} items x 2 conditions = {len(queries)} queries")


if __name__ == "__main__":
    main()
