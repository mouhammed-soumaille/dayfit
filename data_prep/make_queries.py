"""Pick ~300 catalog items as evaluation queries and make phone versions of each.

Writes data/queries.csv with one "clean" row plus one row per phone severity
for every chosen item:
  - condition "clean":        the original catalog image
  - condition "phone_mild",
    "phone_medium",
    "phone_strong":           synthetic phone-style versions, saved under
                              data/queries/phone_<severity>/

Every severity of the same item uses the SAME random draws (same background
colours, same rotation direction, same crop position...), only scaled up or
down. So when accuracy drops from mild to strong, it is because of the
severity, not because a different random background was picked. That is what
makes the robustness curve (accuracy vs severity) a fair comparison.

How these are scored (see evaluation/run_eval.py):
  - Precision@5: the source item is removed from the results, then we count how
    many of the top 5 have the same article_type. (Proposal's metric.)
  - Recall@1/5: is the source item itself in the top 1/5? Only meaningful for
    phone queries on Kaggle data (a clean query is literally the same image,
    so its Recall is always 1.0 and should not be reported).

Run from the repo root:
    python -m data_prep.make_queries
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from data_prep.phone_aug import SEVERITY, make_phone_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--out", default="data/queries.csv")
    parser.add_argument("--phone_dir", default="data/queries")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--severities", default="mild,medium,strong",
                        help="comma-separated, from: " + ", ".join(SEVERITY))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    severities = args.severities.split(",")

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})

    # same number of queries from every article type
    n_types = catalog["article_type"].nunique()
    per_type = max(1, args.n // n_types)
    shuffled = catalog.sample(frac=1, random_state=args.seed)
    chosen = shuffled.groupby("article_type").head(per_type).reset_index(drop=True)

    for severity in severities:
        os.makedirs(os.path.join(args.phone_dir, f"phone_{severity}"), exist_ok=True)

    rows = []
    for item in chosen.itertuples():
        labels = dict(category=item.category, article_type=item.article_type, color=item.color)

        rows.append(dict(query_id=f"{item.item_id}_clean", image_path=item.image_path,
                         source_item_id=item.item_id, condition="clean", **labels))

        source = Image.open(item.image_path)
        for severity in severities:
            # a fresh generator seeded by (seed, item_id): the same item always
            # gets the same random draws, whatever the severity
            rng = np.random.default_rng([args.seed, int(item.item_id)])
            phone = make_phone_version(source, rng, severity)
            phone_path = os.path.join(args.phone_dir, f"phone_{severity}", f"{item.item_id}.jpg")
            phone.save(phone_path, quality=95)
            rows.append(dict(query_id=f"{item.item_id}_phone_{severity}", image_path=phone_path,
                             source_item_id=item.item_id, condition=f"phone_{severity}", **labels))

    queries = pd.DataFrame(rows)
    queries.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(chosen)} items x {1 + len(severities)} conditions "
          f"= {len(queries)} queries")
    print(queries["condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
