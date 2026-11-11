"""C1: "how shoppers search today" baseline: filter the catalogue by menu labels, no image model.

The shopper picks the garment type from a menu (and, on the synthetic data, also the colour).
The shop then shows all matching items in an order that has nothing to do with the photo,
which we model as a uniformly random order. If r of the m items shown are pictures of the
shopper's item, everything can be computed exactly (no simulation):
    P(a picture of the item is in the top k)  = 1 - C(m-r, k) / C(m, k)
    expected position of the first picture    = (m+1) / (r+1)
Precision@5 (same type) is 1 by construction for a type filter, so we compare on exact-item
Recall@1 / Recall@5 and on how far the shopper has to scroll.
Optimistic for menus: it assumes the shopper can name the type and colour, which our user
story says they cannot. LookBench has no colour labels, so only the type filter runs there,
counted in distinct pictures (duplicate gallery entries merged, Finding 15).

Run from the repo root:
    python -m evaluation.filter_baseline
"""
import hashlib
from math import comb

import numpy as np
import pandas as pd


def hit_at_k(m, r, k):
    """Chance that at least one of the r relevant items is in the top k of a random order of m."""
    if r == 0:
        return 0.0                          # the item was filtered out: it can never be found
    if m <= k:
        return 1.0                          # everything shown fits on the first screen
    return 1 - comb(m - r, k) / comb(m, k)


def first_position(m, r):
    """Expected position of the first relevant item in a random order (nan if filtered out)."""
    return (m + 1) / (r + 1) if r > 0 else np.nan


def summarize(data, filter_name, shown, relevant):
    """One results row from the list of (items shown m, relevant items r), one pair per query."""
    rows = pd.DataFrame(dict(m=shown, r=relevant))
    rows["hit1"] = [hit_at_k(m, r, 1) for m, r in zip(rows.m, rows.r)]
    rows["hit5"] = [hit_at_k(m, r, 5) for m, r in zip(rows.m, rows.r)]
    rows["scroll"] = [first_position(m, r) for m, r in zip(rows.m, rows.r)]
    return dict(data=data, filter=filter_name, n_queries=len(rows),
                filtered_out=int((rows.r == 0).sum()),
                median_shown=rows.m.median(),
                recall1=rows.hit1.mean(), recall5=rows.hit5.mean(),
                mean_scroll=rows.scroll.mean(), median_scroll=rows.scroll.median())


def synthetic_rows():
    """Our catalogue: one row per test item (a menu filter ignores the photo condition)."""
    catalog = pd.read_csv("data/catalog.csv", dtype=str).fillna({"color": "unknown"})
    queries = pd.read_csv("data/queries.csv", dtype=str).fillna({"color": "unknown"})
    items = queries[queries["source_item_id"].isin(catalog["item_id"])]
    items = items.drop_duplicates("source_item_id")

    type_count = catalog["article_type"].value_counts()
    type_color_count = catalog.groupby(["article_type", "color"]).size()
    shown_type = [type_count[a] for a in items["article_type"]]
    shown_tc = [type_color_count[(a, c)] for a, c in zip(items["article_type"], items["color"])]
    one = [1] * len(items)                  # each test item appears once in the catalogue
    return [summarize("synthetic", "type", shown_type, one),
            summarize("synthetic", "type + colour", shown_tc, one)]


def lookbench_rows():
    """LookBench real street photos: filter the gallery by the query's category.

    Counted in distinct PICTURES (md5 of the file), not gallery entries: 633 outfit photos are
    listed once per item they show (Finding 15), so counting entries would show duplicates twice.
    """
    from evaluation.run_lookbench import prepare
    gallery, queries = prepare("real_streetlook")
    queries = queries[queries["label"].isin(gallery["label"])]
    gallery = gallery.assign(picture=[hashlib.md5(open(p, "rb").read()).hexdigest()
                                      for p in gallery["image_path"]])

    pictures_in_cat = gallery.groupby("category")["picture"].nunique()
    pictures_of_item = gallery.groupby(["label", "category"])["picture"].nunique()
    shown = [pictures_in_cat.get(c, 0) for c in queries["category"]]
    relevant = [pictures_of_item.get((lab, c), 0) for lab, c in zip(queries["label"], queries["category"])]
    return [summarize("lookbench_real", "type", shown, relevant)]


def main():
    table = pd.DataFrame(synthetic_rows() + lookbench_rows())
    table.to_csv("results/filter_baseline.csv", index=False)
    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()