"""D10a: per-category breakdown of one model's results.

Averages hide where the model fails. Here we split the saved per-query results
(results/perquery/<method>_<condition>.csv) by garment type and report, per type,
Recall@1 and (category) Precision@5 with 95% bootstrap intervals.

Two data sources:
  synthetic : our test set, 14 article types x 21 items, 4 photo conditions.
              The query_id starts with the item_id ("50664_phone_medium"), and
              data/catalog.csv gives the item's article_type.
  lookbench : 981 real street photos (dedup protocol, the main number in the report).
              data/lookbench/real_streetlook/queries.csv gives each query's category.
              Many LookBench categories have fewer than 10 queries, so we also report
              the 7 category groups fixed earlier for grouped Mondrian (GROUPS in
              evaluation/conformal_extra.py). Small categories are flagged, not hidden.

Sanity check: the per-category numbers, weighted by number of queries, must give back
the overall numbers already saved in results/summary.csv and lookbench_summary.csv
(e.g. LookBench dedup R@1 0.520, category P@5 0.725). The script stops if they don't.

Run from the repo root:
    python -m evaluation.per_category --method finetuned_random
"""
import argparse

import numpy as np
import pandas as pd

from evaluation.conformal_extra import to_group
from evaluation.stats import bootstrap_ci

SYNTHETIC = ["clean", "phone_mild", "phone_medium", "phone_strong"]
LOOKBENCH = "lookbench_real_streetlook_image_dedup"
SMALL_N = 20          # categories with fewer queries than this get small=True


def breakdown(perquery, labels, source, condition, level):
    """One row per label value: n, R@1, P@5 and their 95% bootstrap intervals."""
    rows = []
    for label in sorted(set(labels)):
        part = perquery[labels == label]
        r1_lo, r1_hi = bootstrap_ci(part["hit1"])
        p5_lo, p5_hi = bootstrap_ci(part["p5"])
        rows.append(dict(source=source, condition=condition, level=level, label=label,
                         n=len(part), small=len(part) < SMALL_N,
                         R1=part["hit1"].mean(), R1_lo=r1_lo, R1_hi=r1_hi,
                         P5=part["p5"].mean(), P5_lo=p5_lo, P5_hi=p5_hi))
    return rows


def check(rows, expected_r1, expected_p5, name):
    """Query-weighted mean of the per-category rows must equal the saved overall number."""
    table = pd.DataFrame(rows)
    r1 = np.average(table["R1"], weights=table["n"])
    p5 = np.average(table["P5"], weights=table["n"])
    ok = np.isclose(r1, expected_r1) and np.isclose(p5, expected_p5)
    print(f"  check {name}: R@1 {r1:.4f} vs saved {expected_r1:.4f}, "
          f"P@5 {p5:.4f} vs saved {expected_p5:.4f} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("sanity check failed, not saving")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="finetuned_random")
    parser.add_argument("--out", default="results/per_category.csv")
    args = parser.parse_args()

    rows = []

    # synthetic test set: per article type, for each photo condition
    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    type_of = dict(zip(catalog["item_id"], catalog["article_type"]))
    summary = pd.read_csv("results/summary.csv")
    for condition in SYNTHETIC:
        perquery = pd.read_csv(f"results/perquery/{args.method}_{condition}.csv")
        item_ids = perquery["query_id"].str.split("_").str[0]
        types = item_ids.map(type_of).to_numpy()
        new = breakdown(perquery, types, "synthetic", condition, "article_type")
        saved = summary[(summary["method"] == args.method) & (summary["condition"] == condition)]
        check(new, saved["Recall@1"].item(), saved["Precision@5"].item(), condition)
        rows += new

    # LookBench real photos (dedup): per category and per category group
    queries = pd.read_csv("data/lookbench/real_streetlook/queries.csv", dtype=str)
    perquery = pd.read_csv(f"results/perquery/{args.method}_{LOOKBENCH}.csv")
    perquery = perquery.merge(queries[["query_id", "category"]], on="query_id")
    categories = perquery["category"].to_numpy()
    lb = pd.read_csv("results/lookbench_summary.csv")
    saved = lb[(lb["method"] == args.method) & (lb["subset"] == "real_streetlook")
               & (lb["query_image"] == "image") & (lb["protocol"] == "dedup")]
    for level, labels in [("category", categories), ("group", to_group(categories))]:
        new = breakdown(perquery, labels, "lookbench", "real_dedup", level)
        check(new, saved["Recall@1"].item(), saved["CategoryP@5"].item(), f"lookbench {level}")
        rows += new

    table = pd.DataFrame(rows)
    table.insert(0, "method", args.method)
    table.to_csv(args.out, index=False)
    print(f"saved {len(table)} rows to {args.out}")

    # short printout: worst groups/types first
    pd.set_option("display.width", 120)
    show = ["label", "n", "R1", "R1_lo", "R1_hi", "P5", "P5_lo", "P5_hi"]
    for source, condition, level in [("lookbench", "real_dedup", "group"),
                                     ("synthetic", "phone_strong", "article_type")]:
        part = table[(table["source"] == source) & (table["condition"] == condition)
                     & (table["level"] == level)]
        print(f"\n{source} {condition} by {level} (sorted by P@5):")
        print(part.sort_values("P5")[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()