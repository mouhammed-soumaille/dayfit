"""Group-level Precision@5 on LookBench (fix for overlapping category names).

LookBench's category P@5 needs the result's category STRING to equal the query's. But the
category names overlap: bag / handbag / clutch bag, skirt / pleated skirt / denim skirt,
shoe / sneaker, shirt / t-shirt ... so a pleated skirt returned for a "skirt" query counts
as a wrong category (seen in the manual check, D8b). That makes category P@5 too strict.

Group P@5: a result counts if it belongs to the same GROUP as the query, using the 7 groups
of evaluation/conformal_extra.py (GROUPS), which were fixed for grouped Mondrian (Finding 12)
BEFORE any grouped result was seen, so they were not chosen to make this metric look good.
Groups are coarse (e.g. a cardigan counts for a blouse query), so group P@5 is too lenient:
exact P@5 and group P@5 together bracket what a shopper would judge.

The gallery also uses 38 names that never occur as query categories (boots, jeans, sneaker
variants, ...; 109 entries): GALLERY_EXTRA maps them to their obvious group, fixed before
computing anything. Under exact P@5 all of these always count as wrong.

Same dedup protocol as the main numbers: the first 5 distinct pictures; a picture counts if
ANY of its gallery copies is in the query's group.

Sanity check: with exact category names instead of groups, the script must reproduce the
saved dedup category P@5 of every model, query by query.

Run from the repo root:
    python -m evaluation.group_p5
"""
import hashlib

import numpy as np
import pandas as pd

from evaluation.conformal_extra import GROUP_OF
from evaluation.stats import bootstrap_ci, paired_bootstrap

LB = "data/lookbench/real_streetlook"
METHODS = ["fashion_siglip", "ensemble_mv", "finetuned", "finetuned_random"]
# Gallery-only category names that are not in GROUPS (109 of 3,278 entries). Mapped by the obvious
# garment type BEFORE computing group P@5; "product label" is not a garment and never counts.
GALLERY_EXTRA = {
    "footwear": ["boot", "boots", "knee-high boot", "knee-high boots", "over-the-knee boots", "sock boot",
                 "shoes", "sandals", "gladiator sandals", "high-heeled pump", "high heel pump",
                 "high-heeled shoe", "high-heeled sandal", "high-heeled sandals", "red high-heeled sandals",
                 "white sneaker", "white sneakers", "white low-top sneaker", "white low-top sneakers",
                 "white casual sneaker", "nike air force 1 sneaker"],
    "bottoms": ["jeans", "denim shorts", "kate faux leather shorts - black", "brown satin skirt",
                "lindex skirt mary kram"],
    "tops": ["hoodie", "white t-shirt", "denim vest"],
    "outerwear": ["trench coat"],
    "bags": ["tote bag", "transparent tote bag", "duffel bag", "ysl crossbody bag", "saint laurent handbag"],
    "accessories": ["tights", "колготки 40den"],
    "not_a_garment": ["product label"],
}
GROUP_OF_GALLERY = {**GROUP_OF, **{c: g for g, cats in GALLERY_EXTRA.items() for c in cats}}
COMPARISONS = [("fashion_siglip", "finetuned_random"),
               ("finetuned", "finetuned_random"),
               ("ensemble_mv", "finetuned_random")]


def p5_per_query(ranks, queries, pic_values, query_value):
    """Per-query share of the top-5 pictures whose value set contains the query's value."""
    top = ranks[ranks["rank"] <= 5].merge(queries, on="query_id")
    top["ok"] = [query_value(q) in pic_values[p] for q, p in zip(top["category"], top["picture"])]
    return top.groupby("query_id")["ok"].mean()


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)[["query_id", "category"]]
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    assert gallery["group"].notna().all(), "a gallery category has no group"
    pic_cats = gallery.groupby("picture")["category"].apply(set)
    pic_groups = gallery.groupby("picture")["group"].apply(set)

    rows, per_method = [], {}
    for m in METHODS:
        ranks = pd.read_csv(f"results/rankings/{m}_lookbench_real_streetlook_image_dedup.csv")
        exact = p5_per_query(ranks, queries, pic_cats, lambda c: c)
        group = p5_per_query(ranks, queries, pic_groups, lambda c: GROUP_OF[c.lower()])

        saved = pd.read_csv(f"results/perquery/{m}_lookbench_real_streetlook_image_dedup.csv")
        saved = saved.set_index("query_id")["p5"].loc[exact.index]
        ok = np.allclose(exact, saved)
        print(f"  check {m}: exact-category P@5 reproduces the saved values -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("sanity check failed, not saving")

        per_method[m] = group
        row = {"method": m, "n_queries": len(group),
               "CategoryP@5": exact.mean(), "GroupP@5": group.mean()}
        row["GroupP@5_lo"], row["GroupP@5_hi"] = bootstrap_ci(group)
        rows.append(row)
        print(f"  {m:17s} exact P@5 {exact.mean():.3f}   group P@5 {group.mean():.3f} "
              f"[{row['GroupP@5_lo']:.3f}, {row['GroupP@5_hi']:.3f}]")

    tests = []
    for a, b in COMPARISONS:
        diff, lo, hi, p = paired_bootstrap(per_method[b], per_method[a])
        tests.append(dict(change=f"{a} -> {b}", metric="GroupP@5", difference=diff, lo=lo, hi=hi, p_value=p))
        print(f"  {a} -> {b}  GroupP@5: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")

    # per-group view for finetuned_random (where does the strict metric hurt most?)
    q = queries.set_index("query_id")
    ranks = pd.read_csv("results/rankings/finetuned_random_lookbench_real_streetlook_image_dedup.csv")
    exact = p5_per_query(ranks, queries, pic_cats, lambda c: c)
    view = pd.DataFrame({"group": q.loc[exact.index, "category"].str.lower().map(GROUP_OF),
                         "exact": exact, "group_p5": per_method["finetuned_random"].loc[exact.index]})
    print("\nfinetuned_random by group (exact vs group P@5):")
    print(view.groupby("group").agg(n=("exact", "size"), exact=("exact", "mean"),
                                    group_p5=("group_p5", "mean")).round(3).sort_values("exact").to_string())

    pd.DataFrame(rows).to_csv("results/group_p5.csv", index=False)
    pd.DataFrame(tests).to_csv("results/group_p5_tests.csv", index=False)
    print("\nsaved results/group_p5.csv and results/group_p5_tests.csv")


if __name__ == "__main__":
    main()