"""D10b: failure gallery for one model (default finetuned_random).

Each row = one query photo followed by its top-5 results. Border colour of a result:
  green  = the exact item (or, on LookBench, a picture that shows it)
  orange = right category, different item
  red    = wrong category
Three cases, chosen from Finding 16 (per-category breakdown):
  real_dresses     LookBench dress/jumpsuit queries with at least 3 wrong-category results (P@5 <= 0.4)
  real_outerwear   LookBench coat/jacket/cape queries whose top-1 is not the exact item
  synth_trousers   synthetic Trousers queries, strong phone photos, top-1 not the exact item
Rows are drawn at RANDOM (fixed seed) from all failures of that case, not hand-picked;
the figure title says how many failures there were.

Inputs (no model needed):
  results/rankings/<method>_lookbench_real_streetlook_image_dedup.csv  (top-5 distinct pictures)
  results/<method>_phone_strong.csv                                   (synthetic top-5)
  results/perquery/...                                                (saved hit1 / p5 per query)

Sanity check: before drawing, the hit@1 and P@5 recomputed from the rankings must equal
the saved per-query values for every query; the script stops otherwise. (Synthetic P@5
excludes the exact item and may use rank 6, which is not saved: see check().)

Run from the repo root:
    python -m evaluation.failure_gallery --method finetuned_random
"""
import argparse
import hashlib

import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from evaluation.conformal_extra import GROUP_OF

LB = "data/lookbench/real_streetlook"
N_ROWS = 6
COLOURS = {"exact": "tab:green", "category": "tab:orange", "wrong": "tab:red"}


def lookbench_rows(method):
    """One row per (query, rank) with the result's image, colour class and score."""
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    labels = gallery.groupby("picture")["label"].apply(set)        # a picture can show several items
    cats = gallery.groupby("picture")["category"].apply(set)
    path_of = dict(zip(gallery["item_id"], gallery["image_path"]))

    ranks = pd.read_csv(f"results/rankings/{method}_lookbench_real_streetlook_image_dedup.csv",
                        dtype={"item_id": str})
    ranks = ranks.merge(queries.rename(columns={"image": "query_path"})
                        [["query_id", "query_path", "label", "category"]], on="query_id")
    ranks["result_path"] = ranks["item_id"].map(path_of)
    ranks["exact"] = [lab in labels[p] for lab, p in zip(ranks["label"], ranks["picture"])]
    ranks["same_cat"] = [c in cats[p] for c, p in zip(ranks["category"], ranks["picture"])]
    ranks["group"] = ranks["category"].str.lower().map(GROUP_OF)
    saved = pd.read_csv(f"results/perquery/{method}_lookbench_real_streetlook_image_dedup.csv")
    return ranks, saved


def synthetic_rows(method, condition="phone_strong"):
    """Same columns as lookbench_rows, for our synthetic test set (category = article type)."""
    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    ranks = pd.read_csv(f"results/{method}_{condition}.csv", dtype={"item_id": str})
    ranks = ranks.merge(queries.rename(columns={"image_path": "query_path"})
                        [["query_id", "query_path", "source_item_id", "article_type"]], on="query_id")
    type_of = dict(zip(catalog["item_id"], catalog["article_type"]))
    path_of = dict(zip(catalog["item_id"], catalog["image_path"]))
    ranks["result_path"] = ranks["item_id"].map(path_of)
    ranks["exact"] = ranks["item_id"] == ranks["source_item_id"]
    ranks["same_cat"] = ranks["item_id"].map(type_of) == ranks["article_type"]
    ranks["category"] = ranks["article_type"]
    ranks["group"] = ranks["article_type"]           # no groups needed here: one group per type
    saved = pd.read_csv(f"results/perquery/{method}_{condition}.csv")
    return ranks, saved


def check(ranks, saved, name, p5_excludes_exact=False):
    """hit@1 and P@5 recomputed from the rankings must equal the saved per-query values.

    Synthetic P@5 (run_eval) removes the exact item and scores the 5 OTHER results, so it
    sometimes uses rank 6, which is not saved. For those queries (exact item in our top 5)
    the saved P@5 must be ours minus 0.2 (rank 6 is another type) or ours (rank 6 same type).
    LookBench P@5 (run_lookbench) counts the exact item, so it must match exactly.
    """
    ranks = ranks.sort_values(["query_id", "rank"])
    hit1 = ranks.groupby("query_id")["exact"].first().astype(float)
    p5 = ranks.groupby("query_id")["same_cat"].mean()
    found = ranks.groupby("query_id")["exact"].any()          # exact item somewhere in our top 5
    saved = saved.set_index("query_id").loc[hit1.index]

    ok_hit1 = np.allclose(hit1, saved["hit1"])
    if p5_excludes_exact:
        exact_ok = np.isclose(p5, saved["p5"])
        shifted_ok = np.isclose(p5 - 0.2, saved["p5"])
        ok_p5 = (exact_ok[~found]).all() and (exact_ok | shifted_ok)[found].all()
    else:
        ok_p5 = np.allclose(p5, saved["p5"])
    ok = ok_hit1 and ok_p5
    print(f"  check {name}: {len(hit1)} queries, hit@1 {'OK' if ok_hit1 else 'MISMATCH'}, "
          f"P@5 {'OK' if ok_p5 else 'MISMATCH'}")
    if not ok:
        raise SystemExit("sanity check failed, not drawing")


def per_query(ranks):
    """One row per query: its group, category, hit@1 and P@5."""
    ranks = ranks.sort_values(["query_id", "rank"])
    return ranks.groupby("query_id").agg(group=("group", "first"),
                                         category=("category", "first"),
                                         hit1=("exact", "first"),   # sorted, so first = rank 1
                                         p5=("same_cat", "mean"))


def draw(ranks, query_ids, title, out):
    """Grid: one row per query, column 0 = query photo, columns 1-5 = results."""
    fig, axes = plt.subplots(len(query_ids), 6, figsize=(12, 2.2 * len(query_ids)))
    for r, qid in enumerate(query_ids):
        rows = ranks[ranks["query_id"] == qid].sort_values("rank")
        first = rows.iloc[0]
        axes[r, 0].imshow(Image.open(first["query_path"]).convert("RGB"))
        axes[r, 0].set_title(f"query: {first['category']}", fontsize=8)
        for c, row in enumerate(rows.itertuples(), start=1):
            kind = "exact" if row.exact else ("category" if row.same_cat else "wrong")
            ax = axes[r, c]
            ax.imshow(Image.open(row.result_path).convert("RGB"))
            for side in ax.spines.values():                      # coloured border
                side.set_edgecolor(COLOURS[kind])
                side.set_linewidth(4)
            ax.set_title(f"{row.rank}. {kind} ({row.score:.2f})", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="finetuned_random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    real, real_saved = lookbench_rows(args.method)
    check(real, real_saved, "lookbench")
    synth, synth_saved = synthetic_rows(args.method)
    check(synth, synth_saved, "synthetic phone_strong", p5_excludes_exact=True)

    real_q = per_query(real)
    synth_q = per_query(synth)

    cases = [
        ("real_dresses", real, real_q[(real_q["group"] == "dresses") & (real_q["p5"] <= 0.4)],
         "LookBench dresses: at least 3 of 5 results are another category"),
        ("real_outerwear", real, real_q[(real_q["group"] == "outerwear") & ~real_q["hit1"]],
         "LookBench outerwear: top-1 is not the exact item"),
        ("synth_trousers", synth, synth_q[(synth_q["category"] == "Trousers") & ~synth_q["hit1"]],
         "Synthetic Trousers, strong phone photos: top-1 is not the exact item"),
    ]
    for name, ranks, failures, description in cases:
        chosen = sorted(rng.choice(failures.index, size=min(N_ROWS, len(failures)), replace=False))
        title = (f"{args.method} - {description}\n"
                 f"{len(chosen)} random rows (seed {args.seed}) out of {len(failures)} such queries; "
                 f"green = exact item, orange = right category, red = wrong category")
        draw(ranks, chosen, title, f"report/figures/fig10_failures_{name}.png")

def check(ranks, saved, name, p5_excludes_exact=False):
    """hit@1 and P@5 recomputed from the rankings must equal the saved per-query values.

    Synthetic P@5 (run_eval) removes the exact item and scores the 5 OTHER results, so it
    sometimes uses rank 6, which is not saved. For those queries (exact item in our top 5)
    the saved P@5 must be ours minus 0.2 (rank 6 is another type) or ours (rank 6 same type).
    LookBench P@5 (run_lookbench) counts the exact item, so it must match exactly.
    """
    ranks = ranks.sort_values(["query_id", "rank"])
    hit1 = ranks.groupby("query_id")["exact"].first().astype(float)
    p5 = ranks.groupby("query_id")["same_cat"].mean()
    found = ranks.groupby("query_id")["exact"].any()          # exact item somewhere in our top 5
    saved = saved.set_index("query_id").loc[hit1.index]

    ok_hit1 = np.allclose(hit1, saved["hit1"])
    if p5_excludes_exact:
        exact_ok = np.isclose(p5, saved["p5"])
        shifted_ok = np.isclose(p5 - 0.2, saved["p5"])
        ok_p5 = (exact_ok[~found]).all() and (exact_ok | shifted_ok)[found].all()
    else:
        ok_p5 = np.allclose(p5, saved["p5"])
    ok = ok_hit1 and ok_p5
    print(f"  check {name}: {len(hit1)} queries, hit@1 {'OK' if ok_hit1 else 'MISMATCH'}, "
          f"P@5 {'OK' if ok_p5 else 'MISMATCH'}")
    if not ok:
        raise SystemExit("sanity check failed, not drawing")

if __name__ == "__main__":
    main()