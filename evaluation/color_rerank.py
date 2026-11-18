"""Colour re-ranking on LookBench real photos (pre-registered in report/results_log.md).

Finding 31: most misses find the right kind of garment but the wrong design. Colour is part of the design,
so we re-rank the top 20 of finetuned_random with
    new score = similarity + beta * colour similarity
where colour similarity = cosine of the 128-bin HSV histograms of the baseline (retrieval/color_hist.py).
Variant "masked" leaves out near-white gallery pixels (white product-photo backgrounds).

Choose beta and the variant on DEV only:
    python -m evaluation.color_rerank --side dev
Then run the chosen setting ONCE on test:
    python -m evaluation.color_rerank --side test --variant masked --beta 0.05
Sanity check: beta = 0 must reproduce the saved R@1 and group P@5 of finetuned_random.
"""
import argparse
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY
from retrieval.color_hist import BINS

LB = "data/lookbench/real_streetlook"
MODEL = "finetuned_random"
BETAS = [0, 0.02, 0.05, 0.1, 0.2]


def histogram(path, mask_white):
    """Normalised 128-bin HSV histogram; optionally without near-white pixels."""
    hsv = np.asarray(Image.open(path).convert("RGB").convert("HSV")).reshape(-1, 3)
    if mask_white:
        keep = ~((hsv[:, 1] < 26) & (hsv[:, 2] > 230))
        if keep.sum() > 100:                       # keep all pixels if almost everything is white
            hsv = hsv[keep]
    hist, _ = np.histogramdd(hsv, bins=BINS, range=[(0, 256)] * 3)
    hist = hist.flatten()
    return hist / (np.linalg.norm(hist) + 1e-12)


def load():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    ranks = pd.read_csv(f"results/rankings/{MODEL}_lookbench_real_streetlook_image_dedup.csv")
    ranks = ranks.merge(queries[["query_id", "label", "category", "image", "side"]], on="query_id")
    labels = gallery.groupby("picture")["label"].apply(set)
    groups = gallery.groupby("picture")["group"].apply(set)
    ranks["true"] = [lab in labels[p] for lab, p in zip(ranks["label"], ranks["picture"])]
    ranks["ok_group"] = [GROUP_OF[c.lower()] in groups[p] for c, p in zip(ranks["category"], ranks["picture"])]
    path_of = gallery.drop_duplicates("picture").set_index("picture")["image_path"]
    return ranks, path_of


def evaluate(ranks, colour_sim, beta):
    """R@1 and group P@5 after re-ranking each query's top 20 by similarity + beta * colour."""
    r = ranks.assign(new=ranks["score"] + beta * colour_sim)
    r = r.sort_values(["query_id", "new"], ascending=[True, False])
    r["new_rank"] = r.groupby("query_id").cumcount() + 1
    r1 = r[r["new_rank"] == 1]["true"].mean()
    gp5 = r[r["new_rank"] <= 5].groupby("query_id")["ok_group"].mean().mean()
    return r1, gp5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["dev", "test"], required=True)
    parser.add_argument("--variant", choices=["plain", "masked"])
    parser.add_argument("--beta", type=float)
    args = parser.parse_args()

    ranks, path_of = load()
    ranks = ranks[ranks["side"] == args.side].reset_index(drop=True)

    # check: beta = 0 reproduces the saved per-query values on this side
    saved = pd.read_csv(f"results/perquery/{MODEL}_lookbench_real_streetlook_image_dedup.csv")
    saved = saved[saved["query_id"].isin(ranks["query_id"])]
    r1, _ = evaluate(ranks, np.zeros(len(ranks)), 0)
    assert np.isclose(r1, saved["hit1"].mean()), "beta = 0 does not reproduce the saved R@1"
    print(f"check OK: beta = 0 gives the saved R@1 {r1:.3f}")

    variants = [args.variant] if args.variant else ["plain", "masked"]
    betas = [args.beta] if args.beta is not None else BETAS
    query_hist = {p: histogram(p, False) for p in ranks["image"].unique()}
    rows = []
    for variant in variants:
        pic_hist = {p: histogram(path_of[p], variant == "masked") for p in ranks["picture"].unique()}
        colour = np.array([query_hist[q] @ pic_hist[p] for q, p in zip(ranks["image"], ranks["picture"])])
        for beta in betas:
            r1, gp5 = evaluate(ranks, colour, beta)
            rows.append(dict(side=args.side, variant=variant, beta=beta, r1=round(r1, 3), group_p5=round(gp5, 3)))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(f"results/color_rerank_{args.side}.csv", index=False)


if __name__ == "__main__":
    main()
