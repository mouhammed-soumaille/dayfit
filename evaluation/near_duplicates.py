"""D8a: near-duplicate pictures in the LookBench real_streetlook gallery.

Finding 15 merged gallery entries that are the SAME FILE (same md5). The failure gallery
(Finding 17) suggested there may also be NEAR-duplicates: the same photo saved differently
(re-encoded, resized, slightly cropped), so md5 differs. If they exist, retrieving the "wrong"
copy still counts as a miss, the same issue as in Finding 15.

How we find candidates (two independent signals, no model run needed):
  1. cosine similarity of the PRETRAINED FashionSigLIP gallery embeddings (already saved).
     We deliberately do not use finetuned_random: the model we evaluate should not decide
     what counts as a duplicate in its own test set.
  2. dHash: shrink the picture to 9x8 grey pixels, record whether each pixel is brighter than
     its right neighbour (64 bits). Two copies of one photo differ in only a few bits;
     different photos differ in about 32. Distance = number of differing bits (Hamming).
Every pair of distinct pictures with cosine >= 0.90 is saved to results/near_duplicates.csv.

The threshold is NOT chosen here. The script draws a contact sheet with random pairs from
several similarity bins; we look at it and fix the threshold BEFORE recomputing any metric.

Sanity checks:
  - 2,537 distinct pictures (the number found in Finding 15);
  - copies of the same file have identical embeddings (as stated in Finding 15);
  - the saved embeddings are in the same order as gallery.csv.

Run from the repo root:
    python -m evaluation.near_duplicates
"""
import hashlib

import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

LB = "data/lookbench/real_streetlook"
INDEX = "models/index_lookbench/real_streetlook/fashion_siglip"
MIN_SIM = 0.90
BINS = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0001]
PAIRS_PER_BIN = 4


def dhash(path):
    """64-bit difference hash of a picture, as an array of 64 booleans."""
    grey = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=float)
    return (grey[:, 1:] > grey[:, :-1]).flatten()


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]

    emb = np.load(f"{INDEX}.npy")
    with open(f"{INDEX}_paths.txt") as f:
        saved_paths = f.read().split("\n")[1:]     # first line = weights tag
    assert saved_paths == list(gallery["image_path"]), "embeddings not in gallery order"

    # copies of the same file must have identical embeddings
    worst = max(np.abs(emb[idx] - emb[idx[0]]).max()
                for idx in gallery.groupby("picture").indices.values())
    print(f"  check: copies of one file differ by at most {worst:.2e} in embedding (expect 0)")

    # one row per distinct picture (keep its first entry), with all its labels and categories
    pics = gallery.groupby("picture").agg(entry=("item_id", "first"),
                                          path=("image_path", "first"),
                                          labels=("label", lambda s: set(s)),
                                          categories=("category", lambda s: set(s)))
    print(f"  check: {len(pics)} distinct pictures (expect 2537)")
    row_of = {item: i for i, item in enumerate(gallery["item_id"])}
    E = emb[[row_of[e] for e in pics["entry"]]]

    sim = E @ E.T                                   # cosine similarity (embeddings have length 1)
    i, j = np.where(np.triu(sim, k=1) >= MIN_SIM)   # each unordered pair once, no self-pairs
    hashes = np.array([dhash(p) for p in pics["path"]])
    pairs = pd.DataFrame(dict(
        path_a=pics["path"].values[i], path_b=pics["path"].values[j],
        cosine=sim[i, j],
        hamming=(hashes[i] != hashes[j]).sum(axis=1),
        cats_a=[", ".join(sorted(c)) for c in pics["categories"].values[i]],
        cats_b=[", ".join(sorted(c)) for c in pics["categories"].values[j]],
        share_label=[bool(a & b) for a, b in zip(pics["labels"].values[i], pics["labels"].values[j])],
    )).sort_values("cosine", ascending=False)
    pairs.to_csv("results/near_duplicates.csv", index=False)
    print(f"  saved {len(pairs)} pairs with cosine >= {MIN_SIM} to results/near_duplicates.csv")

    # how many pairs per similarity bin, and how often they look alike pixel-wise / share an item
    pairs["bin"] = pd.cut(pairs["cosine"], BINS, right=False)
    print(pairs.groupby("bin", observed=False).agg(
        pairs=("cosine", "size"),
        median_hamming=("hamming", "median"),
        hamming_le_10=("hamming", lambda h: (h <= 10).mean()),
        share_label=("share_label", "mean")).round(2).to_string())

    # contact sheet: PAIRS_PER_BIN random pairs from every bin (fixed seed), two pictures per pair
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(BINS) - 1, 2 * PAIRS_PER_BIN, figsize=(16, 2.6 * (len(BINS) - 1)))
    for r, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        in_bin = pairs[(pairs["cosine"] >= lo) & (pairs["cosine"] < hi)]
        chosen = in_bin.iloc[rng.choice(len(in_bin), size=min(PAIRS_PER_BIN, len(in_bin)), replace=False)] \
            if len(in_bin) else in_bin
        for c in range(2 * PAIRS_PER_BIN):
            axes[r, c].axis("off")
        for k, pair in enumerate(chosen.itertuples()):
            for side, path, cats in [(0, pair.path_a, pair.cats_a), (1, pair.path_b, pair.cats_b)]:
                ax = axes[r, 2 * k + side]
                ax.imshow(Image.open(path).convert("RGB"))
                ax.set_title(cats if side else f"cos {pair.cosine:.3f}  ham {pair.hamming}\n{cats}",
                             fontsize=7)
        axes[r, 0].text(-0.15, 0.5, f"[{lo:.2f}, {min(hi, 1):.2f})", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Candidate near-duplicate gallery pictures (pretrained FashionSigLIP cosine), "
                 f"{PAIRS_PER_BIN} random pairs per bin, seed 0", fontsize=10)
    fig.tight_layout()
    fig.savefig("report/figures/d8_near_duplicate_candidates.png", dpi=100)
    print("  saved report/figures/d8_near_duplicate_candidates.png")


if __name__ == "__main__":
    main()