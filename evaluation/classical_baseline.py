"""Classical (non-deep-learning) image retrieval baseline (task 4b; pre-registered in report/results_log.md).

Problem: our only non-ML image baseline was a plain colour histogram (a strawman). A fair classical
system describes each image with three hand-made features and compares them:
  colour  : HSV colour histogram (8 x 4 x 4 bins), near-white background pixels left out
            (product photos have white backgrounds, street crops do not)
  texture : LBP histogram (local binary patterns, 8 neighbours, radius 1, "uniform": 10 bins), same mask
  shape   : HOG (histogram of oriented gradients) of the grayscale image resized to 64 x 128
score = w_colour * cos(colour) + w_texture * cos(texture) + w_shape * cos(shape)

The weights are chosen on the LookBench DEV half only (grid: all weights in steps of 0.1 that sum to 1,
66 settings; highest dev R@1 on the official corpus), then run ONCE on the TEST half.
Variant "classical + menu": the shopper first picks the garment group from a menu (idealised: always the
right group), and only gallery items of that group are ranked. This needs category labels, so it runs on the
3,278 gallery (the noise images have none). For reference the same menu is put in front of our photo model.

Sanity checks (stop if they fail): saved finetuned_random query embeddings reproduce the saved gallery R@1
(0.520) without a menu; the full-corpus test-half R@1 of finetuned_random is read from the saved per-query file.

Run from the repo root (features of 61,553 images, about 10-20 minutes the first time):
    python -m evaluation.classical_baseline
Writes results/classical_dev.csv and results/classical_baseline.csv.
"""
import hashlib
import itertools
import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.fusion_test_stats import holm
from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY
from evaluation.stats import bootstrap_ci

LB = "data/lookbench/real_streetlook"
NOISE = "data/lookbench/noise"
FEATURES = "models/index_lookbench/classical_full.npz"
GRID = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3) if abs(sum(w) - 1) < 1e-9]


def features(path):
    """One image -> (colour histogram, LBP histogram, HOG vector), each of length 1."""
    image = Image.open(path).convert("RGB").resize((128, 256))
    hsv = np.asarray(image.convert("HSV")).reshape(-1, 3)
    keep = ~((hsv[:, 1] < 26) & (hsv[:, 2] > 230))        # leave out near-white background
    if keep.mean() < 0.05:                                # (almost) all white: it IS a white garment
        keep[:] = True
    colour, _ = np.histogramdd(hsv[keep], bins=(8, 4, 4), range=[(0, 256)] * 3)
    gray = np.asarray(image.convert("L"), dtype=float)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform").reshape(-1)
    texture = np.bincount(lbp[keep].astype(int), minlength=10)[:10].astype(float)
    shape = hog(np.asarray(image.convert("L").resize((64, 128)), dtype=float), orientations=9,
                pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return [v / (np.linalg.norm(v) + 1e-12) for v in (colour.reshape(-1), texture, shape)]


def feature_matrix(paths):
    rows = []
    for n, p in enumerate(paths, start=1):
        rows.append(features(p))
        if n % 500 == 0:
            print(f"  {n}/{len(paths)}", end="\r")
    return [np.stack([r[k] for r in rows]).astype(np.float32) for k in range(3)]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def first_picture_hits(scores, picture, labels, query_label, k=5):
    """hit@1 and hit@5 over the first k DISTINCT pictures (dedup protocol)."""
    n = min(300, len(scores))
    best = np.argpartition(-scores, n - 1)[:n]
    seen = []
    for i in best[np.argsort(-scores[best])]:
        if picture[i] not in seen:
            seen.append(picture[i])
            if len(seen) == k:
                break
    return float(query_label in labels[seen[0]]), float(any(query_label in labels[p] for p in seen)), seen


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    noise = pd.read_csv(f"{NOISE}/gallery.csv", dtype=str)
    corpus = pd.concat([gallery, noise], ignore_index=True)
    corpus["picture"] = [md5(p) for p in corpus["image_path"]]
    picture = corpus["picture"].to_numpy()
    labels = corpus.groupby("picture")["label"].apply(set).to_dict()
    n_real = len(gallery)
    gallery_group = gallery["category"].str.lower().map(GROUP_OF_GALLERY).to_numpy()
    pic_groups = corpus[:n_real].assign(group=gallery_group).groupby("picture")["group"].apply(set).to_dict()
    query_group = queries["category"].str.lower().map(GROUP_OF).to_numpy()

    # sanity check: saved photo-model embeddings reproduce the saved gallery R@1
    q_ft = np.load("models/index_lookbench/real_streetlook/finetuned_random_queries.npy")
    g_ft = np.load("models/index_lookbench/real_streetlook/finetuned_random.npy")
    check = np.mean([first_picture_hits(g_ft @ q_ft[i], picture[:n_real], labels, queries.loc[i, "label"])[0]
                     for i in range(len(queries))])
    print(f"  check: saved finetuned_random embeddings give gallery R@1 {check:.4f} (saved 0.5199)")
    if abs(check - 0.5199) > 0.001:
        raise SystemExit("PROBLEM: saved embeddings do not reproduce the saved R@1; stopping.")

    if not os.path.exists(FEATURES):
        print("computing classical features of the 61,553 corpus images ...")
        c, t, s = feature_matrix(list(corpus["image_path"]))
        np.savez(FEATURES, colour=c, texture=t, shape=s)
    saved = np.load(FEATURES)
    corpus_f = [saved["colour"], saved["texture"], saved["shape"]]
    print("computing classical features of the 981 query crops ...")
    query_f = feature_matrix(list(queries["image"]))

    # DEV: weight grid on the full corpus, chosen by dev R@1
    dev = np.flatnonzero((queries["side"] == "dev").to_numpy())
    sims = [query_f[k][dev] @ corpus_f[k].T for k in range(3)]            # three (dev, corpus) matrices
    grid_rows = []
    for w in GRID:
        score = w[0] * sims[0] + w[1] * sims[1] + w[2] * sims[2]
        hits = [first_picture_hits(score[j], picture, labels, queries.loc[i, "label"])[0] for j, i in enumerate(dev)]
        grid_rows.append(dict(w_colour=w[0], w_texture=w[1], w_shape=w[2], dev_R1=np.mean(hits)))
    grid = pd.DataFrame(grid_rows).sort_values("dev_R1", ascending=False)
    grid.to_csv("results/classical_dev.csv", index=False)
    print("best 5 weight settings on dev:")
    print(grid.head().to_string(index=False))
    w = grid.iloc[0][["w_colour", "w_texture", "w_shape"]].to_numpy(dtype=float)
    colour_only = np.array([1.0, 0.0, 0.0])

    # TEST, once: classical (full corpus), colour only (reference), and the menu variants (gallery)
    test = np.flatnonzero((queries["side"] == "test").to_numpy())
    per = {name: dict(hit1=[], hit5=[], gp5=[]) for name in
           ["classical", "colour only (masked)", "classical + menu", "photo + menu (reference)"]}
    for i in test:
        label = queries.loc[i, "label"]
        sims_i = [query_f[k][i] @ corpus_f[k].T for k in range(3)]
        menu = gallery_group == query_group[i]                # gallery items of the right group
        for name, weights in [("classical", w), ("colour only (masked)", colour_only)]:
            score = sum(weights[k] * sims_i[k] for k in range(3))
            h1, h5, _ = first_picture_hits(score, picture, labels, label)
            _, _, top = first_picture_hits(score[:n_real], picture[:n_real], labels, label)
            per[name]["hit1"].append(h1)
            per[name]["hit5"].append(h5)
            per[name]["gp5"].append(np.mean([query_group[i] in pic_groups[p] for p in top]))
        for name, score in [("classical + menu", sum(w[k] * sims_i[k][:n_real] for k in range(3))),
                            ("photo + menu (reference)", g_ft @ q_ft[i])]:
            score = np.where(menu, score, -np.inf)
            h1, h5, top = first_picture_hits(score, picture[:n_real], labels, label)
            per[name]["hit1"].append(h1)
            per[name]["hit5"].append(h5)
            per[name]["gp5"].append(np.mean([query_group[i] in pic_groups[p] for p in top]))
        print(f"  test {len(per['classical']['hit1'])}/{len(test)}", end="\r")
    print()

    photo_full = pd.read_csv("results/perquery/finetuned_random_lookbench_real_streetlook_image_full.csv").set_index("query_id")
    photo_gallery = pd.read_csv("results/perquery/finetuned_random_lookbench_real_streetlook_image_dedup.csv").set_index("query_id")
    test_ids = queries.loc[test, "query_id"]
    photo = dict(hit1=photo_full.loc[test_ids, "hit1"].to_numpy(), hit5=photo_full.loc[test_ids, "hit5"].to_numpy())
    photo_gal = photo_gallery.loc[test_ids, "hit1"].to_numpy()
    clusters = queries.loc[test, "cluster"].to_numpy()

    rows = [dict(method="photo (finetuned_random)", corpus="full 61,553", R1=photo["hit1"].mean(), R5=photo["hit5"].mean())]
    for name, r in per.items():
        corpus_name = "gallery 3,278" if "menu" in name else "full 61,553"
        row = dict(method=name, corpus=corpus_name, R1=np.mean(r["hit1"]), R5=np.mean(r["hit5"]),
                   groupP5_gallery=np.mean(r["gp5"]))
        row["R1_lo"], row["R1_hi"] = bootstrap_ci(r["hit1"])
        rows.append(row)
    # photo - baseline, R@1, Holm over the two methods taken to test (classical, classical + menu)
    tests = [cluster_bootstrap(photo["hit1"], np.array(per["classical"]["hit1"]), clusters),
             cluster_bootstrap(photo_gal, np.array(per["classical + menu"]["hit1"]), clusters)]
    for name, (diff, lo, hi, p), p_holm in zip(["photo - classical (full corpus)", "photo (gallery) - classical + menu"],
                                               tests, holm(np.array([t[3] for t in tests]))):
        rows.append(dict(method=name, R1_difference=diff, lo=lo, hi=hi, p=p, p_holm=p_holm))
    table = pd.DataFrame(rows)
    table.insert(0, "weights", f"colour {w[0]}, texture {w[1]}, shape {w[2]}")
    table.to_csv("results/classical_baseline.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table[table["R1_difference"].isna()].dropna(axis=1, how="all").to_string(index=False))
        print(table[table["R1_difference"].notna()].dropna(axis=1, how="all").to_string(index=False))
    print("saved results/classical_dev.csv and results/classical_baseline.csv")


if __name__ == "__main__":
    main()
