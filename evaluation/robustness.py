"""Robustness to bad photos on REAL LookBench crops (task 3; pre-registered in report/results_log.md).

Problem: the project's real-world constraint is imperfect user photos. Our only robustness numbers so far
are on SYNTHETIC phone photos, made with the same augmentation the model was trained on, so they mostly
measure the training augmentation. Here the real street crops are degraded with fixed, pre-registered
filters, some INSIDE the training augmentation (data_prep/phone_aug.py: round Gaussian blur up to 3 px at
224, brightness and contrast 0.45-1.55) and some OUTSIDE it (motion blur, very dark, grayscale, colour cast).

For finetuned_random, the pretrained FashionSigLIP and the colour histogram (reference baseline), per filter:
  - R@1 and R@5 on the official corpus (61,553 entries, main protocol) and group P@5 on the 3,278 gallery;
  - "not sure" rate of the app rule (best full-corpus similarity < lambda), finetuned_random only;
  - how often the colour badge wrongly says "different" for the TRUE item, before and after the
    black-and-white fix of Finding 35 (after: no badge when the photo has almost no colour).
Tests: paired cluster bootstrap (split clusters) finetuned_random - fashion_siglip for every filter, R@1 and
group P@5, Holm over the filters per metric; drop from the clean photo with a 95% CI for each model.

Sanity check (stops if it fails): with no filter, finetuned_random and fashion_siglip reproduce the saved
per-query full-corpus R@1 (results/perquery/*_image_full.csv).

Run from the repo root (~30k query embeddings + colour histograms of 61,553 images, about 20-30 min on MPS):
    python -m evaluation.robustness
Writes results/robustness.csv, results/robustness_tests.csv and report/figures/robustness.png.
Redraw the figure only:  python -m evaluation.robustness --plot_only
"""
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import uniform_filter1d

from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.fusion_test_stats import holm
from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY
from evaluation.run_eval import RETRIEVERS
from evaluation.run_lookbench import prepare, use_index_dir
from retrieval.color_hist import color_histogram
from serving.badges import colour_band, garment_colour
from serving.colour_check import photo_numbers

LB = "data/lookbench/real_streetlook"
NOISE = "data/lookbench/noise"
# (name, kind, strength, inside the training augmentation?) -- fixed before running
CONDITIONS = [
    ("clean", "none", 0, True),
    ("gauss_1.5", "gauss", 1.5, True), ("gauss_3", "gauss", 3, True), ("gauss_6", "gauss", 6, False),
    ("motion_5", "motion", 5, False), ("motion_9", "motion", 9, False), ("motion_15", "motion", 15, False),
    ("dark_0.5", "bright", 0.5, True), ("dark_0.3", "bright", 0.3, False), ("dark_0.15", "bright", 0.15, False),
    ("desaturate_0.5", "colour", 0.5, False), ("grayscale", "colour", 0.0, False),
    ("cast_warm_mild", "cast", (1.0, 0.85, 0.65), False), ("cast_warm_strong", "cast", (1.0, 0.7, 0.45), False),
]
MODELS = ["finetuned_random", "fashion_siglip", "color_hist"]
CANDIDATES = 300


def degrade(image, kind, strength):
    """Apply one filter. Blur sizes are in pixels at the models' 224-pixel input, as in phone_aug.py."""
    image = image.convert("RGB")
    scale = max(image.size) / 224
    if kind == "gauss":
        return image.filter(ImageFilter.GaussianBlur(strength * scale))
    if kind == "motion":                          # camera moved sideways: average along each row
        length = max(1, round(strength * scale))
        arr = uniform_filter1d(np.asarray(image, dtype=float), size=length, axis=1, mode="nearest")
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "bright":
        return ImageEnhance.Brightness(image).enhance(strength)
    if kind == "colour":                          # 0 = grayscale, 0.5 = half the colour
        return ImageEnhance.Color(image).enhance(strength)
    if kind == "cast":                            # warm indoor light: less green and blue
        arr = np.asarray(image, dtype=float) * np.array(strength)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return image


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def top(scores, n):
    n = min(n, len(scores))
    best = np.argpartition(-scores, n - 1)[:n]
    return best[np.argsort(-scores[best])]


def first_pictures(order, picture, depth):
    seen = []
    for i in order:
        if picture[i] not in seen:
            seen.append(picture[i])
            if len(seen) == depth:
                break
    return seen


def corpus_features(method, gallery, noise):
    """Embeddings of gallery + noise (saved ones for the models; colour histograms computed once and saved)."""
    if method == "color_hist":
        path = "models/index_lookbench/color_hist_full.npy"
        if not os.path.exists(path):
            print("computing colour histograms of the 61,553 corpus images ...")
            paths = list(gallery["image_path"]) + list(noise["image_path"])
            np.save(path, np.stack([color_histogram(Image.open(p)) for p in paths]))
        return None, np.load(path)
    retriever = RETRIEVERS[method]()
    use_index_dir(retriever, "models/index_lookbench/real_streetlook")
    retriever.build_index(gallery)
    real = retriever.features
    use_index_dir(retriever, "models/index_lookbench/noise")
    retriever.build_index(noise)
    return retriever, np.concatenate([real, retriever.features])


def plot(table):
    """Bar chart of R@1 per filter for the three methods (legend above the bars, so nothing is hidden)."""
    order = [c[0] for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(order))
    for k, m in enumerate(MODELS):
        values = table[table["method"] == m].set_index("condition").loc[order, "R1"]
        ax.bar(x + (k - 1) * 0.27, values, width=0.27, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([c + ("" if inside else " *") for c, _, _, inside in CONDITIONS], rotation=45, ha="right")
    ax.set_ylabel("R@1, 61,553-image corpus")
    ax.set_ylim(0, 0.65)
    ax.set_title("Real LookBench crops with filters (* = outside the training augmentation)")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig("report/figures/robustness.png", dpi=150)


def main():
    if "--plot_only" in sys.argv:              # redraw the figure from the saved table, no re-run
        plot(pd.read_csv("results/robustness.csv"))
        print("saved report/figures/robustness.png")
        return
    gallery, queries = prepare("real_streetlook")
    queries = queries.merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    noise = pd.read_csv(f"{NOISE}/gallery.csv", dtype=str)
    corpus = pd.concat([gallery, noise], ignore_index=True)
    corpus["picture"] = [md5(p) for p in corpus["image_path"]]
    picture = corpus["picture"].to_numpy()
    n_real = len(gallery)
    labels = corpus.groupby("picture")["label"].apply(set).to_dict()
    corpus["group"] = corpus["category"].str.lower().map(GROUP_OF_GALLERY)
    groups = corpus[:n_real].groupby("picture")["group"].apply(set).to_dict()
    query_group = queries["category"].str.lower().map(GROUP_OF).to_numpy()
    lam = json.load(open("serving/calibration.json"))["not_sure"]["lambda"]
    bands = json.load(open("serving/badges.json"))["colour_bands"]
    colourfulness_min = json.load(open("serving/colour_check.json"))["colourfulness_min"]
    # colour of each query's TRUE item (its first gallery picture), for the colour-badge check
    first_path = gallery.drop_duplicates("label").set_index("label")["image_path"]
    true_colour = {lab: garment_colour(Image.open(first_path[lab])) for lab in queries["label"].unique()
                   if lab in first_path.index}
    crops = [Image.open(p).convert("RGB") for p in queries["image"]]

    per_query = []
    for method in MODELS:
        retriever, features = corpus_features(method, gallery, noise)
        for name, kind, strength, inside in CONDITIONS:
            for i, q in enumerate(queries.itertuples()):
                image = degrade(crops[i], kind, strength)
                vector = retriever.embed_query(image) if retriever else color_histogram(image)
                scores = features @ vector
                pics = first_pictures(top(scores, CANDIDATES), picture, 5)
                gallery_pics = first_pictures(top(scores[:n_real], CANDIDATES), picture, 5)
                row = dict(method=method, condition=name, inside=inside, query_id=q.query_id,
                           hit1=float(q.label in labels[pics[0]]), hit5=float(any(q.label in labels[p] for p in pics)),
                           gp5=float(np.mean([query_group[i] in groups[p] for p in gallery_pics])),
                           not_sure=float(scores.max() < lam))
                if method == "finetuned_random" and q.label in true_colour:   # colour badge of the true item
                    band, _ = colour_band(garment_colour(image), true_colour[q.label], bands)
                    hidden = photo_numbers(image)[0] < colourfulness_min
                    row.update(true_badge_different_before=float(band == "different"),
                               true_badge_different_after=float(band == "different" and not hidden))
                per_query.append(row)
                print(f"  {method} {name} {i + 1}/{len(queries)}", end="\r")
            if name == "clean" and method != "color_hist":
                saved = pd.read_csv(f"results/perquery/{method}_lookbench_real_streetlook_image_full.csv")
                now = pd.DataFrame([r for r in per_query if r["method"] == method and r["condition"] == "clean"])
                ok = np.array_equal(now["hit1"].to_numpy(), saved.set_index("query_id").loc[now["query_id"], "hit1"].to_numpy())
                print(f"\n  check {method}: clean photos reproduce the saved full-corpus R@1 -> {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    raise SystemExit("PROBLEM: clean results do not match lookbench_full.py; stopping.")
        print()

    per_query = pd.DataFrame(per_query)
    per_query.to_csv("results/perquery/robustness.csv", index=False)
    order = [c[0] for c in CONDITIONS]
    table = per_query.groupby(["method", "condition"], sort=False).agg(
        inside=("inside", "first"), R1=("hit1", "mean"), R5=("hit5", "mean"), groupP5=("gp5", "mean"),
        not_sure=("not_sure", "mean"), true_badge_different_before=("true_badge_different_before", "mean"),
        true_badge_different_after=("true_badge_different_after", "mean")).reset_index()

    # drop from clean (each model) and finetuned_random - fashion_siglip (each filter), cluster bootstrap
    clusters = queries["cluster"].to_numpy()
    get = {(m, c): per_query[(per_query["method"] == m) & (per_query["condition"] == c)].set_index("query_id").loc[queries["query_id"]]
           for m in MODELS for c in order}
    drops = []
    for m in MODELS:
        for c in order[1:]:
            diff, lo, hi, _ = cluster_bootstrap(get[(m, c)]["hit1"], get[(m, "clean")]["hit1"], clusters)
            drops.append(dict(method=m, condition=c, R1_drop=diff, R1_drop_lo=lo, R1_drop_hi=hi))
    table = table.merge(pd.DataFrame(drops), on=["method", "condition"], how="left")
    tests = []
    for metric in ["hit1", "gp5"]:
        results = [cluster_bootstrap(get[("finetuned_random", c)][metric], get[("fashion_siglip", c)][metric], clusters)
                   for c in order]
        adjusted = holm(np.array([r[3] for r in results]))
        for c, (diff, lo, hi, p), p_holm in zip(order, results, adjusted):
            tests.append(dict(comparison="finetuned_random - fashion_siglip", condition=c, metric=metric,
                              difference=diff, lo=lo, hi=hi, p=p, p_holm=p_holm))
    table.to_csv("results/robustness.csv", index=False)
    pd.DataFrame(tests).to_csv("results/robustness_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
        print(pd.DataFrame(tests).to_string(index=False))

    plot(table)
    print("saved results/robustness.csv, results/robustness_tests.csv, report/figures/robustness.png")


if __name__ == "__main__":
    main()
