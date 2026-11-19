"""Re-calibrate the app's shortlist on the official LookBench corpus (61,553 images; Finding 38).

Problem: the shortlist constant q (serving/calibrate.py) was calibrated against LookBench's 3,278-item
gallery, but the app searches 54k items. With more items the best similarity can only go up, so the gap
between the best item and the true item grows and the old q covers less than 90%. The official corpus
(gallery + 58,275 noise images) is close to the app's size, so q is re-calibrated there.

Shortlist rule (unchanged, relative split conformal, alpha 0.1): gap = best similarity - similarity of the
true item; q = the ceil((n+1)(1-alpha))-th smallest gap; shortlist = items within q of the best one.
Done separately for crops and whole photos (the app sends either); the app uses the LARGER q, so the 90%
holds for both kinds of photo on LookBench (rule fixed before running, see results_log.md).
"Not sure" is NOT re-calibrated (its loss needs garment groups, which the noise images do not have); we only
report how often it would answer at the bigger scale with the current lambda.

Sanity checks (stop if they fail):
  - on the 3,278 gallery alone, the crop photos give back the saved q (serving/calibration.json);
  - the best full-corpus similarity of each crop photo equals the rank-1 score saved by lookbench_full.py.

Run from the repo root (embeds the 981 crops and 981 whole photos, a few minutes on MPS):
    python -m serving.calibrate_full
Updates serving/calibration.json (old q kept as q_gallery_3278) and writes results/calibration_full.csv.
"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from evaluation.conformal import relative_q
from evaluation.run_eval import RETRIEVERS
from evaluation.run_lookbench import prepare, use_index_dir

METHOD = "finetuned_random"
ALPHA = 0.1
TRIALS = 200
VIEWS = {"crop": "image", "whole": "raw_image"}
OUT = "serving/calibration.json"


def corpus_features(retriever, gallery, noise):
    """Saved embeddings of the gallery and of the noise images (both made earlier, nothing re-embedded)."""
    use_index_dir(retriever, "models/index_lookbench/real_streetlook")
    retriever.build_index(gallery)
    real = retriever.features
    use_index_dir(retriever, "models/index_lookbench/noise")
    retriever.build_index(noise)
    return real, retriever.features


def split_coverage(gaps, rng):
    """200 random 50/50 splits: mean coverage on one half of the q calibrated on the other half."""
    coverage = []
    n = len(gaps)
    for _ in range(TRIALS):
        order = rng.permutation(n)
        cal, test = order[:n // 2], order[n // 2:]
        coverage.append(np.mean(gaps[test] <= relative_q(gaps[cal], ALPHA)))
    return float(np.mean(coverage))


def main():
    gallery, queries = prepare("real_streetlook")
    queries = queries[queries["label"].isin(gallery["label"])].reset_index(drop=True)   # as calibrate.py
    noise = pd.read_csv("data/lookbench/noise/gallery.csv", dtype=str)
    retriever = RETRIEVERS[METHOD]()
    real, extra = corpus_features(retriever, gallery, noise)
    labels = gallery["label"].to_numpy()
    calibration = json.load(open(OUT))
    old_q = calibration["shortlist"].get("q_gallery_3278", calibration["shortlist"]["q"])
    lam = calibration["not_sure"]["lambda"]

    rows, results = [], {}
    for view, column in VIEWS.items():
        print(f"embedding the {len(queries)} LookBench {view} photos ...")
        vectors = np.stack([retriever.embed_query(Image.open(p).convert("RGB")) for p in queries[column]])
        s_real, s_extra = vectors @ real.T, vectors @ extra.T
        true = np.array([s_real[i, labels == lab].max() for i, lab in enumerate(queries["label"])])
        best_gallery = s_real.max(axis=1)
        best_full = np.maximum(best_gallery, s_extra.max(axis=1))

        if view == "crop":
            # check 1: gallery-only crops give back the saved q
            q_check = relative_q(best_gallery - true, ALPHA)
            print(f"  check: gallery-only q {q_check:.6f}, saved {old_q:.6f}")
            if abs(q_check - old_q) > 1e-5:
                raise SystemExit("PROBLEM: the gallery-only q does not match serving/calibration.json; stopping.")
            # check 2: best full-corpus similarity = saved rank-1 score of lookbench_full.py
            ranks = pd.read_csv(f"results/rankings/{METHOD}_lookbench_real_streetlook_image_full.csv")
            saved = ranks[ranks["rank"] == 1].set_index("query_id")["score"].loc[queries["query_id"]]
            ok = np.allclose(best_full, saved.to_numpy(), atol=1e-5)
            print(f"  check: best full-corpus similarity = saved rank-1 score for all photos -> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit("PROBLEM: full-corpus scores do not match lookbench_full.py; stopping.")

        gaps = best_full - true
        q_full = relative_q(gaps, ALPHA)
        coverage = split_coverage(gaps, np.random.default_rng(0))
        # shortlist size (corpus entries within q of the best one) for every photo, with this view's q
        sizes = (s_real >= (best_full - q_full)[:, None]).sum(axis=1) + (s_extra >= (best_full - q_full)[:, None]).sum(axis=1)
        median_size = float(np.median(sizes))
        results[view] = dict(gaps=gaps, q=q_full)
        rows.append(dict(view=view, q_full=q_full, coverage_50_50=coverage, median_shortlist_full=median_size,
                         old_q_coverage_full=float(np.mean(gaps <= old_q)),
                         answer_rate_gallery=float(np.mean(best_gallery >= lam)),
                         answer_rate_full=float(np.mean(best_full >= lam))))

    # cross check: does each view's q cover the other view's photos?
    for a, b in [("crop", "whole"), ("whole", "crop")]:
        rows.append(dict(view=f"q from {a} applied to {b}",
                         coverage_all=float(np.mean(results[b]["gaps"] <= results[a]["q"]))))
    table = pd.DataFrame(rows)
    table.to_csv("results/calibration_full.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))

    q_app = max(results["crop"]["q"], results["whole"]["q"])
    calibration["shortlist"].update(
        q=float(q_app), q_gallery_3278=float(old_q), q_full_crop=float(results["crop"]["q"]),
        q_full_whole=float(results["whole"]["q"]),
        source="LookBench official corpus (3,278 gallery + 58,275 noise); larger q of crops and whole photos")
    with open(OUT, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"app shortlist q: {old_q:.4f} -> {q_app:.4f}; saved {OUT} and results/calibration_full.csv")


if __name__ == "__main__":
    main()
