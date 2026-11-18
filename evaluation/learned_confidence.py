"""Can a learned confidence score make "not sure" more useful? (top similarity vs logistic regression
vs gradient-boosted trees)

Finding 23: the app's "not sure" uses the top similarity as its confidence score. Risk control then
guarantees E[loss] <= alpha (loss = share of the top 5 in the wrong category group, abstaining counts
as 0), but it mostly reaches alpha by answering LESS, not by recognising the bad photos. Conformal risk
control works with ANY confidence score, as long as the score is learned on other photos than the ones
used to calibrate lambda. So a better score should answer MORE photos at the same alpha.

Plan (fixed before any result):
  features of one photo (finetuned_random on LookBench, dedup not needed: same entries as the app's loss):
    top1        best similarity
    gap12       best minus 2nd-best similarity (small gap = ambiguous)
    gap15       best minus 5th-best similarity
    shortlist   number of gallery entries within q of the best (q from serving/calibration.json)
    group_prob  type-badge confidence (pretrained FashionSigLIP, serving/badges.py)
    outside_prob  out-of-catalogue probability (serving/badges.py)
    colour_de   mean CIEDE2000 colour difference between the photo and its top 5
  target: loss = share of the top 5 in the wrong group; the classifiers learn y = (loss == 0)
  scores compared: top1 (no training) | logistic regression | gradient-boosted trees
    (HistGradientBoostingClassifier with the fixed settings below, no tuning)
  50 random splits BY CROP (identical query crops stay together, Finding 20):
    40% train the model, 30% calibrate lambda (alpha 0.1), 30% test
  main result: answer rate on the test part at alpha 0.1 (mean over splits, and in how many splits
  the trees beat top1); also the test loss (must stay <= 0.1) and AUROC for y on the test part.

Stage 1 (needs the models, run on the Mac):  python -m evaluation.learned_confidence --features
  -> results/confidence_features.csv
Stage 2 (only pandas + scikit-learn):       python -m evaluation.learned_confidence
  -> results/learned_confidence.csv

Sanity check (stage 1): the mean loss over all photos equals the no-abstention loss of Finding 23 (0.1297).
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from evaluation.conformal_extra import pick_lambda

FEATURES = ["top1", "gap12", "gap15", "shortlist", "group_prob", "outside_prob", "colour_de"]
FEATURE_CSV = "results/confidence_features.csv"
OUT = "results/learned_confidence.csv"
ALPHA = 0.1
SPLITS = 50
GBT_SETTINGS = dict(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)


def make_features():
    """Stage 1: one row of features (and the loss) per LookBench photo."""
    from PIL import Image
    from skimage.color import deltaE_ciede2000

    from evaluation.conformal_shift import lookbench_scores
    from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY
    from evaluation.run_eval import RETRIEVERS
    from evaluation.run_lookbench import prepare, use_index_dir
    from serving.badges import TypePredictor, garment_colour

    gallery, queries = prepare("real_streetlook")
    queries = queries[queries["label"].isin(gallery["label"])].reset_index(drop=True)
    retriever = RETRIEVERS["finetuned_random"]()
    use_index_dir(retriever, os.path.join("models", "index_lookbench", "real_streetlook"))
    retriever.build_index(gallery)
    print("embedding LookBench photos ...")
    scores, _ = lookbench_scores(retriever, gallery, queries)
    order = np.argsort(-scores, axis=1)
    top5 = order[:, :5]
    sorted_scores = np.take_along_axis(scores, order[:, :5], axis=1)
    with open("serving/calibration.json") as f:
        q = json.load(f)["shortlist"]["q"]

    group_q = np.array([GROUP_OF[c.lower()] for c in queries["category"]])
    group_g = np.array([GROUP_OF_GALLERY[c.lower()] for c in gallery["category"]])
    loss = (group_g[top5] != group_q[:, None]).mean(axis=1)

    predictor = TypePredictor()
    colour_cache = {}

    def colour_of(path):
        if path not in colour_cache:
            colour_cache[path] = garment_colour(Image.open(path))
        return colour_cache[path]

    rows = []
    for i, query in enumerate(queries.itertuples()):
        image = Image.open(query.image)
        seen = predictor.analyse(image)
        query_colour = garment_colour(image)
        de = [float(deltaE_ciede2000(query_colour[None], colour_of(gallery["image_path"].iloc[j])[None])[0])
              for j in top5[i]]
        rows.append(dict(query_id=query.query_id,
                         crop=hashlib.md5(open(query.image, "rb").read()).hexdigest(),
                         top1=sorted_scores[i, 0], gap12=sorted_scores[i, 0] - sorted_scores[i, 1],
                         gap15=sorted_scores[i, 0] - sorted_scores[i, 4],
                         shortlist=int((scores[i] >= sorted_scores[i, 0] - q).sum()),
                         group_prob=seen["group_prob"], outside_prob=seen["outside_prob"],
                         colour_de=float(np.mean(de)), loss=loss[i]))
        print(f"  {i + 1}/{len(queries)}", end="\r")
    table = pd.DataFrame(rows)
    print(f"\n  check: mean loss {table['loss'].mean():.4f} (Finding 23 no-abstention loss: 0.1297)")
    table.to_csv(FEATURE_CSV, index=False)
    print(f"saved {FEATURE_CSV}")


def evaluate(conf_cal, loss_cal, conf_test, loss_test):
    """Risk control at ALPHA with a given confidence score: calibrate lambda, apply it to the test part."""
    lam = pick_lambda(conf_cal, loss_cal, ALPHA)
    answer = conf_test >= lam
    return dict(answer_rate=answer.mean(), loss=np.mean(answer * loss_test),
                p5_when_answered=1 - loss_test[answer].mean() if answer.any() else np.nan)


def compare():
    """Stage 2: 50 random splits by crop; three confidence scores under the same risk control."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = pd.read_csv(FEATURE_CSV)
    crops = d["crop"].unique()
    y = (d["loss"] == 0).astype(int).to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for split in range(SPLITS):
        shuffled = rng.permutation(crops)
        n = len(shuffled)
        part = {c: ("train" if k < 0.4 * n else "cal" if k < 0.7 * n else "test") for k, c in enumerate(shuffled)}
        which = d["crop"].map(part).to_numpy()
        tr, ca, te = which == "train", which == "cal", which == "test"
        X = d[FEATURES].to_numpy()
        models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                  "trees": HistGradientBoostingClassifier(**GBT_SETTINGS)}
        confidence = {"top1": d["top1"].to_numpy()}
        for name, model in models.items():
            model.fit(X[tr], y[tr])
            confidence[name] = model.predict_proba(X)[:, 1]          # P(top 5 is entirely the right kind)
        loss = d["loss"].to_numpy()
        for name, conf in confidence.items():
            result = evaluate(conf[ca], loss[ca], conf[te], loss[te])
            result.update(split=split, score=name, auroc=roc_auc_score(y[te], conf[te]))
            rows.append(result)
        print(f"  split {split + 1}/{SPLITS}", end="\r")

    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    summary = table.groupby("score")[["answer_rate", "loss", "p5_when_answered", "auroc"]].agg(["mean", "std"])
    print("\n\nrisk control at alpha 0.1, test parts of 50 splits by crop (mean, sd):")
    print(summary.round(3).to_string())
    wide = table.pivot(index="split", columns="score", values="answer_rate")
    for name in ["logistic", "trees"]:
        diff = wide[name] - wide["top1"]
        print(f"  {name} vs top1: answer rate {diff.mean():+.3f} on average, better in "
              f"{(diff > 0).sum()}/{SPLITS} splits, worse in {(diff < 0).sum()}/{SPLITS}")
    print(f"  check: mean test loss <= {ALPHA} for every score -> "
          f"{'OK' if (table.groupby('score')['loss'].mean() <= ALPHA + 1e-9).all() else 'NOT MET'}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="store_true", help="stage 1: compute the features (needs the models)")
    args = parser.parse_args()
    if args.features:
        make_features()
    else:
        compare()
