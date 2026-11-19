"""Learned re-ranker trained on REAL photos (task 5, step 2; pre-registered in report/results_log.md).

Stage 1: finetuned_random's top 20 on the official corpus. Stage 2: a small model scores every
(photo, candidate) pair from the features of evaluation/rerank_features.py and re-orders the 20.
It is trained "pointwise": each pair is one example, label 1 if the candidate is the true item.

Two fixed models (no tuning of their settings):
  logistic regression        (features standardised, C = 1)
  gradient-boosted trees     (sklearn HistGradientBoostingClassifier, 200 trees, learning rate 0.05,
                              at most 15 leaves, random_state 0)
Choice, on the DEV half only: 5-fold cross-validation BY CLUSTER inside dev (photos that share a crop,
street photo or item stay in the same fold); the model with the higher cross-validated dev R@1 is chosen
(tie: logistic regression, the simpler one). It is then trained on all of dev and run ONCE on the TEST half,
whatever its dev result (a negative result is reported).

After the single test run, descriptive only (nothing is chosen with them):
  - learning curve: train on 50 / 100 / 250 / all 500 dev photos (20 random subsets each) -> test R@1;
  - permutation importance on test: R@1 lost when one feature is shuffled (20 shuffles);
  - CPU latency of stage 2 (DINOv2 on the photo + patch matching against 20 stored candidates + the model).

Sanity check (stops if it fails): keeping the stage-1 order gives the saved dev and test R@1.

Run from the repo root (after rerank_features.py; a few minutes):
    python -m evaluation.rerank_model
Writes results/rerank_dev.csv, results/rerank_test.csv, results/rerank_curve.csv, results/rerank_importance.csv.
"""
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.rerank_features import load_dino

LB = "data/lookbench/real_streetlook"
FEATURES = ["ft_sim", "ft_gap", "rank", "pre_sim", "dino_sim", "patch_score", "colour_sim", "delta_e", "group_agree"]
CURVE_SIZES = [50, 100, 250, 500]


def new_model(name):
    if name == "logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)


def hits(pairs, score):
    """Per photo: is the highest-scoring candidate the true item (hit1), and is it in the best 5 (hit5)?"""
    ranked = pairs.assign(s=score).sort_values(["query_id", "s"], ascending=[True, False])
    top = ranked.groupby("query_id").head(5)
    hit1 = ranked.groupby("query_id")["label"].first()
    hit5 = top.groupby("query_id")["label"].max()
    return hit1, hit5


def stage1_score(pairs):
    return -pairs["rank"].to_numpy(dtype=float)          # keep the stage-1 order


def cross_validated_r1(name, dev_pairs):
    """5-fold CV by cluster inside dev: every dev photo is scored by a model that never saw its cluster."""
    oof = np.zeros(len(dev_pairs))
    for train, held in GroupKFold(n_splits=5).split(dev_pairs, groups=dev_pairs["cluster"]):
        model = new_model(name).fit(dev_pairs.iloc[train][FEATURES], dev_pairs.iloc[train]["label"])
        oof[held] = model.predict_proba(dev_pairs.iloc[held][FEATURES])[:, 1]
    return hits(dev_pairs, oof)[0].mean()


def main():
    pairs = pd.read_csv("results/rerank_pairs.csv").merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    dev, test = pairs[pairs["side"] == "dev"].reset_index(drop=True), pairs[pairs["side"] == "test"].reset_index(drop=True)
    saved = pd.read_csv("results/perquery/finetuned_random_lookbench_real_streetlook_image_full.csv").set_index("query_id")
    for name, part in [("dev", dev), ("test", test)]:
        mine = hits(part, stage1_score(part))[0]
        want = saved.loc[mine.index, "hit1"]
        print(f"  check: stage-1 order gives {name} R@1 {mine.mean():.4f} (saved {want.mean():.4f})")
        if not np.array_equal(mine.to_numpy(), want.to_numpy()):
            raise SystemExit("PROBLEM: stage-1 order does not reproduce the saved R@1; stopping.")
    ceiling = dev.groupby("query_id")["label"].max().mean()
    print(f"  dev: plain R@1 {hits(dev, stage1_score(dev))[0].mean():.3f}, ceiling (true item in top 20) {ceiling:.3f}")

    # DEV: choose between the two models by cross-validated R@1
    dev_rows = [dict(model=name, dev_cv_R1=cross_validated_r1(name, dev)) for name in ["logistic regression", "gradient-boosted trees"]]
    dev_rows.append(dict(model="stage 1 only", dev_cv_R1=hits(dev, stage1_score(dev))[0].mean()))
    dev_table = pd.DataFrame(dev_rows)
    dev_table.to_csv("results/rerank_dev.csv", index=False)
    print(dev_table.to_string(index=False))
    lr, hgb = dev_table["dev_cv_R1"].iloc[0], dev_table["dev_cv_R1"].iloc[1]
    chosen = "gradient-boosted trees" if hgb > lr else "logistic regression"
    print(f"chosen on dev: {chosen}")

    # TEST, once
    model = new_model(chosen).fit(dev[FEATURES], dev["label"])
    score = model.predict_proba(test[FEATURES])[:, 1]
    hit1, hit5 = hits(test, score)
    base1, base5 = hits(test, stage1_score(test))
    clusters = test.groupby("query_id")["cluster"].first().loc[hit1.index].to_numpy()
    d1 = cluster_bootstrap(hit1.to_numpy(), base1.loc[hit1.index].to_numpy(), clusters)
    d5 = cluster_bootstrap(hit5.loc[hit1.index].to_numpy(), base5.loc[hit1.index].to_numpy(), clusters)
    test_table = pd.DataFrame([
        dict(method="stage 1 (finetuned_random)", R1=base1.mean(), R5=base5.mean()),
        dict(method=f"re-ranker ({chosen})", R1=hit1.mean(), R5=hit5.mean(),
             R1_difference=d1[0], R1_lo=d1[1], R1_hi=d1[2], R1_p=d1[3],
             R5_difference=d5[0], R5_lo=d5[1], R5_hi=d5[2], R5_p=d5[3])])
    test_table.to_csv("results/rerank_test.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(test_table.to_string(index=False))

    # learning curve (descriptive): how many labelled real photos does a shop need?
    rng = np.random.default_rng(0)
    dev_photos = dev["query_id"].unique()
    curve = []
    for size in CURVE_SIZES:
        repeats = 1 if size >= len(dev_photos) else 20
        for r in range(repeats):
            chosen_photos = rng.choice(dev_photos, size=min(size, len(dev_photos)), replace=False)
            part = dev[dev["query_id"].isin(chosen_photos)]
            if part["label"].nunique() < 2:
                continue
            m = new_model(chosen).fit(part[FEATURES], part["label"])
            curve.append(dict(train_photos=size, repeat=r, test_R1=hits(test, m.predict_proba(test[FEATURES])[:, 1])[0].mean()))
    curve = pd.DataFrame(curve)
    curve.to_csv("results/rerank_curve.csv", index=False)
    print(curve.groupby("train_photos")["test_R1"].agg(["mean", "std", "count"]).round(3).to_string())

    # permutation importance on test (descriptive): R@1 lost when one feature is shuffled
    importance = []
    for feature in FEATURES:
        losses = []
        for r in range(20):
            shuffled = test[FEATURES].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            losses.append(hit1.mean() - hits(test, model.predict_proba(shuffled)[:, 1])[0].mean())
        importance.append(dict(feature=feature, R1_lost=np.mean(losses), sd=np.std(losses)))
    importance = pd.DataFrame(importance).sort_values("R1_lost", ascending=False)
    importance.to_csv("results/rerank_importance.csv", index=False)
    print(importance.round(4).to_string(index=False))

    # CPU latency of stage 2 for one photo (DINOv2 on the photo, patch matching with 20 stored candidates, model)
    torch.set_num_threads(4)
    dino, transform = load_dino("cpu")
    photo = transform(Image.open(pd.read_csv(f"{LB}/queries.csv")["image"][0]).convert("RGB"))
    stored = torch.nn.functional.normalize(torch.randn(20, 256, 768), dim=-1)     # candidates' patches, precomputed offline
    times = []
    for r in range(12):
        start = time.perf_counter()
        with torch.no_grad():
            tokens = dino.forward_features(photo[None])
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
            torch.einsum("pd,cqd->cpq", tokens[0, 1:], stored).max(dim=2).values.mean(dim=1)
        model.predict_proba(test[FEATURES].iloc[:20])
        times.append((time.perf_counter() - start) * 1000)
    print(f"stage 2 on CPU (4 threads), median of 10 after 2 warm-up: {np.median(times[2:]):.0f} ms per photo "
          f"(the candidates' DINOv2 patches would be stored with the catalogue)")
    print("saved results/rerank_dev.csv, rerank_test.csv, rerank_curve.csv, rerank_importance.csv")


if __name__ == "__main__":
    main()
