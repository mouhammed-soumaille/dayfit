"""Fusion experiments on LookBench real photos, numpy only (pre-registered in report/results_log.md).

Uses the embeddings saved by evaluation/export_lookbench_features.py. For every query:
    score = (1 - a) * finetuned_random similarity + a * DINOv2 similarity      (b: DINOv2 fusion)
          + w * P(picture's group | photo)                                        (a: group bonus)
then optionally the top 20 are re-ranked with + beta * colour similarity       (colour, Finding 32)
Dedup protocol as everywhere: duplicate gallery files are merged into one "picture".

Grids on DEV:      python -m evaluation.fusion_lookbench --side dev
One setting:       python -m evaluation.fusion_lookbench --side dev --a 0.3 --w 0.05 --beta 0.02
Final, ONCE:       python -m evaluation.fusion_lookbench --side test --a 0.3 --w 0.05 --beta 0.02 --save NAME
(--save writes per-query hit1 and group P@5 to results/perquery/NAME_lookbench_test.csv for the bootstrap.)
Sanity check: a = w = beta = 0 reproduces the saved R@1 and group P@5 of finetuned_random on that side.
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from evaluation.color_rerank import histogram
from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY
from serving.badges import APP_GROUP_OF_TYPE, APP_TYPE_WORDS

LB = "data/lookbench/real_streetlook"
EMB = "models/index_lookbench/real_streetlook"
GROUPS = sorted(set(APP_GROUP_OF_TYPE.values()))
A_GRID = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]
W_GRID = [0, 0.02, 0.05, 0.1, 0.2]
DEPTH = 20                                   # colour re-ranking only looks at the top 20


def load():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    first = ~gallery["picture"].duplicated()          # one row per distinct picture (copies are identical)
    pics = gallery[first].reset_index()
    labels = gallery.groupby("picture")["label"].apply(set).loc[pics["picture"]].to_numpy()
    groups = gallery.groupby("picture")["group"].apply(set).loc[pics["picture"]].to_numpy()

    ft = np.load(f"{EMB}/finetuned_random_queries.npy") @ np.load(f"{EMB}/finetuned_random.npy")[first].T
    dino = np.load(f"{EMB}/dinov2_queries.npy") @ np.load(f"{EMB}/dinov2_gallery.npy")[first].T

    # zero-shot group probabilities of each query photo (pretrained FashionSigLIP, app prompts)
    scale = float(open(f"{EMB}/siglip_logit_scale.txt").read())
    logits = scale * np.load(f"{EMB}/siglip_queries.npy") @ np.load("serving/type_text_app.npy").T
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    member = np.array([[APP_GROUP_OF_TYPE[t] == g for g in GROUPS] for t in APP_TYPE_WORDS], dtype=float)
    group_prob = probs @ member                                               # (queries, 7 groups)
    pic_member = np.array([[g in s for g in GROUPS] for s in groups], dtype=float)  # (pictures, 7)
    bonus = (group_prob[:, None, :] * pic_member[None, :, :]).max(axis=2)    # P(picture's group | photo)

    queries["true_group"] = queries["category"].str.lower().map(GROUP_OF)
    return queries, pics, labels, groups, ft, dino, bonus


def evaluate(queries, pics, labels, groups, score, beta=0.0, colour=None):
    """Per-query hit@1 and group P@5 for a (queries x pictures) score matrix."""
    hit1, gp5 = [], []
    for k, q in enumerate(queries.itertuples()):
        top = np.argsort(-score[k])[:DEPTH]
        if beta:
            top = top[np.argsort(-(score[k, top] + beta * colour(q.image, top)), kind="stable")]
        hit1.append(float(q.label in labels[top[0]]))
        gp5.append(np.mean([q.true_group in groups[i] for i in top[:5]]))
    return np.array(hit1), np.array(gp5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["dev", "test"], required=True)
    parser.add_argument("--a", type=float)
    parser.add_argument("--w", type=float)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save")
    args = parser.parse_args()

    queries, pics, labels, groups, ft, dino, bonus = load()
    keep = (queries["side"] == args.side).to_numpy()
    queries, ft, dino, bonus = queries[keep].reset_index(drop=True), ft[keep], dino[keep], bonus[keep]

    # check: plain finetuned_random reproduces the saved values on this side
    saved = pd.read_csv(f"results/perquery/finetuned_random_lookbench_real_streetlook_image_dedup.csv")
    saved = saved.set_index("query_id").loc[queries["query_id"]]
    h, g = evaluate(queries, pics, labels, groups, ft)
    print(f"check: R@1 {h.mean():.3f} (saved {saved['hit1'].mean():.3f}), group P@5 {g.mean():.3f}")
    assert abs(h.mean() - saved["hit1"].mean()) < 0.003, "plain model does not reproduce the saved R@1"

    hist = {}
    def colour(query_path, top):
        if query_path not in hist:
            hist[query_path] = histogram(query_path, False)
        for i in top:
            if i not in hist:
                hist[i] = histogram(pics.loc[i, "image_path"], False)
        return np.array([hist[query_path] @ hist[i] for i in top])

    if args.a is None:                         # grids on dev
        assert args.side == "dev", "grids only on dev"
        rows = []
        for a in A_GRID:
            h, g = evaluate(queries, pics, labels, groups, (1 - a) * ft + a * dino)
            rows.append(dict(part="b_dinov2", value=a, r1=h.mean(), group_p5=g.mean()))
        for w in W_GRID:
            h, g = evaluate(queries, pics, labels, groups, ft + w * bonus)
            rows.append(dict(part="a_group_bonus", value=w, r1=h.mean(), group_p5=g.mean()))
        table = pd.DataFrame(rows).round(3)
        print(table.to_string(index=False))
        table.to_csv("results/fusion_dev.csv", index=False)
        return

    score = (1 - args.a) * ft + args.a * dino + (args.w or 0) * bonus
    h, g = evaluate(queries, pics, labels, groups, score, args.beta, colour)
    print(f"{args.side}: a={args.a} w={args.w} beta={args.beta}  R@1 {h.mean():.3f}  group P@5 {g.mean():.3f}")
    if args.save:
        out = pd.DataFrame(dict(query_id=queries["query_id"], hit1=h, gp5=g))
        out.to_csv(f"results/perquery/{args.save}_lookbench_{args.side}.csv", index=False)
        print("saved per-query values")


if __name__ == "__main__":
    main()
