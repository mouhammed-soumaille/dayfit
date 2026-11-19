"""Whole photo vs hand-drawn crop on LookBench real photos (pre-registered in report/results_log.md).

Problem: every real-photo number so far uses LookBench's hand-drawn crop around the garment, but the app
sends the WHOLE photo by default (the crop box is optional). The whole photo also has a person, a
background and usually several garments. One street photo can be the query for several items (a top, a
pair of trousers, a bag): the app returns one ranking per photo, so it cannot put all of them first.

This script reads the per-query results that evaluation/run_lookbench.py saved for the crop ("image")
and the whole photo ("raw_image"), for finetuned_random and the pretrained FashionSigLIP, and reports:
  - R@1, R@5 and group P@5 (dedup protocol), on all 981 queries and on the TEST half;
  - paired cluster bootstrap (clusters = the split clusters): whole - crop for each model, and
    finetuned_random - fashion_siglip on whole photos; Holm over these 3 tests per metric and side;
  - the R@1 ceiling on whole photos: a photo's rank-1 picture can only be right for the queries whose
    item it shows, so R@1 <= (number of queries whose item can be first) / 981;
  - R@1 for photos with ONE query vs photos shared by several queries.

Sanity check: the crop numbers reproduce the saved R@1 (results/lookbench_summary.csv) and group P@5
(results/group_p5.csv) of both models; stops otherwise.

Run from the repo root, AFTER the two whole-photo runs:
    python -m evaluation.run_lookbench --subset real_streetlook --method finetuned_random --query_image raw_image
    python -m evaluation.run_lookbench --subset real_streetlook --method fashion_siglip --query_image raw_image
    python -m evaluation.whole_photo
Writes results/whole_photo.csv.
"""
import hashlib

import numpy as np
import pandas as pd

from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.fusion_test_stats import holm
from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY, p5_per_query

LB = "data/lookbench/real_streetlook"
MODELS = ["finetuned_random", "fashion_siglip"]
VIEWS = {"crop": "image", "whole": "raw_image"}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_gallery():
    """Each distinct gallery picture's item labels and garment groups (dedup protocol, Finding 15)."""
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    gallery["picture"] = [md5(p) for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    return gallery.groupby("picture")["label"].apply(set), gallery.groupby("picture")["group"].apply(set)


def per_query(model, view, queries, pic_groups):
    """hit1, hit5 (saved by run_lookbench) and group P@5 (from the saved rankings) for every query."""
    saved = pd.read_csv(f"results/perquery/{model}_lookbench_real_streetlook_{VIEWS[view]}_dedup.csv")
    saved = saved.set_index("query_id")
    ranks = pd.read_csv(f"results/rankings/{model}_lookbench_real_streetlook_{VIEWS[view]}_dedup.csv")
    gp5 = p5_per_query(ranks, queries[["query_id", "category"]], pic_groups, lambda c: GROUP_OF[c.lower()])
    out = pd.DataFrame({"hit1": saved["hit1"], "hit5": saved["hit5"]})
    out["gp5"] = gp5
    return out.loc[queries["query_id"]]


def sanity_check(results):
    summary = pd.read_csv("results/lookbench_summary.csv")
    group_saved = pd.read_csv("results/group_p5.csv").set_index("method")["GroupP@5"]
    for model in MODELS:
        row = summary[(summary["method"] == model) & (summary["query_image"] == "image")
                      & (summary["protocol"] == "dedup")].iloc[0]
        r1, gp5 = results[(model, "crop")]["hit1"].mean(), results[(model, "crop")]["gp5"].mean()
        print(f"  check {model}: crop R@1 {r1:.4f} (saved {row['Recall@1']:.4f}), "
              f"group P@5 {gp5:.4f} (saved {group_saved[model]:.4f})")
        if abs(r1 - row["Recall@1"]) > 1e-9 or abs(gp5 - group_saved[model]) > 1e-9:
            raise SystemExit("PROBLEM: crop results do not reproduce the saved numbers; stopping.")


def main():
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    queries["photo"] = [md5(p) for p in queries["raw_image"]]
    pic_labels, pic_groups = load_gallery()
    results = {(m, v): per_query(m, v, queries, pic_groups) for m in MODELS for v in VIEWS}
    sanity_check(results)

    shared = queries["photo"].map(queries["photo"].value_counts()) > 1
    print(f"  {len(queries)} queries, {queries['photo'].nunique()} distinct whole photos; "
          f"{int(shared.sum())} queries share their photo with other queries")

    # R@1 ceiling on whole photos: for each photo, the best one picture can do is to show as many of
    # the photo's target items as possible; it can never be right for more queries than that
    ceiling = 0
    for _, part in queries.groupby("photo"):
        wanted = set(part["label"])
        best = max(len(wanted & labels) for labels in pic_labels)
        ceiling += sum(part["label"].isin(wanted)) if best == len(wanted) else best
    print(f"  R@1 ceiling on whole photos (one ranking per photo): {ceiling / len(queries):.3f}")

    rows = []
    for side in ["all", "test"]:
        keep = (queries["side"] == "test").to_numpy() if side == "test" else np.ones(len(queries), bool)
        clusters = queries.loc[keep, "cluster"].to_numpy()
        for (model, view), r in results.items():
            part = r[keep]
            rows.append(dict(side=side, row="value", model=model, view=view, n=int(keep.sum()),
                             R1=part["hit1"].mean(), R5=part["hit5"].mean(), groupP5=part["gp5"].mean(),
                             R1_one_query_photos=part.loc[~shared[keep].to_numpy(), "hit1"].mean(),
                             R1_shared_photos=part.loc[shared[keep].to_numpy(), "hit1"].mean()))
        comparisons = [("finetuned_random whole - crop", results[("finetuned_random", "whole")], results[("finetuned_random", "crop")]),
                       ("fashion_siglip whole - crop", results[("fashion_siglip", "whole")], results[("fashion_siglip", "crop")]),
                       ("whole: finetuned_random - fashion_siglip", results[("finetuned_random", "whole")], results[("fashion_siglip", "whole")])]
        for metric in ["hit1", "gp5"]:
            tests = [cluster_bootstrap(a[keep][metric], b[keep][metric], clusters) for _, a, b in comparisons]
            adjusted = holm(np.array([t[3] for t in tests]))
            for (name, _, _), (diff, lo, hi, p), p_holm in zip(comparisons, tests, adjusted):
                rows.append(dict(side=side, row="test", model=name, view=metric, n=int(keep.sum()),
                                 difference=diff, lo=lo, hi=hi, p=p, p_holm=p_holm))

    table = pd.DataFrame(rows)
    table.to_csv("results/whole_photo.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(table[table["row"] == "value"].drop(columns=["row", "difference", "lo", "hi", "p", "p_holm"]).to_string(index=False))
        print(table[table["row"] == "test"][["side", "model", "view", "difference", "lo", "hi", "p", "p_holm"]].to_string(index=False))
    print("saved results/whole_photo.csv")


if __name__ == "__main__":
    main()
