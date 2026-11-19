"""Keyword search baseline, "how shops work today" (task 4a; pre-registered in report/results_log.md).

Problem: our non-ML baselines were strawmen (a colour histogram, and a type menu that shows items in a
random order). A real shop lets the shopper TYPE what they see and ranks items by how well their text
matches. This script simulates that with BM25, the standard text-search ranking, over LookBench's own
item descriptions (category, main attribute, other attributes).

It is an IDEALISED shopper: the query text is LookBench's own annotation of the photo, so the shopper always
names the category and attributes with the shop's own words. Real shoppers would do worse.
  keywords_basic    : category + main attribute          (e.g. "bag backpack")
  keywords_detailed : category + main + other attributes (e.g. "bag backpack printed leather suede")
Only the 3,278 gallery can be searched: the 58,275 noise images have no text, so on the full corpus they
would be invisible to keyword search (an unfair advantage). So everything here, including the photo model
we compare with, uses the 3,278-gallery protocol (dedup).

Ties: short attribute lists give many items the same score. Ties are broken at random, averaged over 50
random orders (a shop has no better information to break them).

BM25 (Robertson & Zaragoza), fixed settings k1 = 1.2, b = 0.75, written out below in plain numpy.

Sanity checks (stop if they fail): the attributes downloaded from Hugging Face line up with our saved
gallery.csv / queries.csv (same item_ID and category in the same order); the saved finetuned_random
gallery R@1 and group P@5 are reproduced from the saved per-query files.

Run from the repo root (reads only the text columns from Hugging Face; a few minutes):
    python -m evaluation.keyword_baseline
Writes data/lookbench/real_streetlook/attributes_*.csv and results/keyword_baseline.csv.
"""
import hashlib
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.fusion_test_stats import holm
from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY, p5_per_query
from evaluation.stats import bootstrap_ci

LB = "data/lookbench/real_streetlook"
K1, B = 1.2, 0.75
RANDOM_ORDERS = 50
VARIANTS = {"keywords_basic": ["category", "main_attribute"],
            "keywords_detailed": ["category", "main_attribute", "other_attributes"]}


def attributes(split):
    """category / main_attribute / other_attributes for every row of one split (text columns only)."""
    path = f"{LB}/attributes_{split}.csv"
    if not os.path.exists(path):
        from datasets import load_dataset
        data = load_dataset("srpone/look-bench", "real_streetlook", split=split)
        data = data.select_columns(["item_ID", "category", "main_attribute", "other_attributes"])
        data.to_pandas().to_csv(path, index=False)
    return pd.read_csv(path, dtype=str).fillna("")


def words(text):
    """Lower-case words; 'high-heeled pump, leather' -> ['high', 'heeled', 'pump', 'leather']."""
    return re.findall(r"[a-z]+", text.lower())


def bm25_scores(query_words, docs, df, avg_len):
    """BM25 score of one query against every document (docs: list of Counters)."""
    n = len(docs)
    scores = np.zeros(n)
    for w in set(query_words):
        if w not in df:
            continue
        idf = np.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
        for i, doc in enumerate(docs):
            tf = doc.get(w, 0)
            if tf:
                length = sum(doc.values())
                scores[i] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_len))
    return scores


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    g_attr, q_attr = attributes("gallery"), attributes("query")

    # check 1: the downloaded text lines up with our saved files, row by row
    ok = (len(g_attr) == len(gallery) and (g_attr["item_ID"] == gallery["label"]).all()
          and (g_attr["category"] == gallery["category"]).all()
          and len(q_attr) == len(queries) and (q_attr["item_ID"] == queries["label"]).all())
    print(f"  check: attributes line up with gallery.csv and queries.csv -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("PROBLEM: attribute rows do not match our saved LookBench files; stopping.")

    # dedup protocol: one entry per distinct picture (Finding 15); a picture's text = all its entries' text
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    text = (g_attr["category"] + " " + g_attr["main_attribute"] + " " + g_attr["other_attributes"])
    pictures = gallery["picture"].drop_duplicates().tolist()
    pic_labels = gallery.groupby("picture")["label"].apply(set)
    pic_groups = gallery.groupby("picture")["group"].apply(set)
    pic_words = text.groupby(gallery["picture"]).apply(lambda t: Counter(w for x in t for w in words(x)))
    docs = [pic_words[p] for p in pictures]
    df = Counter(w for d in docs for w in d)
    avg_len = np.mean([sum(d.values()) for d in docs])

    # how often is the query's own text identical to its true item's text? (how "idealised" the shopper is)
    first_text = dict(zip(g_attr["item_ID"], text))
    q_text = q_attr["category"] + " " + q_attr["main_attribute"] + " " + q_attr["other_attributes"]
    same = np.mean([sorted(words(t)) == sorted(words(first_text.get(lab, ""))) for t, lab in zip(q_text, queries["label"])])
    print(f"  queries whose full annotation equals their true item's annotation: {same:.1%}")

    rng = np.random.default_rng(0)
    results, rows = {}, []
    for name, columns in VARIANTS.items():
        hit1, hit5, ranks = np.zeros(len(queries)), np.zeros(len(queries)), []
        for i in range(len(queries)):
            qwords = words(" ".join(q_attr.loc[i, c] for c in columns))
            scores = bm25_scores(qwords, docs, df, avg_len)
            for r in range(RANDOM_ORDERS):            # random tie-breaking, averaged
                order = np.lexsort((rng.random(len(pictures)), -scores))[:5]
                top = [pictures[k] for k in order]
                hit1[i] += (queries.loc[i, "label"] in pic_labels[top[0]]) / RANDOM_ORDERS
                hit5[i] += any(queries.loc[i, "label"] in pic_labels[p] for p in top) / RANDOM_ORDERS
                if r == 0:                            # one random order is kept for group P@5
                    ranks += [dict(query_id=queries.loc[i, "query_id"], rank=k + 1, picture=p) for k, p in enumerate(top)]
            print(f"  {name} {i + 1}/{len(queries)}", end="\r")
        gp5 = p5_per_query(pd.DataFrame(ranks), queries[["query_id", "category"]], pic_groups,
                           lambda c: GROUP_OF[c.lower()]).loc[queries["query_id"]].to_numpy()
        results[name] = dict(hit1=hit1, hit5=hit5, gp5=gp5)
        print()

    # the photo model on the same 3,278 gallery (saved results; check they give the saved numbers)
    saved = pd.read_csv("results/perquery/finetuned_random_lookbench_real_streetlook_image_dedup.csv").set_index("query_id")
    ranks = pd.read_csv("results/rankings/finetuned_random_lookbench_real_streetlook_image_dedup.csv")
    gp5 = p5_per_query(ranks, queries[["query_id", "category"]], pic_groups, lambda c: GROUP_OF[c.lower()])
    photo = dict(hit1=saved.loc[queries["query_id"], "hit1"].to_numpy(), hit5=saved.loc[queries["query_id"], "hit5"].to_numpy(),
                 gp5=gp5.loc[queries["query_id"]].to_numpy())
    print(f"  check: finetuned_random gallery R@1 {photo['hit1'].mean():.3f} (saved 0.520), "
          f"group P@5 {photo['gp5'].mean():.3f} (saved 0.869)")
    if abs(photo["hit1"].mean() - 0.5199) > 0.001 or abs(photo["gp5"].mean() - 0.8687) > 0.001:
        raise SystemExit("PROBLEM: saved photo-model results not reproduced; stopping.")
    results["finetuned_random (photo)"] = photo

    for side in ["all", "test"]:
        keep = (queries["side"] == "test").to_numpy() if side == "test" else np.ones(len(queries), bool)
        for name, r in results.items():
            row = dict(side=side, method=name, n=int(keep.sum()))
            for metric in ["hit1", "hit5", "gp5"]:
                row[metric] = r[metric][keep].mean()
                row[f"{metric}_lo"], row[f"{metric}_hi"] = bootstrap_ci(r[metric][keep])
            rows.append(row)
        clusters = queries.loc[keep, "cluster"].to_numpy()
        for metric in ["hit1", "gp5"]:
            tests = [cluster_bootstrap(results["finetuned_random (photo)"][metric][keep], results[v][metric][keep], clusters)
                     for v in VARIANTS]
            for v, (diff, lo, hi, p), p_holm in zip(VARIANTS, tests, holm(np.array([t[3] for t in tests]))):
                rows.append(dict(side=side, method=f"photo - {v}", metric=metric, difference=diff, lo=lo, hi=hi,
                                 p=p, p_holm=p_holm))
    table = pd.DataFrame(rows)
    table.to_csv("results/keyword_baseline.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print("saved results/keyword_baseline.csv")


if __name__ == "__main__":
    main()
