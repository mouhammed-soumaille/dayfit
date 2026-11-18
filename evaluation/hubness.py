"""T5: hubness on LookBench real photos.

In high-dimensional embedding spaces some points become "hubs": they are among the nearest
neighbours of very many queries, while others ("anti-hubs") are almost never retrieved
(Radovanovic et al., 2010). A hub that shows up in many top-5 lists is often a WRONG result
for most of them, so hubness can explain part of the failures.

k-occurrence: N_k(x) = number of query crops whose top-k list contains gallery picture x.
Hubness is measured by the skewness of N_k over all 2,537 distinct gallery pictures
(skewness = E[(N - mean)^3] / sd^3; 0 for a symmetric distribution, large and positive when
a few pictures collect most of the occurrences).
Reference: if every query picked its top-k uniformly at random, N_k ~ Binomial(Q, k/G) with
skewness (1 - 2p) / sqrt(Q p (1 - p)), p = k/G. We print this to compare against.

Queries: we count each DISTINCT crop once (700, not 981): identical crops give identical
rankings (Finding 20), and counting them twice would create fake hubs.

Question 2: are hubs over-represented in failures? Hubs = the top 1% of pictures by N_5.
We compare, over all top-5 slots of all crops, the share of slots filled by a hub when the
result is in the WRONG category group vs the RIGHT group (group as in Finding 19).

Sanity check: the counts must add up: sum over pictures of N_k = (number of crops) x k.

Run from the repo root:
    python -m evaluation.hubness
"""
import hashlib

import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY

LB = "data/lookbench/real_streetlook"
METHODS = ["fashion_siglip", "ensemble_mv", "finetuned", "finetuned_random"]
KS = [5, 10]
HUB_SHARE = 0.01                            # hubs = top 1% of pictures by N_5


def skewness(x):
    x = np.asarray(x, dtype=float)
    return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)


def main():
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)
    queries["crop"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in queries["image"]]
    first_of_crop = queries.drop_duplicates("crop")           # one query per distinct crop
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    gallery["picture"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]]
    gallery["group"] = gallery["category"].str.lower().map(GROUP_OF_GALLERY)
    pic_groups = gallery.groupby("picture")["group"].apply(set)
    pic_path = gallery.groupby("picture")["image_path"].first()
    pictures = pic_groups.index
    Q, G = len(first_of_crop), len(pictures)
    print(f"  {Q} distinct query crops, {G} distinct gallery pictures")

    rows, hubs_of = [], {}
    for m in METHODS:
        ranks = pd.read_csv(f"results/rankings/{m}_lookbench_real_streetlook_image_dedup.csv")
        ranks = ranks[ranks["query_id"].isin(first_of_crop["query_id"])]
        row = {"method": m}
        for k in KS:
            top = ranks[ranks["rank"] <= k]
            counts = top["picture"].value_counts().reindex(pictures, fill_value=0)
            assert counts.sum() == Q * k, "k-occurrence counts do not add up"
            p = k / G
            row[f"skew_N{k}"] = skewness(counts)
            row[f"skew_N{k}_random"] = (1 - 2 * p) / np.sqrt(Q * p * (1 - p))
            row[f"never_in_top{k}"] = float((counts == 0).mean())
            row[f"max_N{k}"] = int(counts.max())
            if k == 5:
                n_hubs = int(np.ceil(HUB_SHARE * G))
                hubs = set(counts.sort_values(ascending=False).index[:n_hubs])
                hubs_of[m] = counts.sort_values(ascending=False)
                row["hub_slot_share"] = float(top["picture"].isin(hubs).mean())

                # is a hub more often a wrong-group result than an ordinary picture?
                top = top.merge(queries[["query_id", "category"]], on="query_id")
                top["right_group"] = [GROUP_OF[c.lower()] in pic_groups[p]
                                      for c, p in zip(top["category"], top["picture"])]
                top["is_hub"] = top["picture"].isin(hubs)
                row["hub_share_in_wrong"] = float(top.loc[~top["right_group"], "is_hub"].mean())
                row["hub_share_in_right"] = float(top.loc[top["right_group"], "is_hub"].mean())
                row["wrong_rate_hub"] = float(1 - top.loc[top["is_hub"], "right_group"].mean())
                row["wrong_rate_other"] = float(1 - top.loc[~top["is_hub"], "right_group"].mean())
        rows.append(row)
        print(f"  {m:17s} skew N5 {row['skew_N5']:.2f} (random {row['skew_N5_random']:.2f})  "
              f"skew N10 {row['skew_N10']:.2f}  never in top-5 {row['never_in_top5']:.0%}  "
              f"max N5 {row['max_N5']}  | top-1% hubs fill {row['hub_slot_share']:.1%} of slots; "
              f"wrong-group rate: hubs {row['wrong_rate_hub']:.1%} vs others {row['wrong_rate_other']:.1%}")
    print("  check: k-occurrence counts add up to (crops x k) for every model and k")

    pd.DataFrame(rows).to_csv("results/hubness.csv", index=False)
    print("saved results/hubness.csv")

    # the 8 biggest hubs of the recommended model, with how many crops retrieve them
    top8 = hubs_of["finetuned_random"].head(8)
    fig, axes = plt.subplots(1, 8, figsize=(16, 2.8))
    for ax, (pic, n) in zip(axes, top8.items()):
        ax.imshow(Image.open(pic_path[pic]).convert("RGB"))
        ax.set_title(f"N5 = {n}\n{', '.join(sorted(pic_groups[pic]))}", fontsize=8)
        ax.axis("off")
    fig.suptitle("finetuned_random: the 8 gallery pictures that appear in the most top-5 lists "
                 f"(out of {Q} distinct query crops)", fontsize=10)
    fig.tight_layout()
    fig.savefig("report/figures/fig11_hubs.png", dpi=100)
    print("saved report/figures/fig11_hubs.png")


if __name__ == "__main__":
    main()