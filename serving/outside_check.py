"""Fix 1: how well does the app recognise garments that the catalogue does NOT sell?

The catalogue has 14 types (no skirts, bags, hats, jumpsuits, ...). For such photos the best results
are necessarily poor, and the shopper should be told why. serving/badges.py (TypePredictor.analyse)
gives outside_prob = total zero-shot probability of 9 out-of-catalogue prompts. The app warns when
outside_prob >= tau.

Evaluation (labels fixed BEFORE running):
  LookBench real crops, by their category:
    OUT     (we sell nothing like it): skirts (all variants), jumpsuit, bags, hat, scarf, belt, sock,
            glasses, glove, tie, umbrella, headband   -> detection rate should be HIGH
    IN      (we sell this kind): t-shirt, shirt, blouse, sweater, sweatshirt, cardigan, vest, jacket,
            pants, shorts, dress, shoe, sneaker, sandals -> false-alarm rate should be LOW
    NEITHER (fuzzy boundary with our jackets): coat, cape -> not counted
  Synthetic phone photos of OUR catalogue (medium, strong): all IN -> false alarms.
Threshold rule, fixed before running: tau = the smallest value in TAUS whose false-alarm rate is at
most 5% on BOTH in-catalogue sets. Saved to serving/outside.json with the full table.

Sanity check: the group badge from analyse() equals predict_group() (the badge of Finding 24 is unchanged).

Run from the repo root (embeds ~1,600 photos with the pretrained model):
    python -m serving.outside_check
"""
import json

import numpy as np
import pandas as pd
from PIL import Image

from serving.badges import TypePredictor

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MAX_FALSE_ALARM = 0.05
SKIRTS = ["skirt", "pleated skirt", "denim skirt", "silk skirt"]
OUT = SKIRTS + ["jumpsuit", "bag", "handbag", "clutch bag", "hat", "scarf", "belt", "sock", "glasses",
                "glove", "tie", "umbrella", "headband"]
IN = ["t-shirt", "shirt", "blouse", "sweater", "sweatshirt", "cardigan", "vest", "jacket", "pants",
      "shorts", "black linen shorts", "dress", "shoe", "sneaker", "lace-up sandals"]
NEITHER = ["coat", "cape"]
OUT_FILE = "serving/outside.json"


def main():
    predictor = TypePredictor()
    lb = pd.read_csv("data/lookbench/real_streetlook/queries.csv", dtype=str)
    lb["cat"] = lb["category"].str.lower()
    unmapped = sorted(set(lb["cat"]) - set(OUT) - set(IN) - set(NEITHER))
    assert not unmapped, f"LookBench categories without a label: {unmapped}"
    lb = lb[~lb["cat"].isin(NEITHER)]

    # check: analyse() gives the same group badge as predict_group()
    for path in lb["image"].head(5):
        a = predictor.analyse(Image.open(path))
        g, p = predictor.predict_group(Image.open(path))
        assert a["group"] == g and np.isclose(a["group_prob"], p), "analyse() changed the group badge"
    print("  check: analyse() reproduces predict_group() (group badge unchanged)")

    rows = []
    for q in lb.itertuples():
        a = predictor.analyse(Image.open(q.image))
        rows.append(dict(source="lookbench", label="out" if q.cat in OUT else "in", cat=q.cat,
                         outside_prob=a["outside_prob"], outside_type=a["outside_type"]))
    queries = pd.read_csv("data/queries.csv")
    for q in queries[queries["condition"].isin(["phone_medium", "phone_strong"])].itertuples():
        a = predictor.analyse(Image.open(q.image_path))
        rows.append(dict(source="synthetic", label="in", cat=q.article_type,
                         outside_prob=a["outside_prob"], outside_type=a["outside_type"]))
    d = pd.DataFrame(rows)
    out_lb = d[(d["source"] == "lookbench") & (d["label"] == "out")]
    in_lb = d[(d["source"] == "lookbench") & (d["label"] == "in")]
    in_syn = d[d["source"] == "synthetic"]
    print(f"  photos: LookBench out-of-catalogue {len(out_lb)}, LookBench in-catalogue {len(in_lb)}, "
          f"synthetic {len(in_syn)}")

    table = []
    for tau in TAUS:
        kinds = {k: float((out_lb.loc[out_lb["cat"].isin(c), "outside_prob"] >= tau).mean())
                 for k, c in [("skirts", SKIRTS), ("bags", ["bag", "handbag", "clutch bag"]),
                              ("other", [c for c in OUT if c not in SKIRTS + ["bag", "handbag", "clutch bag"]])]}
        table.append(dict(tau=tau, detected=float((out_lb["outside_prob"] >= tau).mean()), **{f"detected_{k}": v for k, v in kinds.items()},
                          false_alarm_lookbench=float((in_lb["outside_prob"] >= tau).mean()),
                          false_alarm_synthetic=float((in_syn["outside_prob"] >= tau).mean())))
    table = pd.DataFrame(table)
    print(table.round(3).to_string(index=False))

    ok = table[(table["false_alarm_lookbench"] <= MAX_FALSE_ALARM) & (table["false_alarm_synthetic"] <= MAX_FALSE_ALARM)]
    tau = float(ok["tau"].min()) if len(ok) else None
    print(f"  chosen tau (rule fixed before running): {tau}")
    named = out_lb[out_lb["outside_prob"] >= (tau or 1.0)]
    if len(named):
        skirt_named = named[named["cat"].isin(SKIRTS)]
        print(f"  among detected skirts, named 'skirt': {(skirt_named['outside_type'] == 'skirt').mean():.3f}")
    with open(OUT_FILE, "w") as f:
        json.dump(dict(tau=tau, rule=f"smallest tau with false alarm <= {MAX_FALSE_ALARM} on both in-catalogue sets",
                       table=table.to_dict(orient="records")), f, indent=2)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
