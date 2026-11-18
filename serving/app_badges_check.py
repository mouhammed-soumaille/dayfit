"""App catalogue, step 4: are the type badge and the "not in our catalogue" warning still trustworthy
with the APP vocabulary (72 types in 7 groups, serving/badges.py) instead of the evaluation one
(14 types in 5 groups)?

Labels fixed BEFORE running:
  LookBench real crops: category -> group with LB_GROUP below. Skirts, bags, hats, scarves, belts, socks,
  glasses, gloves, ties, headbands, jumpsuits are now SOLD; coat and cape now count as outerwear.
  Umbrella is the only LookBench category the app catalogue does not sell.
  Synthetic phone photos (medium, strong) of the evaluation catalogue: true group from APP_GROUP_OF_TYPE.
Measured:
  group badge: how often it is shown and how often it is right, at confidence 0, 0.5, 0.7, 0.9
               (the app shows it at >= 0.9), plus per group on LookBench at 0.9
  warning    : false-alarm rate on photos of things we sell, for each tau; detection can only be
               measured on the 5 umbrella photos (too few to conclude anything; reported anyway)
Threshold rule (the same as serving/outside_check.py, fixed before running): tau = the smallest value
in TAUS with false alarm <= 5% on BOTH in-catalogue sets. Saved to serving/outside_app.json.

Sanity checks: every app catalogue type has a prompt and a group; every LookBench category has a label;
the EVALUATION badge is unchanged: it reproduces the saved phone_medium group-badge numbers in
serving/badges.json (Finding 24).

Run from the repo root (first run also saves the app prompt embeddings, needs transformers):
    python -m serving.app_badges_check
"""
import json

import numpy as np
import pandas as pd
from PIL import Image

from serving.badges import APP_GROUP_OF_TYPE, APP_TYPE_WORDS, GROUP_OF_TYPE, TypePredictor

TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CONFIDENCE = [0.0, 0.5, 0.7, 0.9]
MAX_FALSE_ALARM = 0.05
LB_GROUP = {
    "t-shirt": "tops", "shirt": "tops", "blouse": "tops", "sweater": "tops", "sweatshirt": "tops",
    "cardigan": "tops", "vest": "tops",
    "jacket": "outerwear", "coat": "outerwear", "cape": "outerwear",
    "pants": "bottoms", "shorts": "bottoms", "black linen shorts": "bottoms", "skirt": "bottoms",
    "pleated skirt": "bottoms", "denim skirt": "bottoms", "silk skirt": "bottoms",
    "dress": "dresses", "jumpsuit": "dresses",
    "shoe": "footwear", "sneaker": "footwear", "lace-up sandals": "footwear",
    "bag": "bags", "handbag": "bags", "clutch bag": "bags",
    "hat": "accessories", "scarf": "accessories", "belt": "accessories", "sock": "accessories",
    "glasses": "accessories", "glove": "accessories", "tie": "accessories", "headband": "accessories",
}
NOT_SOLD = ["umbrella"]
OUT_FILE = "serving/outside_app.json"


def main():
    app_types = set(pd.read_csv("data/app_catalog.csv")["article_type"])
    assert app_types <= set(APP_TYPE_WORDS) and set(APP_TYPE_WORDS) == set(APP_GROUP_OF_TYPE)
    print(f"  check: all {len(app_types)} app catalogue types have a prompt and a group")

    lb = pd.read_csv("data/lookbench/real_streetlook/queries.csv", dtype=str)
    lb["cat"] = lb["category"].str.lower()
    unmapped = sorted(set(lb["cat"]) - set(LB_GROUP) - set(NOT_SOLD))
    assert not unmapped, f"LookBench categories without a label: {unmapped}"
    queries = pd.read_csv("data/queries.csv")
    synthetic = queries[queries["condition"].isin(["phone_medium", "phone_strong"])]

    # check: the evaluation badge is unchanged (same numbers as serving/badges.json)
    with open("serving/badges.json") as f:
        saved = json.load(f)["evaluation"]["phone_medium"]["group_badge"]["conf>=0.9"]
    old = TypePredictor("evaluation")
    right, conf = [], []
    for q in queries[queries["condition"] == "phone_medium"].itertuples():
        group, prob = old.predict_group(Image.open(q.image_path))
        right.append(group == GROUP_OF_TYPE[q.article_type])
        conf.append(prob)
    right, conf = np.array(right), np.array(conf)
    shown, accuracy = (conf >= 0.9).mean(), right[conf >= 0.9].mean()
    print(f"  check: evaluation badge on phone_medium: shown {shown:.3f} (saved {saved['shown']:.3f}), "
          f"correct {accuracy:.3f} (saved {saved['accuracy_when_shown']:.3f})")
    assert np.isclose(shown, saved["shown"], atol=1e-6) and np.isclose(accuracy, saved["accuracy_when_shown"], atol=1e-6)
    del old

    predictor = TypePredictor("app")
    rows = []
    for q in lb.itertuples():
        a = predictor.analyse(Image.open(q.image))
        rows.append(dict(source="lookbench", cat=q.cat, sold=q.cat not in NOT_SOLD,
                         true_group=LB_GROUP.get(q.cat), **a))
    for q in synthetic.itertuples():
        a = predictor.analyse(Image.open(q.image_path))
        rows.append(dict(source="synthetic", cat=q.article_type, sold=True,
                         true_group=APP_GROUP_OF_TYPE[q.article_type], **a))
    d = pd.DataFrame(rows)
    d["correct"] = d["group"] == d["true_group"]
    sold = d[d["sold"]]
    print(f"  photos: LookBench sold {(sold['source'] == 'lookbench').sum()}, LookBench not sold "
          f"{(~d['sold']).sum()}, synthetic {(d['source'] == 'synthetic').sum()}")

    # group badge
    badge = []
    for source in ["lookbench", "synthetic"]:
        part = sold[sold["source"] == source]
        for t in CONFIDENCE:
            show = part["group_prob"] >= t
            badge.append(dict(source=source, min_confidence=t, shown=float(show.mean()),
                              accuracy_when_shown=float(part.loc[show, "correct"].mean())))
    badge = pd.DataFrame(badge)
    print("  group badge (7 groups):")
    print(badge.round(3).to_string(index=False))
    lb_shown = sold[(sold["source"] == "lookbench") & (sold["group_prob"] >= 0.9)]
    per_group = lb_shown.groupby("true_group")["correct"].agg(accuracy="mean", n="size").reset_index()
    print("  LookBench, badge shown (conf >= 0.9), per true group:")
    print(per_group.round(3).to_string(index=False))

    # out-of-catalogue warning
    table = []
    for tau in TAUS:
        table.append(dict(tau=tau,
                          false_alarm_lookbench=float((sold.loc[sold["source"] == "lookbench", "outside_prob"] >= tau).mean()),
                          false_alarm_synthetic=float((sold.loc[sold["source"] == "synthetic", "outside_prob"] >= tau).mean()),
                          detected_umbrella=float((d.loc[~d["sold"], "outside_prob"] >= tau).mean())))
    table = pd.DataFrame(table)
    print("  warning:")
    print(table.round(3).to_string(index=False))
    ok = table[(table["false_alarm_lookbench"] <= MAX_FALSE_ALARM) & (table["false_alarm_synthetic"] <= MAX_FALSE_ALARM)]
    tau = float(ok["tau"].min()) if len(ok) else None
    print(f"  chosen tau (rule fixed before running): {tau}")

    with open(OUT_FILE, "w") as f:
        json.dump(dict(tau=tau, rule=f"smallest tau with false alarm <= {MAX_FALSE_ALARM} on both in-catalogue sets",
                       n_umbrella=int((~d["sold"]).sum()), table=table.to_dict(orient="records"),
                       group_badge=badge.to_dict(orient="records"),
                       group_badge_per_group_lookbench=per_group.to_dict(orient="records")), f, indent=2)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()
