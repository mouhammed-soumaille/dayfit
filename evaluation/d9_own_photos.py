"""D9: the final untouched test on the author's own wardrobe photos (protocol + amendment in
report/results_log.md, fixed BEFORE any photo was taken). Run the real evaluation ONCE.

Folder layout (data/real_photos/ is git-ignored, the photos never go to GitHub):
  data/real_photos/<any name>.jpg / .png / .heic   one photo per garment, as taken (no editing)
  data/real_photos/labels.csv                      written by hand BEFORE running anything:
      file,group,eval_type
      IMG_0001.jpg,tops,Tshirts
      IMG_0002.jpg,tops,                           (a sweater: not one of the 14 evaluation types)
      IMG_0003.jpg,other,                          (fits none of the 7 groups, e.g. a watch)
  group     = one of the 7 app groups (tops, outerwear, bottoms, dresses, footwear, bags, accessories)
              or "other" (then the photo is reported but not in either primary result)
  eval_type = one of the 14 evaluation-catalogue types, or empty

Three steps (only the last one runs a model):
  python -m evaluation.d9_own_photos --template   writes labels.csv with every photo name, columns empty
  python -m evaluation.d9_own_photos --check      checks the labels (no model, no results shown)
  python -m evaluation.d9_own_photos --run        THE run: once. Refuses if results already exist.

Primary results (frozen methods, no tuning):
  (1) group P@5 of the app as shipped: app catalogue (54,178 items) + group bonus w = 0.05, kids hidden,
      near-identical results skipped. Exactly serving.api.search_photo, the function behind the web page.
      P@5 for one photo = share of the 5 results whose group equals the photo's label.
  (2) type P@5 of finetuned_random on the EVALUATION catalogue (6,197 items, plain similarity),
      only photos whose eval_type is one of the 14. Share of the 5 results with the same article type.
Both with a 95% bootstrap confidence interval over photos (evaluation/stats.py).
Descriptive only: (2) for color_hist, fashion_siglip, ensemble_mv; and from the app: "not sure" rate,
group-badge accuracy, out-of-catalogue warnings, shortlist sizes.
The whole photo is searched (no crop box): the protocol says photos are used as taken.

Sanity check before any own photo is searched: for 3 saved synthetic test queries, every evaluation
method must return exactly its saved top 5 (results/<method>_phone_medium.csv), and the app must be
the 54,178-item app catalogue with the group bonus on.

Outputs:
  results/d9_app.csv     one row per photo: label, the app's 5 results and their groups, badges, warnings
  results/d9_types.csv   one row per photo x method: top 5 types and type P@5
  results/d9_summary.csv the numbers that go into the results log
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

FOLDER = "data/real_photos"
LABELS = os.path.join(FOLDER, "labels.csv")
PHOTO_ENDINGS = (".jpg", ".jpeg", ".png", ".heic", ".webp")
GROUPS = ["tops", "outerwear", "bottoms", "dresses", "footwear", "bags", "accessories"]
EVAL_TYPES = ["Casual Shoes", "Dresses", "Heels", "Jackets", "Jeans", "Kurtas", "Shirts", "Shorts",
              "Sports Shoes", "Sweatshirts", "Tops", "Track Pants", "Trousers", "Tshirts"]
# group of each evaluation type (same as serving/badges.py GROUP_OF_TYPE), to catch label typos
EVAL_GROUP = {"Tshirts": "tops", "Shirts": "tops", "Tops": "tops", "Kurtas": "tops", "Sweatshirts": "tops",
              "Jackets": "outerwear", "Dresses": "dresses", "Jeans": "bottoms", "Trousers": "bottoms",
              "Track Pants": "bottoms", "Shorts": "bottoms", "Casual Shoes": "footwear",
              "Sports Shoes": "footwear", "Heels": "footwear"}
PRIMARY = "finetuned_random"
DESCRIPTIVE = ["color_hist", "fashion_siglip", "ensemble_mv"]
OUT_APP, OUT_TYPES, OUT_SUMMARY = "results/d9_app.csv", "results/d9_types.csv", "results/d9_summary.csv"


def photo_names():
    return sorted(f for f in os.listdir(FOLDER) if f.lower().endswith(PHOTO_ENDINGS))


def make_template():
    if os.path.exists(LABELS):
        sys.exit(f"{LABELS} already exists; not overwriting it.")
    names = photo_names()
    pd.DataFrame({"file": names, "group": "", "eval_type": ""}).to_csv(LABELS, index=False)
    print(f"wrote {LABELS} with {len(names)} photos; fill in group and eval_type by hand.")


def read_labels():
    """Reads labels.csv and stops with a clear message if anything is wrong."""
    labels = pd.read_csv(LABELS, dtype=str).fillna("")
    labels["group"] = labels["group"].str.strip()
    labels["eval_type"] = labels["eval_type"].str.strip()
    problems = []
    names = set(photo_names())
    for f in names - set(labels["file"]):
        problems.append(f"{f}: photo has no label")
    for f in set(labels["file"]) - names:
        problems.append(f"{f}: label but no photo")
    for f in labels["file"][labels["file"].duplicated()]:
        problems.append(f"{f}: labelled twice")
    for _, row in labels.iterrows():
        if row["group"] not in GROUPS + ["other"]:
            problems.append(f"{row['file']}: group '{row['group']}' is not one of {GROUPS + ['other']}")
        if row["eval_type"] and row["eval_type"] not in EVAL_TYPES:
            problems.append(f"{row['file']}: eval_type '{row['eval_type']}' is not one of the 14 types")
        elif row["eval_type"] and EVAL_GROUP[row["eval_type"]] != row["group"]:
            problems.append(f"{row['file']}: {row['eval_type']} belongs to {EVAL_GROUP[row['eval_type']]}, "
                            f"not {row['group']}")
    if problems:
        sys.exit("label problems:\n  " + "\n  ".join(problems))
    return labels


def check_labels():
    labels = read_labels()
    print(f"labels OK: {len(labels)} photos")
    print("  per group:", labels["group"].value_counts().to_dict())
    typed = labels[labels["eval_type"] != ""]
    print(f"  with one of the 14 evaluation types: {len(typed)}", typed["eval_type"].value_counts().to_dict())


def open_photo(path):
    """Same as the API: respect the phone's rotation, convert to RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def sanity_check(retrievers):
    """Each evaluation method must reproduce its saved top 5 on 3 synthetic test queries."""
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str}).set_index("query_id")
    for name, retriever in retrievers.items():
        saved = pd.read_csv(f"results/{name}_phone_medium.csv", dtype={"item_id": str})
        for query_id in saved["query_id"].unique()[:3]:
            expected = list(saved[saved["query_id"] == query_id].sort_values("rank")["item_id"])
            found = [item for item, _ in retriever.search(Image.open(queries.loc[query_id, "image_path"]), k=5)]
            if found != expected:
                sys.exit(f"sanity check failed: {name} does not reproduce the saved top 5 of {query_id}")
        print(f"  check: {name} reproduces its saved top 5 on 3 synthetic queries")


def run():
    if os.path.exists(OUT_SUMMARY):
        sys.exit(f"{OUT_SUMMARY} exists: D9 has already been run (it is used once).")
    labels = read_labels()

    # the app exactly as shipped (must be set before serving.api is imported, which loads the models)
    os.environ["DAYFIT_CATALOG"] = "app"
    from serving import api
    from evaluation.run_eval import RETRIEVERS
    from evaluation.stats import bootstrap_ci
    from serving.badges import APP_GROUP_OF_TYPE

    # evaluation catalogue retrievers (plain similarity, as in every reported number)
    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    item_type = catalog.set_index("item_id")["article_type"]
    retrievers = {}
    for name in [PRIMARY] + DESCRIPTIVE:
        retrievers[name] = RETRIEVERS[name]()
        retrievers[name].build_index(catalog)

    # sanity checks BEFORE looking at any own photo
    sanity_check(retrievers)
    if len(api.engine.item) != 54178 or api.rerank_w != 0.05:
        sys.exit(f"sanity check failed: app has {len(api.engine.item)} items and w = {api.rerank_w}")
    print(f"  check: app catalogue {len(api.engine.item)} items, group bonus w = {api.rerank_w}")

    app_rows, type_rows, failed = [], [], []
    for _, label in labels.iterrows():
        try:
            image = open_photo(os.path.join(FOLDER, label["file"]))
        except Exception as error:                    # reported, never silently dropped
            failed.append(f"{label['file']}: {error}")
            continue

        # (1) the app as shipped
        out = api.search_photo(image)
        groups = [APP_GROUP_OF_TYPE[r["article_type"]] for r in out["results"]]
        predicted = out["query"]["predicted_group"]  # None if the badge is not shown (confidence < 0.9)
        app_rows.append(dict(
            file=label["file"], group=label["group"], eval_type=label["eval_type"],
            result_types="|".join(r["article_type"] for r in out["results"]),
            result_groups="|".join(groups),
            group_p5=np.mean([g == label["group"] for g in groups]) if label["group"] != "other" else np.nan,
            predicted_group=predicted, badge_shown=predicted is not None,
            badge_correct=float(predicted == label["group"]) if predicted is not None and label["group"] != "other"
            else np.nan,
            not_sure=out["not_sure"], shortlist_size=out["shortlist_size"],
            best_similarity=out["best_similarity"], not_in_catalogue=out["query"]["not_in_catalogue"]))

        # (2) type P@5 on the evaluation catalogue, only for the 14 types
        if label["eval_type"]:
            for name, retriever in retrievers.items():
                types = [item_type[item] for item, _ in retriever.search(image, k=5)]
                type_rows.append(dict(file=label["file"], eval_type=label["eval_type"], method=name,
                                      result_types="|".join(types),
                                      type_p5=np.mean([t == label["eval_type"] for t in types])))

    app = pd.DataFrame(app_rows)
    by_type = pd.DataFrame(type_rows)
    app.to_csv(OUT_APP, index=False)
    by_type.to_csv(OUT_TYPES, index=False)

    # summary: one row per number
    summary = []
    scored = app.dropna(subset=["group_p5"])
    low, high = bootstrap_ci(scored["group_p5"])
    summary.append(dict(result="(1) app group P@5", n=len(scored), value=scored["group_p5"].mean(),
                        ci_low=low, ci_high=high))
    for name in [PRIMARY] + DESCRIPTIVE:
        rows = by_type[by_type["method"] == name] if len(by_type) else by_type
        if len(rows):
            low, high = bootstrap_ci(rows["type_p5"])
            label = "(2) type P@5" if name == PRIMARY else "descriptive type P@5"
            summary.append(dict(result=f"{label} {name}", n=len(rows), value=rows["type_p5"].mean(),
                                ci_low=low, ci_high=high))
    shown = app[app["badge_shown"]]
    summary += [
        dict(result="app: not sure rate", n=len(app), value=app["not_sure"].mean()),
        dict(result="app: group badge shown", n=len(app), value=app["badge_shown"].mean()),
        dict(result="app: group badge correct when shown (excl. other)",
             n=int(shown["badge_correct"].notna().sum()), value=shown["badge_correct"].mean()),
        dict(result="app: out-of-catalogue warnings", n=len(app), value=app["not_in_catalogue"].notna().sum()),
        dict(result="app: shortlist size median", n=len(app), value=app["shortlist_size"].median()),
        dict(result="photos that failed to load", n=len(labels), value=len(failed)),
    ]
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nper group (app group P@5):")
    print(scored.groupby("group")["group_p5"].agg(["count", "mean"]).to_string(float_format=lambda x: f"{x:.3f}"))
    for line in failed:
        print("  FAILED TO LOAD:", line)
    print(f"\nsaved {OUT_APP}, {OUT_TYPES}, {OUT_SUMMARY}")


def main():
    parser = argparse.ArgumentParser()
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--template", action="store_true", help="write an empty labels.csv")
    step.add_argument("--check", action="store_true", help="check the labels (no model)")
    step.add_argument("--run", action="store_true", help="the one D9 run")
    args = parser.parse_args()
    if args.template:
        make_template()
    elif args.check:
        check_labels()
    else:
        run()


if __name__ == "__main__":
    main()
