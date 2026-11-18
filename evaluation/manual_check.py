"""D8b: material for a MANUAL check of LookBench queries (finetuned_random, dedup top 5).

Two samples, fixed before looking:
  dress_failure : all LookBench dress/jumpsuit queries with P@5 <= 0.4 (the 31 of Finding 17)
  random        : 30 random other queries (seed 1), to see how common the problems are overall
For each query, one row of a contact sheet: the FULL original photo, LookBench's crop (what the
model sees), then the top 5 with the usual borders (green exact, orange category, red wrong).

The human verdicts go in results/manual_check.csv (this script never overwrites that file):
  label_check : correct   = the crop shows a garment of the labelled category
                wrong     = the crop clearly shows another kind of garment (e.g. "dress" shows a top)
                ambiguous = several garments / cannot tell which one is meant
  metric_fair : yes = a shopper would agree with how the metric scores these top 5
                no  = the metric is clearly unfair (e.g. correct garment scored as wrong category)
  note        : optional, a few words

Run from the repo root:
    python -m evaluation.manual_check
"""
import os

import matplotlib
matplotlib.use("Agg")                       # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from evaluation.failure_gallery import COLOURS, LB, check, lookbench_rows, per_query

METHOD = "finetuned_random"
N_RANDOM = 30
ROWS_PER_SHEET = 10
OUT_DIR = "results/manual_check"
CSV = "results/manual_check.csv"


def main():
    ranks, saved = lookbench_rows(METHOD)
    check(ranks, saved, "lookbench")                 # same sanity check as the failure gallery
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).set_index("query_id")
    q = per_query(ranks)

    dress = q[(q["group"] == "dresses") & (q["p5"] <= 0.4)].index
    rest = q.index.difference(dress)
    rng = np.random.default_rng(1)
    random = rest[np.sort(rng.choice(len(rest), size=N_RANDOM, replace=False))]
    sample = pd.DataFrame({"query_id": list(dress) + list(random),
                           "set": ["dress_failure"] * len(dress) + ["random"] * len(random)})
    sample["row"] = range(1, len(sample) + 1)
    sample = sample.merge(q[["category", "hit1", "p5"]], left_on="query_id", right_index=True)
    print(f"  {len(dress)} dress failures + {len(random)} random queries = {len(sample)} rows")

    # contact sheets: ROWS_PER_SHEET queries per picture
    os.makedirs(OUT_DIR, exist_ok=True)
    for start in range(0, len(sample), ROWS_PER_SHEET):
        part = sample.iloc[start:start + ROWS_PER_SHEET]
        fig, axes = plt.subplots(len(part), 7, figsize=(14, 2.1 * len(part)), squeeze=False)
        for r, s in enumerate(part.itertuples()):
            axes[r, 0].imshow(Image.open(queries.loc[s.query_id, "raw_image"]).convert("RGB"))
            axes[r, 0].set_title(f"row {s.row} ({s.set})\nfull photo", fontsize=7)
            axes[r, 1].imshow(Image.open(queries.loc[s.query_id, "image"]).convert("RGB"))
            axes[r, 1].set_title(f"crop, label: {s.category}", fontsize=7)
            top = ranks[ranks["query_id"] == s.query_id].sort_values("rank")
            for c, t in enumerate(top.itertuples(), start=2):
                kind = "exact" if t.exact else ("category" if t.same_cat else "wrong")
                axes[r, c].imshow(Image.open(t.result_path).convert("RGB"))
                for side in axes[r, c].spines.values():
                    side.set_edgecolor(COLOURS[kind])
                    side.set_linewidth(4)
                axes[r, c].set_title(f"{t.rank}. {kind}", fontsize=7)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        name = f"{OUT_DIR}/sheet_{start // ROWS_PER_SHEET + 1}.png"
        fig.savefig(name, dpi=90)
        plt.close(fig)
        print(f"  saved {name}")

    # empty verdict file, only if it does not exist yet (never overwrite human verdicts)
    if os.path.exists(CSV):
        print(f"  {CSV} already exists, left untouched")
    else:
        sample["label_check"] = ""
        sample["metric_fair"] = ""
        sample["note"] = ""
        sample[["row", "query_id", "set", "category", "hit1", "p5",
                "label_check", "metric_fair", "note"]].to_csv(CSV, index=False)
        print(f"  saved empty verdict file {CSV}")


if __name__ == "__main__":
    main()