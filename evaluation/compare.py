"""Is method B really better than method A, or is it just noise?

Uses the per-query files written by run_eval (results/perquery/) and a paired
bootstrap (see evaluation/stats.py). Give 2 or more methods; each one is
compared with the one before it, which is exactly the results-table "ladder".

Run from the repo root:
    python -m evaluation.compare fashion_clip fashion_siglip
    python -m evaluation.compare clip fashion_siglip fashion_siglip_mv ensemble --metric hit5
"""
import argparse
import os

import pandas as pd

from evaluation.stats import paired_bootstrap

NAMES = {"hit1": "Recall@1", "hit5": "Recall@5", "p5": "Precision@5"}


def load(method, condition, folder):
    path = os.path.join(folder, f"{method}_{condition}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="+")
    parser.add_argument("--metric", default="hit1", choices=list(NAMES))
    parser.add_argument("--conditions", default="phone_mild,phone_medium,phone_strong")
    parser.add_argument("--folder", default="results/perquery")
    parser.add_argument("--out", default="results/comparisons.csv",
                        help="every comparison is also saved here (re-running one replaces it)")
    args = parser.parse_args()

    rows = []
    for condition in args.conditions.split(","):
        for a, b in zip(args.methods, args.methods[1:]):
            old, new = load(a, condition, args.folder), load(b, condition, args.folder)
            if old is None or new is None:
                print(f"skipping {a} vs {b} on {condition}: run run_eval for both first")
                continue
            both = old.merge(new, on="query_id", suffixes=("_a", "_b"))  # same queries, same order
            both = both.dropna(subset=[f"{args.metric}_a", f"{args.metric}_b"])
            diff, low, high, p = paired_bootstrap(both[f"{args.metric}_b"], both[f"{args.metric}_a"])
            rows.append(dict(condition=condition, change=f"{a} -> {b}",
                             before=round(both[f"{args.metric}_a"].mean(), 3),
                             after=round(both[f"{args.metric}_b"].mean(), 3),
                             difference=f"{diff:+.3f} [{low:+.3f},{high:+.3f}]",
                             p_value=round(p, 4),
                             significant="yes" if p < 0.05 else "no"))

    table = pd.DataFrame(rows)
    print(f"\nPaired bootstrap on {NAMES[args.metric]} (95% CI of the difference):\n")
    print(table.to_string(index=False))

    # save for the report: keep earlier comparisons, replace any that were re-run
    if len(table):
        table.insert(0, "metric", NAMES[args.metric])
        if os.path.exists(args.out):
            old = pd.read_csv(args.out)
            key = ["metric", "condition", "change"]
            old = old[~old.set_index(key).index.isin(table.set_index(key).index)]
            table = pd.concat([old, table], ignore_index=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
