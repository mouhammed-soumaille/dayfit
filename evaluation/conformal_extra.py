"""T3 and T4 on real photos (LookBench, calibrated on real photos).

T3, per-category (Mondrian) conformal:
  Standard conformal promises 90% coverage on AVERAGE over all queries; some garment
  categories can be far below 90% while others are above. Mondrian conformal calibrates
  each category separately (only on calibration queries of the same category), which
  gives the 90% guarantee inside EVERY category. Cost: fewer calibration queries per
  category, so larger sets. Relative variant, alpha = 0.1.

T4, conformal risk control on Precision@5 (Angelopoulos et al., 2022):
  The app shows the top 5 only when it is confident (top similarity >= lambda), and
  otherwise says "not sure". Loss of one query = share of wrong-category items among
  the 5 shown, or 0 if the app abstained. The loss can only go down as lambda goes up
  and lies in [0, 1], so choosing
      lambda = smallest value with  n/(n+1) * (mean calibration loss) + 1/(n+1) <= alpha
  guarantees  E[loss on a new query] <= alpha:
  on average at most alpha of what we show is the wrong kind of garment
  (abstaining counts as no error). We also report how often the app still answers.

Run from the repo root:
    python -m evaluation.conformal_extra --method finetuned_random
"""
import argparse
import os

import numpy as np
import pandas as pd

from evaluation.conformal import relative_q
from evaluation.conformal_shift import lookbench_scores, save


def mondrian(scores, true, categories, trials, alpha, rng):
    """Coverage and set size per category: one shared calibration vs per-category calibration."""
    top = scores.max(axis=1)
    gaps = top - true                       # relative score: how far below the top result
    n = len(scores)
    found = {}                              # (category, variant) -> (covered flags, set sizes)
    for _ in range(trials):
        order = rng.permutation(n)
        cal, test = order[:n // 2], order[n // 2:]
        q_all = relative_q(gaps[cal], alpha)
        for cat in np.unique(categories):
            t = test[categories[test] == cat]
            if len(t) == 0:
                continue
            q_cat = relative_q(gaps[cal[categories[cal] == cat]], alpha)  # inf if too few queries
            for variant, q in [("standard", q_all), ("mondrian", q_cat)]:
                cutoff = top[t] - q
                covered, sizes = found.setdefault((cat, variant), ([], []))
                covered.extend(true[t] >= cutoff)
                sizes.extend((scores[t] >= cutoff[:, None]).sum(axis=1))

    rows = []
    for cat in np.unique(categories):
        row = dict(category=cat, n_queries=int((categories == cat).sum()))
        for variant in ["standard", "mondrian"]:
            covered, sizes = found[(cat, variant)]
            row[f"coverage_{variant}"] = np.mean(covered)
            row[f"median_size_{variant}"] = np.median(sizes)
        rows.append(row)
    return pd.DataFrame(rows)


def pick_lambda(conf_cal, wrong_cal, alpha):
    """Smallest confidence threshold whose corrected calibration risk is <= alpha."""
    n = len(conf_cal)
    for lam in np.sort(conf_cal):           # from low (answer often) to high (answer rarely)
        risk = np.mean((conf_cal >= lam) * wrong_cal)
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return lam
    return np.inf                           # never answer


def risk_control(scores, query_cat, gallery_cat, trials, alphas, rng):
    """T4: answer only when confident; guarantee average share of wrong-category results <= alpha."""
    top5 = np.argsort(-scores, axis=1)[:, :5]
    wrong = (gallery_cat[top5] != query_cat[:, None]).mean(axis=1)   # = 1 - category P@5
    conf = scores.max(axis=1)
    n = len(scores)
    rows = [dict(alpha="no abstention", loss=wrong.mean(), answer_rate=1.0,
                 p5_when_answered=1 - wrong.mean())]
    for alpha in alphas:
        res = []
        for _ in range(trials):
            order = rng.permutation(n)
            cal, test = order[:n // 2], order[n // 2:]
            lam = pick_lambda(conf[cal], wrong[cal], alpha)
            answer = conf[test] >= lam
            res.append(dict(loss=np.mean(answer * wrong[test]), answer_rate=answer.mean(),
                            p5_when_answered=1 - wrong[test][answer].mean() if answer.any() else np.nan))
        rows.append(dict(alpha=alpha, **{k: float(np.nanmean([r[k] for r in res])) for k in res[0]}))
    return pd.DataFrame(rows)


def main():
    from evaluation.run_eval import RETRIEVERS
    from evaluation.run_lookbench import prepare, use_index_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="finetuned_random", choices=list(RETRIEVERS))
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    gallery, lb_queries = prepare("real_streetlook")
    lb_queries = lb_queries[lb_queries["label"].isin(gallery["label"])].reset_index(drop=True)
    retriever = RETRIEVERS[args.method]()
    use_index_dir(retriever, os.path.join("models", "index_lookbench", "real_streetlook"))
    retriever.build_index(gallery)
    print("embedding LookBench street photos ...")
    scores, true = lookbench_scores(retriever, gallery, lb_queries)
    query_cat = lb_queries["category"].to_numpy()
    gallery_cat = gallery["category"].to_numpy()

    rng = np.random.default_rng(0)
    mond = mondrian(scores, true, query_cat, args.trials, 0.1, rng)
    risk = risk_control(scores, query_cat, gallery_cat, args.trials, [0.2, 0.1, 0.05], rng)
    mond.insert(0, "model", args.method)
    risk.insert(0, "model", args.method)
    save(mond, "results/conformal_mondrian.csv", args.method)
    save(risk, "results/conformal_risk.csv", args.method)

    pd.set_option("display.width", 160)
    print("\n\nT3: per-category coverage, target 90% (relative variant, real -> real):\n")
    print(mond.drop(columns="model").round(3).to_string(index=False))
    print("\nT4: risk control on wrong-category share among shown results (real -> real):\n")
    print(risk.drop(columns="model").round(3).to_string(index=False))


if __name__ == "__main__":
    main()