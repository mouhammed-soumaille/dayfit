"""E2: fix the app's two uncertainty constants, calibrated on REAL photos (LookBench).

1. Shortlist ("the right item is very likely among these N"): relative split conformal
   (evaluation/conformal.py). For each LookBench photo, gap = (best similarity) - (similarity
   to its true item). q = the ceil((n+1)(1-alpha))-th smallest gap, alpha = 0.1. In the app,
   the shortlist = every catalogue item whose similarity is within q of the best one.
2. "Not sure" (risk control, evaluation/conformal_extra.py): the app shows its top 5 only if
   the best similarity is >= lambda. Loss of a photo = share of the top 5 in the wrong GROUP
   (Finding 19: exact category names are too strict); alpha = 0.1. (0.2 was vacuous: the
   model's group loss is already 0.13 without abstention, so the app would almost never
   abstain.) lambda = the smallest threshold whose corrected calibration risk
   n/(n+1) * risk + 1/(n+1) is <= alpha.
   (The exact-name lambda is saved too, for comparison with Finding 11.)

Decision 1a (Nov 18): calibrate on LookBench real photos, because the kind of photo matters
most (Finding 10). LIMITATION, stated in the app and the report: LookBench uses a different
gallery from our catalogue, so neither guarantee formally holds in the app; D9 (own photos)
checks the "not sure" behaviour on our catalogue.

For transparency we also re-estimate, with 200 random 50/50 splits of LookBench, what these
rules achieve there (coverage, median shortlist size, answer rate, loss).

Sanity checks:
  - each photo's best similarity equals the rank-1 score saved by run_lookbench;
  - with exact category names and no abstention, the loss equals the saved Finding 11 value.

Run from the repo root (embeds the 981 LookBench photos, about the time of one run_lookbench):
    python -m serving.calibrate
"""
import json
import os

import numpy as np
import pandas as pd

from evaluation.conformal import relative_q
from evaluation.conformal_extra import pick_lambda, risk_control
from evaluation.conformal_shift import lookbench_scores
from evaluation.group_p5 import GROUP_OF, GROUP_OF_GALLERY
from evaluation.run_eval import RETRIEVERS
from evaluation.run_lookbench import prepare, use_index_dir

METHOD = "finetuned_random"
ALPHA_SET, ALPHA_RISK = 0.1, 0.1     # alpha_risk 0.2 was vacuous with group loss (Nov 18)
RISK_TABLE = [0.2, 0.1, 0.05]
TRIALS = 200
OUT = "serving/calibration.json"


def main():
    gallery, queries = prepare("real_streetlook")
    queries = queries[queries["label"].isin(gallery["label"])].reset_index(drop=True)
    retriever = RETRIEVERS[METHOD]()
    use_index_dir(retriever, os.path.join("models", "index_lookbench", "real_streetlook"))
    retriever.build_index(gallery)
    print("embedding LookBench street photos ...")
    scores, true = lookbench_scores(retriever, gallery, queries)
    best = scores.max(axis=1)
    n = len(scores)

    # check 1: best similarity = saved rank-1 score
    ranks = pd.read_csv(f"results/rankings/{METHOD}_lookbench_real_streetlook_image_dedup.csv")
    saved_best = ranks[ranks["rank"] == 1].set_index("query_id")["score"].loc[queries["query_id"]]
    ok1 = np.allclose(best, saved_best.values, atol=1e-5)
    print(f"\n  check: best similarity equals the saved rank-1 score for all {n} photos -> {'OK' if ok1 else 'MISMATCH'}")

    # check 2: exact-name loss without abstention = Finding 11
    exact_q, exact_g = queries["category"].to_numpy(), gallery["category"].to_numpy()
    top5 = np.argsort(-scores, axis=1)[:, :5]
    exact_wrong = (exact_g[top5] != exact_q[:, None]).mean(axis=1)
    saved = pd.read_csv("results/conformal_risk.csv")
    saved_loss = saved[(saved["model"] == METHOD) & (saved["alpha"] == "no abstention")]["loss"].item()
    ok2 = np.isclose(exact_wrong.mean(), saved_loss)
    print(f"  check: exact-name loss {exact_wrong.mean():.4f} vs saved {saved_loss:.4f} -> {'OK' if ok2 else 'MISMATCH'}")
    if not (ok1 and ok2):
        raise SystemExit("sanity check failed, not saving")

    # the two constants, from ALL photos
    gaps = best - true
    q = relative_q(gaps, ALPHA_SET)
    group_q = np.array([GROUP_OF[c.lower()] for c in exact_q])
    group_g = np.array([GROUP_OF_GALLERY[c.lower()] for c in exact_g])
    group_wrong = (group_g[top5] != group_q[:, None]).mean(axis=1)
    lam_group = pick_lambda(best, group_wrong, ALPHA_RISK)
    lam_exact = pick_lambda(best, exact_wrong, ALPHA_RISK)

    # what these rules achieve on LookBench (200 random 50/50 splits)
    rng = np.random.default_rng(0)
    coverage, sizes = [], []
    for _ in range(TRIALS):
        order = rng.permutation(n)
        cal, test = order[:n // 2], order[n // 2:]
        q_cal = relative_q(gaps[cal], ALPHA_SET)
        coverage.append(np.mean(gaps[test] <= q_cal))
        sizes.append(np.median((scores[test] >= best[test, None] - q_cal).sum(axis=1)))
    risk = risk_control(scores, group_q, group_g, TRIALS, RISK_TABLE, rng)
    print("\nrisk control with group loss on LookBench (200 random 50/50 splits):")
    print(risk.round(3).to_string(index=False))
    at_alpha = risk[risk["alpha"] == ALPHA_RISK].iloc[0]

    calibration = dict(
        method=METHOD, source="LookBench real_streetlook (all photos)", n_calibration=int(n),
        shortlist=dict(alpha=ALPHA_SET, q=float(q),
                       lookbench_coverage=float(np.mean(coverage)),
                       lookbench_median_size=float(np.median(sizes))),
        not_sure=dict(alpha=ALPHA_RISK, loss="share of top 5 in the wrong category group",
                      **{"lambda": float(lam_group)}, lambda_exact_names=float(lam_exact),
                      lookbench_answer_rate=float(at_alpha["answer_rate"]),
                      lookbench_loss=float(at_alpha["loss"]),
                      lookbench_group_p5_when_answered=float(at_alpha["p5_when_answered"])),
        limitation="Calibrated on LookBench photos and gallery; the app searches a different "
                   "catalogue, so the guarantees do not formally hold in the app.",
        lookbench_risk_table=risk.astype({"alpha": str}).to_dict(orient="records"))
    with open(OUT, "w") as f:
        json.dump(calibration, f, indent=2)
    print(json.dumps(calibration, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()