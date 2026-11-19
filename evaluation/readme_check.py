"""Check that the numbers in README.md match the saved result files (task 6).

Problem: the README is written by hand, so a number can be mistyped or go stale when a result changes.
For every number in the README that comes from a results file, this script recomputes it from the file,
formats it the way the README writes it, and checks that exactly that text appears in README.md.
Numbers that only exist in report/results_log.md (no CSV) are listed separately so they can be checked by eye.

Run from the repo root (reads files only, writes nothing):
    python -m evaluation.readme_check
"""
import json

import pandas as pd

LB = "data/lookbench/real_streetlook"


def test_half(path, column):
    """Mean of one per-query column over the TEST half of the frozen split."""
    split = pd.read_csv(f"{LB}/split.csv").set_index("query_id")
    values = pd.read_csv(path).set_index("query_id")[column]
    return values.loc[split.index[split["side"] == "test"]].mean()


def claims():
    """(what, text expected in the README) for every number that comes from a results file."""
    out = []
    c = pd.read_csv("results/classical_baseline.csv").set_index("method")
    out += [("classical: colour only R@1 / R@5", f"| {c.loc['colour only (masked)', 'R1']:.3f} | {c.loc['colour only (masked)', 'R5']:.3f} |"),
            ("classical R@1 / R@5", f"| {c.loc['classical', 'R1']:.3f} | {c.loc['classical', 'R5']:.3f} |"),
            ("classical + menu R@1", f"| {c.loc['classical + menu', 'R1']:.3f} |"),
            ("photo + menu R@1", f"| {c.loc['photo + menu (reference)', 'R1']:.3f} |")]
    full = "results/perquery/{}_lookbench_real_streetlook_{}_full.csv"
    out += [("pretrained, official corpus, test R@1 / R@5",
             f"| {test_half(full.format('fashion_siglip', 'image'), 'hit1'):.3f} | {test_half(full.format('fashion_siglip', 'image'), 'hit5'):.3f} |"),
            ("fine-tuned, official corpus, test R@1 / R@5",
             f"| **{test_half(full.format('finetuned_random', 'image'), 'hit1'):.3f}** | **{test_half(full.format('finetuned_random', 'image'), 'hit5'):.3f}** |"),
            ("whole photo, official corpus, test R@1 / R@5",
             f"| {test_half(full.format('finetuned_random', 'raw_image'), 'hit1'):.3f} | {test_half(full.format('finetuned_random', 'raw_image'), 'hit5'):.3f} |")]
    r = pd.read_csv("results/rerank_test.csv").iloc[1]
    out += [("re-ranker R@1, difference, p", f"| {r['R1']:.3f} (+{100 * r['R1_difference']:.1f}, not significant, p = {r['R1_p']:.2f}) | {r['R5']:.3f} |")]
    k = pd.read_csv("results/keyword_baseline.csv")
    k = k[(k["side"] == "test") & k["metric"].isna()].set_index("method")
    out += [("keywords short R@1 / group P@5", f"| {k.loc['keywords_basic', 'hit1']:.3f} | {k.loc['keywords_basic', 'gp5']:.3f} |"),
            ("keywords detailed R@1 / group P@5", f"| {k.loc['keywords_detailed', 'hit1']:.3f} | {k.loc['keywords_detailed', 'gp5']:.3f} |"),
            ("photo on the gallery R@1 / group P@5",
             f"| **{k.loc['finetuned_random (photo)', 'hit1']:.3f}** | **{k.loc['finetuned_random (photo)', 'gp5']:.3f}** |")]
    f = pd.read_csv("results/fusion_test.csv").set_index(["metric", "method"])
    out += [("group bonus R@1 / group P@5", f"| {f.loc[('hit1', 'groupbonus'), 'value']:.3f} | {f.loc[('gp5', 'groupbonus'), 'value']:.3f} |")]
    lf = pd.read_csv("results/lookbench_full.csv")
    crop_all = lf[(lf["method"] == "finetuned_random") & (lf["view"] == "image") & lf["side"].isna()]["R1"].item()
    out += [("official corpus, all 981, crop R@1", f"0.520 -> {crop_all:.3f}")]
    rb = pd.read_csv("results/robustness.csv").set_index(["method", "condition"])
    tests = pd.read_csv("results/robustness_tests.csv")
    tests = tests[tests["metric"] == "hit1"].set_index("condition")["difference"]
    def r1(m, c):
        return f"{rb.loc[(m, c), 'R1']:.3f}"
    def gain(c):
        return f"{100 * tests[c]:+.1f}"
    for label, conditions in [("blur", ["gauss_1.5", "gauss_3", "gauss_6"]), ("motion", ["motion_5", "motion_9", "motion_15"]),
                              ("dark", ["dark_0.5", "dark_0.3", "dark_0.15"])]:
        out += [(f"robustness {label}, fine-tuned", " / ".join(r1("finetuned_random", c) for c in conditions)),
                (f"robustness {label}, pretrained", " / ".join(r1("fashion_siglip", c) for c in conditions)),
                (f"robustness {label}, gain", " / ".join(gain(c) for c in conditions))]
    out += [("robustness grayscale", f"| {r1('finetuned_random', 'grayscale')} | {r1('fashion_siglip', 'grayscale')} | {gain('grayscale')}"),
            ("robustness casts", f"| {r1('finetuned_random', 'cast_warm_mild')} / {r1('finetuned_random', 'cast_warm_strong')} | "
                                 f"{r1('fashion_siglip', 'cast_warm_mild')} / {r1('fashion_siglip', 'cast_warm_strong')} | "
                                 f"{gain('cast_warm_mild')} / {gain('cast_warm_strong')}"),
            ("robustness clean", f"| none | {r1('finetuned_random', 'clean')} | {r1('fashion_siglip', 'clean')} | {gain('clean')} |"),
            ("strong blur R@1 in limitations", f"strong blur: R@1 {rb.loc[('finetuned_random', 'gauss_6'), 'R1']:.2f}"),
            ("not sure on strong blur", f"on {100 * rb.loc[('finetuned_random', 'gauss_6'), 'not_sure']:.0f}% of such photos"),
            ("dark photos, wrong colour badge after the fix",
             f"({100 * rb.loc[('finetuned_random', 'dark_0.15'), 'true_badge_different_after']:.0f}-"
             f"{100 * rb.loc[('finetuned_random', 'dark_0.3'), 'true_badge_different_after']:.0f}% for the true")]
    cal = pd.read_csv("results/calibration_full.csv").set_index("view")
    out += [("shortlist length, crop", f"about {cal.loc['crop', 'median_shortlist_full']:.0f}\n  items long"),
            ("shortlist length, whole photo (rounded to 10)", f"for a whole photo about {round(cal.loc['whole', 'median_shortlist_full'], -1):.0f}")]
    lat = pd.read_csv("results/latency_cpu.csv").set_index(["catalogue", "type_model"])
    on, off = lat.loc[("app", "on"), "median_ms"], lat.loc[("app", "off"), "median_ms"]
    out += [("latency app, type model on, and its share", f"{on:.0f} ms for the 54k-item app\ncatalogue, of which {on - off:.0f} ms")]
    badge = [b for b in json.load(open("serving/outside_app.json"))["group_badge"] if b["source"] == "lookbench" and b["min_confidence"] == 0.9][0]
    out += [("type badge on real photos", f"appears for {100 * badge['shown']:.0f}% of photos and is right\n  {100 * badge['accuracy_when_shown']:.0f}% of the time")]
    s = pd.read_csv("results/summary.csv").set_index(["method", "condition"])
    out += [("synthetic R@1 medium / strong", f"| {s.loc[('finetuned_random', 'phone_medium'), 'Recall@1']:.3f} / {s.loc[('finetuned_random', 'phone_strong'), 'Recall@1']:.3f} |"),
            ("synthetic P@5 medium / strong", f"| {s.loc[('finetuned_random', 'phone_medium'), 'Precision@5']:.3f} / {s.loc[('finetuned_random', 'phone_strong'), 'Precision@5']:.3f} |"),
            ("synthetic gain over pretrained (medium)",
             f"+{100 * (s.loc[('finetuned_random', 'phone_medium'), 'Recall@1'] - s.loc[('fashion_siglip', 'phone_medium'), 'Recall@1']):.1f} points here")]
    app = pd.read_csv("data/app_catalog.csv", dtype=str)
    hidden = pd.read_csv("serving/app_hidden.csv", dtype=str)
    kids = set(hidden.loc[hidden["reasons"].str.contains("kids_name"), "item_id"]) | set(app.loc[app["gender"].isin(["Boys", "Girls"]), "item_id"])
    counts = app["source"].value_counts()
    out += [("app catalogue size", f"{len(app):,} items"), ("extra Kaggle items", f"{counts['extra']:,} more adult items"),
            ("H&M items", f"{counts['hm']:,} newer H&M"), ("hidden kids' items", f"hides {len(kids)} kids' items")]
    return out


FROM_LOG_ONLY = ["266 ms Docker CPU and 36 ms Mac GPU (Findings 26, 33, 45)", "+2 to +3 points clean, +6 to +21 blurred "
                 "(Finding 40 / 38, covered by the robustness rows above)", "none of the jeans are wide-leg (Finding 30)",
                 "H&M label noise (Finding 36)", "621 groups, 500 / 481 split (results_log pre-registration)"]


def main():
    readme = open("README.md").read()
    problems = 0
    for what, text in claims():
        ok = text in readme
        problems += not ok
        print(f"  {'OK      ' if ok else 'MISMATCH'} {what}: expected text {text!r}")
    print(f"\n{problems} mismatch(es)")
    print("Numbers taken from report/results_log.md only (no CSV), check by eye:")
    for line in FROM_LOG_ONLY:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
