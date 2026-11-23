"""Final figures for the README and the report, made only from saved results (no new experiment).

  fig15_headline_real.png  exact-item R@1 on the LookBench TEST half: the photo model against every baseline
                           (left: official 61,553-image corpus, right: the 3,278 shop items only)
  fig16_crop_vs_whole.png  garment marked (crop) vs whole photo, test half, official corpus
  fig17_d9.png             D9 own wardrobe: the app's group P@5 per group, and type P@5 per type
  fig18_d9_gallery.png     D9 examples: the app's top 5 for 6 of my photos, chosen by a fixed rule
                           (the first photo of each group in file order + the photo with the lowest group P@5;
                           IMG_4237 is skipped: it is not a garment, see the correction to Finding 46)

Every number is read from results/ (or recomputed from the per-query files) and checked against the value
in the README before plotting. Error bars: 95% bootstrap CI over photos (evaluation/stats.py), the same
method as the saved tables.

The gallery runs the app again on the 6 chosen photos only to get the result images; it checks that the
result groups are exactly the ones saved by the D9 run (results/d9_app.csv). Nothing is re-scored.

Run from the repo root (needs the venv with torch for the gallery):
    python -m report.make_final_figures
    python -m report.make_final_figures --no_gallery     (the three charts only, no model)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.stats import bootstrap_ci

OUT = "report/figures"
PER = "results/perquery"
SPLIT = "data/lookbench/real_streetlook/split.csv"
BLUE, ORANGE, GREY, GREEN, RED = "#2b78d6", "#e8673a", "#9a9a9a", "#2e9e4f", "#d23b3b"


def check(name, value, expected):
    """Stop if a number does not match the README value it is supposed to show."""
    if round(value, 3) != expected:
        raise SystemExit(f"check failed: {name} = {value:.4f}, README says {expected}")


def test_ids():
    split = pd.read_csv(SPLIT)
    return set(split.loc[split["side"] == "test", "query_id"])


def perquery_test(file, column="hit1"):
    """Per-photo values of one method on the test half, with mean and 95% CI."""
    per = pd.read_csv(os.path.join(PER, file))
    values = per[per["query_id"].isin(test_ids())][column].to_numpy()
    low, high = bootstrap_ci(values)
    return values.mean(), low, high


def bars(ax, rows, title, xmax=0.7):
    """Horizontal bars (best at the bottom), value and CI written next to each bar."""
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    for yi, (name, value, low, high, colour) in zip(y, rows):
        ax.barh(yi, value, color=colour, height=0.6)
        ax.errorbar(value, yi, xerr=[[value - low], [high - value]], color="black", capsize=3, lw=1)
        ax.text(high + 0.01, yi, f"{value:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_headline():
    classical = pd.read_csv("results/classical_baseline.csv").set_index("method")
    keywords = pd.read_csv("results/keyword_baseline.csv")
    keywords = keywords[keywords["side"] == "test"].set_index("method")

    colour = classical.loc["colour only (masked)"]
    classic = classical.loc["classical"]
    pre = perquery_test("fashion_siglip_lookbench_real_streetlook_image_full.csv")
    ft = perquery_test("finetuned_random_lookbench_real_streetlook_image_full.csv")
    check("colour histogram R@1", colour["R1"], 0.081)
    check("classical R@1", classic["R1"], 0.123)
    check("pretrained R@1", pre[0], 0.476)
    check("fine-tuned R@1", ft[0], 0.499)
    left = [("Colour histogram", colour["R1"], colour["R1_lo"], colour["R1_hi"], GREY),
            ("Classical image search\n(colour + texture + shape)", classic["R1"], classic["R1_lo"],
             classic["R1_hi"], GREY),
            ("Pretrained FashionSigLIP", *pre, ORANGE),
            ("Fine-tuned (the app's model)", *ft, BLUE)]

    cmenu = classical.loc["classical + menu"]
    pmenu = classical.loc["photo + menu (reference)"]
    basic, detailed = keywords.loc["keywords_basic"], keywords.loc["keywords_detailed"]
    photo = keywords.loc["finetuned_random (photo)"]
    check("classical + menu R@1", cmenu["R1"], 0.237)
    check("short keywords R@1", basic["hit1"], 0.312)
    check("idealised keywords R@1", detailed["hit1"], 0.486)
    check("photo (gallery) R@1", photo["hit1"], 0.511)
    check("photo + menu R@1", pmenu["R1"], 0.536)
    right = [("Classical image search\n+ an ideal type menu", cmenu["R1"], cmenu["R1_lo"], cmenu["R1_hi"], GREY),
             ("Keyword search, short\n(\"bag backpack\")", basic["hit1"], basic["hit1_lo"], basic["hit1_hi"], GREY),
             ("Keyword search, every attribute\n(idealised shopper)", detailed["hit1"], detailed["hit1_lo"],
              detailed["hit1_hi"], GREY),
             ("Photo (fine-tuned)", photo["hit1"], photo["hit1_lo"], photo["hit1_hi"], BLUE),
             ("Photo + an ideal type menu", pmenu["R1"], pmenu["R1_lo"], pmenu["R1_hi"], BLUE)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bars(axes[0], left, "Against the official corpus (61,553 images)")
    bars(axes[1], right, "Against the shop's 3,278 items only")
    fig.suptitle("Exact item first (R@1) on 481 held-out real street photos (LookBench test half, garment marked)",
                 fontsize=11)
    fig.text(0.5, 0.01, "Error bars: 95% bootstrap CI over photos. Grey: no deep learning or text; "
             "blue / orange: photo models.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{OUT}/fig15_headline_real.png", dpi=150)
    plt.close(fig)


def fig_crop_vs_whole():
    rows = {}
    for model, file in [("Fine-tuned", "finetuned_random"), ("Pretrained", "fashion_siglip")]:
        for view, suffix in [("Garment marked (crop)", "image_full"), ("Whole photo", "raw_image_full")]:
            for metric in ("hit1", "hit5"):
                rows[(model, view, metric)] = perquery_test(f"{file}_lookbench_real_streetlook_{suffix}.csv", metric)
    check("fine-tuned crop R@1", rows[("Fine-tuned", "Garment marked (crop)", "hit1")][0], 0.499)
    check("fine-tuned crop R@5", rows[("Fine-tuned", "Garment marked (crop)", "hit5")][0], 0.688)
    check("fine-tuned whole R@1", rows[("Fine-tuned", "Whole photo", "hit1")][0], 0.391)
    check("fine-tuned whole R@5", rows[("Fine-tuned", "Whole photo", "hit5")][0], 0.530)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "hit1", "Exact item first (R@1)"),
                              (axes[1], "hit5", "Exact item in the top 5 (R@5)")]:
        x = np.arange(2)
        for k, (model, colour) in enumerate([("Pretrained", ORANGE), ("Fine-tuned", BLUE)]):
            values = [rows[(model, view, metric)] for view in ("Garment marked (crop)", "Whole photo")]
            means = [v[0] for v in values]
            errors = [[v[0] - v[1] for v in values], [v[2] - v[0] for v in values]]
            ax.bar(x + (k - 0.5) * 0.36, means, width=0.36, color=colour, label=model,
                   yerr=errors, capsize=3, error_kw=dict(lw=1))
            for xi, v in zip(x, values):                # label above the error bar, not on it
                ax.text(xi + (k - 0.5) * 0.36, v[2] + 0.015, f"{v[0]:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Garment marked (crop)", "Whole photo"])
        ax.set_ylim(0, 0.85)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("The app sends the whole photo unless you mark the garment (481 test photos, 61,553-image corpus)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/fig16_crop_vs_whole.png", dpi=150)
    plt.close(fig)


def fig_d9():
    app = pd.read_csv("results/d9_app.csv")
    app = app[app["group"] != "other"]
    types = pd.read_csv("results/d9_types.csv", dtype={"eval_type": str})
    ft = types[types["method"] == "finetuned_random"]
    check("D9 app group P@5", app["group_p5"].mean(), 0.985)
    check("D9 type P@5", ft["type_p5"].mean(), 0.745)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.5]))
    # left: the app, right kind of garment, per group
    order = ["tops", "bottoms", "outerwear", "footwear", "accessories"]
    per = app.groupby("group")["group_p5"].agg(["mean", "count"]).loc[order]
    low, high = bootstrap_ci(app["group_p5"].to_numpy())
    ax = axes[0]
    labels = [f"All\n(n={len(app)})"] + [f"{g}\n(n={n})" for g, n in zip(order, per["count"])]
    values = [app["group_p5"].mean()] + list(per["mean"])
    ax.bar(range(len(values)), values, color=[BLUE] + [GREY] * len(order))
    ax.errorbar(0, values[0], yerr=[[values[0] - low], [high - values[0]]], color="black", capsize=3, lw=1)
    ax.text(0, high + 0.02, f"{values[0]:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(values[1:], start=1):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("The app as shipped: right kind of garment in the top 5", fontsize=10)

    # right: fine-tuned model, same article type, per type (evaluation catalogue)
    per_type = ft.groupby("eval_type")["type_p5"].agg(["mean", "count"]).sort_values("mean")
    ax = axes[1]
    labels = [f"All\n(n={len(ft)})"] + [f"{t.replace(' ', chr(10))}\n(n={n})"
                                          for t, n in zip(per_type.index, per_type["count"])]
    values = [ft["type_p5"].mean()] + list(per_type["mean"])
    low, high = bootstrap_ci(ft["type_p5"].to_numpy())
    ax.bar(range(len(values)), values, color=[BLUE] + [GREY] * len(per_type))
    ax.errorbar(0, values[0], yerr=[[values[0] - low], [high - values[0]]], color="black", capsize=3, lw=1)
    ax.text(0, high + 0.02, f"{values[0]:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(values[1:], start=1):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_title("Fine-tuned model: same article type in the top 5 (6,197-item catalogue)", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("D9: my own wardrobe, 83 photos, whole photo, run once. Error bar: 95% bootstrap CI; "
                 "groups and types with few photos are descriptive only", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/fig17_d9.png", dpi=150)
    plt.close(fig)


def fig_gallery():
    os.environ["DAYFIT_CATALOG"] = "app"            # the app exactly as in the D9 run
    from evaluation.d9_own_photos import FOLDER, open_photo
    from serving import api
    from serving.badges import APP_GROUP_OF_TYPE

    app = pd.read_csv("results/d9_app.csv")
    # IMG_4237 is not a garment (taken by mistake, see the correction to Finding 46): never shown as an example
    scored = app[(app["group"] != "other") & (app["file"] != "IMG_4237.HEIC")]
    # fixed rule, not chosen by looking: first photo of each group in file order, then the lowest group P@5
    picks = [scored[scored["group"] == g].sort_values("file").iloc[0]
             for g in ["tops", "bottoms", "outerwear", "footwear", "accessories"]]
    worst = scored.sort_values(["group_p5", "file"]).iloc[0]
    if worst["file"] not in [p["file"] for p in picks]:
        picks.append(worst)

    fig, axes = plt.subplots(len(picks), 6, figsize=(12, 2.2 * len(picks)))
    for row, pick in zip(axes, picks):
        photo = open_photo(os.path.join(FOLDER, pick["file"]))
        out = api.search_photo(photo)
        groups = [APP_GROUP_OF_TYPE[r["article_type"]] for r in out["results"]]
        if "|".join(groups) != pick["result_groups"]:
            raise SystemExit(f"{pick['file']}: results differ from the saved D9 run")
        photo.thumbnail((400, 400))
        row[0].imshow(photo)
        row[0].set_title(f"my photo: {pick['group']}\n(group P@5 {pick['group_p5']:.1f})", fontsize=8)
        for ax, result, group in zip(row[1:], out["results"], groups):
            from PIL import Image
            item = Image.open(api.engine.item.loc[result["item_id"], "image_path"]).convert("RGB")
            item.thumbnail((300, 300))
            ax.imshow(item)
            colour = GREEN if group == pick["group"] else RED
            ax.set_title(result["display_type"], fontsize=8, color=colour)
            for side in ax.spines.values():
                side.set_edgecolor(colour)
                side.set_linewidth(2)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("D9 examples: my photo (left) and the app's top 5 (green: right kind of garment, red: wrong). "
                 "Rows: the first photo of each group, plus the lowest-scoring photo", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/fig18_d9_gallery.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gallery", action="store_true")
    args = parser.parse_args()
    fig_headline()
    fig_crop_vs_whole()
    fig_d9()
    print(f"saved {OUT}/fig15_headline_real.png, fig16_crop_vs_whole.png, fig17_d9.png")
    if not args.no_gallery:
        fig_gallery()
        print(f"saved {OUT}/fig18_d9_gallery.png")


if __name__ == "__main__":
    main()
