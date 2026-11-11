"""Make the report figures from the results files.

Reads results/summary.csv (and results/lowres/summary.csv, models/finetune_log.csv
if they exist) and saves PNGs to report/figures/. Re-run any time new results
come in; methods that have not been run yet are simply left out.

    pip install matplotlib
    python -m report.make_figures

Figures:
  fig1_ladder.png      Recall@1 of each pipeline step, medium and strong phone photos, with 95% CIs
  fig2_robustness.png  Recall@1 vs phone-photo severity (mild -> strong) for the main methods
  fig3_tradeoff.png    accuracy vs search time per query
  fig4_resolution.png  same models on 60x80 vs high-res images
  fig5_finetune.png    validation Recall@1 per fine-tuning epoch (+ random-batch ablation if run)
  fig6_recall_vs_p5.png exact-item Recall@1 vs same-type P@5, synthetic and LookBench
  fig7_conformal_shift.png  conformal coverage on real photos: synthetic vs real calibration, T1 bound
  fig8_mondrian.png    per-category coverage and set size, standard vs Mondrian
  fig9_risk.png        risk control: answer rate vs guaranteed wrong-type share
"""
import os

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import pandas as pd

OUT = "report/figures"

# fixed colours: each series keeps its colour in every figure
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# the results ladder, in order: (method name in summary.csv, label in the figure)
LADDER = [
    ("color_hist", "0  Colour histogram"),
    ("clip", "1  CLIP ViT-B/32"),
    ("fashion_siglip", "2  FashionSigLIP"),
    ("fashion_siglip_crop", "3  + garment crop (query)"),
    ("fashion_siglip_cropboth", "3b + crop query & catalog"),
    ("fashion_siglip_mv", "4  + multi-view catalog (no crop)"),
    ("ensemble", "5  Ensemble SigLIP + FashionCLIP"),
    ("ensemble_mv", "5b Ensemble, multi-view"),
    ("finetuned", "7  Fine-tuned SigLIP"),
    ("finetuned_mv", "7b Fine-tuned, multi-view"),
    ("ensemble_ft_mv", "7c Fine-tuned ensemble, multi-view"),
]

# methods drawn as lines in the robustness figure, each with its own colour
MAIN = [
    ("color_hist", "Colour histogram", INK2),
    ("clip", "CLIP", ORANGE),
    ("fashion_siglip", "FashionSigLIP", BLUE),
    ("ensemble_mv", "Ensemble, multi-view", AQUA),
    ("ensemble_ft_mv", "Fine-tuned ensemble", VIOLET),
]
SEVERITIES = ["phone_mild", "phone_medium", "phone_strong"]


def style(ax):
    """Quiet axes: light grid, no top/right box, grey tick labels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def row(summary, method, condition):
    r = summary[(summary["method"] == method) & (summary["condition"] == condition)]
    return None if r.empty else r.iloc[0]


def fig_ladder(summary):
    steps = [(m, label) for m, label in LADDER if row(summary, m, "phone_medium") is not None]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(steps) + 1.2))
    height = 0.36
    for i, (method, _) in enumerate(steps):
        for offset, condition, colour in [(-height / 2, "phone_medium", BLUE),
                                          (height / 2, "phone_strong", ORANGE)]:
            r = row(summary, method, condition)
            y = i + offset
            ax.barh(y, r["Recall@1"], height=height - 0.04, color=colour)
            ax.errorbar(r["Recall@1"], y, xerr=[[r["Recall@1"] - r["Recall@1_lo"]],
                                                [r["Recall@1_hi"] - r["Recall@1"]]],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=2)
            ax.text(r["Recall@1_hi"] + 0.01, y, f"{r['Recall@1']:.0%}", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(range(len(steps)), [label for _, label in steps], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Exact-item Recall@1 (error bars: 95% bootstrap CI)", color=INK2, fontsize=9)
    ax.set_title("Each step of the pipeline, on synthetic phone photos", loc="left",
                 fontsize=11, color=INK, pad=24)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="medium phone photos"),
                       Patch(color=ORANGE, label="strong phone photos")],
              frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    style(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_ladder.png")


def fig_robustness(summary):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(SEVERITIES))
    for method, label, colour in MAIN:
        rows = [row(summary, method, c) for c in SEVERITIES]
        if any(r is None for r in rows):
            continue
        y = [r["Recall@1"] for r in rows]
        ax.fill_between(x, [r["Recall@1_lo"] for r in rows], [r["Recall@1_hi"] for r in rows],
                        color=colour, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        ax.text(len(SEVERITIES) - 1 + 0.06, y[-1], label, va="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), ["mild", "medium", "strong"])
    ax.set_xlim(-0.15, len(SEVERITIES) - 1 + 0.9)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Phone-photo severity (blur, lighting, rotation, crop, background)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("Exact-item Recall@1", color=INK2, fontsize=9)
    ax.set_title("Robustness: accuracy as photos get worse (shaded: 95% CI)", loc="left",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    save(fig, "fig2_robustness.png")


# label text and position for each point in the speed figure: (text, (dx, dy) in points, alignment).
# Points that sit almost on top of each other get one label above and one below.
TRADEOFF_LABELS = {
    "color_hist": ("Colour histogram", (8, 4), "left"),
    "clip": ("CLIP", (8, -3), "left"),
    "fashion_siglip": ("FashionSigLIP", (8, -10), "left"),
    "fashion_siglip_mv": ("+ multi-view", (-8, 4), "right"),
    "ensemble": ("Ensemble", (8, -12), "left"),
    "ensemble_mv": ("Ensemble, multi-view", (8, 6), "left"),
    "fashion_siglip_crop": ("+ crop (query)", (-8, -14), "right"),
    "fashion_siglip_cropboth": ("+ crop (query & catalog)", (-8, 8), "right"),
    "finetuned": ("Fine-tuned", (8, 6), "left"),
    "finetuned_mv": ("Fine-tuned, multi-view", (-8, 8), "right"),
    "ensemble_ft_mv": ("Fine-tuned ensemble", (8, 14), "left"),
}


def fig_tradeoff(summary):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, (text, offset, align) in TRADEOFF_LABELS.items():
        r = row(summary, method, "phone_medium")
        if r is None:
            continue
        ax.plot(r["mean_latency_ms"], r["Recall@1"], "o", markersize=8, color=BLUE,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(text, (r["mean_latency_ms"], r["Recall@1"]), textcoords="offset points",
                    xytext=offset, ha=align, va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(2.5, 600)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    ax.set_xlabel("Mean search time per query (log scale, laptop GPU)", color=INK2, fontsize=9)
    ax.set_ylabel("Recall@1, medium phone photos", color=INK2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs speed (up and to the left is better)", loc="left", fontsize=11, color=INK)
    style(ax)
    save(fig, "fig3_tradeoff.png")


def fig_resolution(summary, lowres_path="results/lowres/summary.csv"):
    if not os.path.exists(lowres_path):
        return
    low = pd.read_csv(lowres_path)
    models = [("clip", "CLIP"), ("fashion_clip", "FashionCLIP"), ("fashion_siglip", "FashionSigLIP")]
    low_vals, high_vals, names = [], [], []
    for method, name in models:
        lo = low[(low["method"] == method) & (low["condition"] == "phone")]
        hi = row(summary, method, "phone_medium")
        if lo.empty or hi is None:
            continue
        low_vals.append(lo.iloc[0]["Recall@1"])
        high_vals.append(hi["Recall@1"])
        names.append(name)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = range(len(names))
    w = 0.38
    for xs, vals, colour, label in [([i - w / 2 for i in x], low_vals, INK2, "60x80 px catalog"),
                                    ([i + w / 2 for i in x], high_vals, BLUE, "high-res catalog")]:
        ax.bar(xs, vals, width=w - 0.03, color=colour, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(x), names, fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Recall@1, medium phone photos", color=INK2, fontsize=9)
    ax.set_title("Image resolution matters", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_resolution.png")


def fig_finetune(log_path="models/finetune_log.csv",
                 random_log_path="models/finetune_random_log.csv"):
    """Validation Recall@1 per epoch; also draws the random-batch ablation if it exists."""
    if not os.path.exists(log_path):
        return
    log = pd.read_csv(log_path)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(log["epoch"], log["val_recall@1"], color=BLUE, linewidth=2, marker="o", markersize=6,
            label="same-type hard batches")
    if os.path.exists(random_log_path):
        rand = pd.read_csv(random_log_path)
        ax.plot(rand["epoch"], rand["val_recall@1"], color=ORANGE, linewidth=2, marker="s",
                markersize=5, label="random batches (ablation)")
        ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.axhline(log["val_recall@1"].iloc[0], color=INK2, linewidth=1, linestyle="--")
    ax.text(0.15, log["val_recall@1"].iloc[0], "pretrained", va="bottom",
            ha="left", fontsize=8, color=INK2)
    ax.set_xticks(log["epoch"])
    ax.set_xlabel("Epoch (0 = before training)", color=INK2, fontsize=9)
    ax.set_ylabel("Validation Recall@1 (medium)", color=INK2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Fine-tuning: validation accuracy per epoch", loc="left", fontsize=11, color=INK)
    style(ax)
    save(fig, "fig5_finetune.png")


# models in the exact-item vs same-type figure: (method, label, colour)
TRADE = [
    ("fashion_siglip", "FashionSigLIP", BLUE),
    ("ensemble_mv", "Ensemble (pretrained)", AQUA),
    ("finetuned", "Fine-tuned", ORANGE),
    ("finetuned_mv", "Fine-tuned, multi-view", YELLOW),
    ("finetuned_random", "Fine-tuned, random batches", MAGENTA),
    ("ensemble_ft_mv", "Fine-tuned ensemble", VIOLET),
]


def fig_recall_vs_p5(summary, lookbench_path="results/lookbench_summary.csv"):
    """Exact-item Recall@1 (x) vs same-type Precision@5 (y), synthetic medium and real photos.

    Each dot is one model with 95% CIs on both axes. Up-and-right is better;
    a model that moves right but down has traded 'similar items' for 'exact item'."""
    panels = [("Synthetic phone photos (medium)", summary, "Recall@1", "Precision@5",
               lambda m: row(summary, m, "phone_medium"))]
    if os.path.exists(lookbench_path):
        lb = pd.read_csv(lookbench_path)
        lb = lb[(lb["subset"] == "real_streetlook") & (lb["query_image"] == "image")]

        def lb_row(m):
            r = lb[lb["method"] == m]
            return None if r.empty else r.iloc[-1]
        panels.append(("Real photos (LookBench street)", lb, "Recall@1", "CategoryP@5", lb_row))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), squeeze=False)
    for ax, (title, _, xcol, ycol, get) in zip(axes[0], panels):
        for method, label, colour in TRADE:
            r = get(method)
            if r is None:
                continue
            x, y = r[xcol], r[ycol]
            ax.errorbar(x, y, xerr=[[x - r[xcol + "_lo"]], [r[xcol + "_hi"] - x]],
                        yerr=[[y - r[ycol + "_lo"]], [r[ycol + "_hi"] - y]],
                        fmt="o", color=colour, markersize=7, markeredgecolor="white",
                        elinewidth=1, capsize=0, label=label)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Exact-item Recall@1", color=INK2, fontsize=9)
        ax.set_ylabel("Same-type Precision@5", color=INK2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style(ax)
    axes[0][-1].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Exact item vs similar items (up and right is better)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    save(fig, "fig6_recall_vs_p5.png")

# models in the conformal figures: (method, label, colour), same colours as above
CONF = [("fashion_siglip", "FashionSigLIP", BLUE),
        ("ensemble_mv", "Ensemble (pretrained)", AQUA),
        ("finetuned_random", "Fine-tuned, random batches", MAGENTA)]


def fig_conformal_shift(path="results/conformal_shift.csv", tv_path="results/conformal_tv.csv"):
    """Coverage on real photos when calibrating on synthetic vs real photos, and the T1 bound."""
    if not (os.path.exists(path) and os.path.exists(tv_path)):
        return
    table = pd.read_csv(path)
    table = table[(table["alpha"] == 0.1) & (table["method"] == "relative")]
    tv = pd.read_csv(tv_path)
    tv = tv[(tv["method"] == "relative") & (tv["bins"] == 50)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (method, label, colour) in enumerate(CONF):
        syn = table[(table["model"] == method) & (table["setting"] == "synthetic_to_real")]
        real = table[(table["model"] == method) & (table["setting"] == "real_to_real")]
        bound = tv[tv["model"] == method]
        if syn.empty or real.empty:
            continue
        ax1.bar(i - 0.2, syn["coverage"].iloc[0], width=0.38, color=colour, alpha=0.45)
        ax1.bar(i + 0.2, real["coverage"].iloc[0], width=0.38, color=colour)
        if not bound.empty:
            ax1.plot([i - 0.38, i - 0.02], [bound["bound_at_90"].iloc[0]] * 2, color=INK, linewidth=2)
        ax2.bar(i, real["median_size"].iloc[0], width=0.6, color=colour)
        ax2.text(i, real["median_size"].iloc[0], f"{real['median_size'].iloc[0]:.0f}",
                 ha="center", va="bottom", fontsize=8, color=INK2)
    ax1.axhline(0.9, color=INK2, linestyle="--", linewidth=1)
    ax1.text(2.45, 0.905, "target 90%", fontsize=8, color=INK2, ha="right", va="bottom")
    names = [label.replace(" (", "\n(").replace(", ", ",\n") for _, label, _ in CONF]
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(CONF)))
        ax.set_xticklabels(names, fontsize=8)
        style(ax)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_ylabel("Coverage on real photos", color=INK2, fontsize=9)
    ax1.set_title("Light: calibrated on synthetic. Dark: on real.\nBlack line: T1 bound 0.9 - TV",
                  loc="left", fontsize=9.5, color=INK)
    ax2.set_ylabel("Median set size (of 3,278)", color=INK2, fontsize=9)
    ax2.set_title("Honest set size on real photos\n(calibrated on real)", loc="left",
                  fontsize=9.5, color=INK)
    save(fig, "fig7_conformal_shift.png")


def fig_mondrian(path="results/conformal_mondrian.csv", model="finetuned_random", min_queries=18):
    """Per-category coverage, standard vs per-category (Mondrian) calibration."""
    if not os.path.exists(path):
        return
    table = pd.read_csv(path)
    table = table[(table["model"] == model) & (table["n_queries"] >= min_queries)]
    table = table.sort_values("coverage_standard")
    y = range(len(table))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax1.plot(table["coverage_standard"], y, "o", color=INK2, label="standard (one calibration)")
    ax1.plot(table["coverage_mondrian"], y, "o", color=VIOLET, label="per category (Mondrian)")
    ax1.axvline(0.9, color=INK2, linestyle="--", linewidth=1)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([f"{c} ({n})" for c, n in zip(table["category"], table["n_queries"])], fontsize=8)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_xlabel("Coverage (target 90%)", color=INK2, fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("Coverage per category", loc="left", fontsize=10, color=INK)
    ax2.plot(table["median_size_standard"], y, "o", color=INK2)
    ax2.plot(table["median_size_mondrian"], y, "o", color=VIOLET)
    ax2.set_xscale("log")
    ax2.set_xlabel("Median set size (log scale)", color=INK2, fontsize=9)
    ax2.set_title("Set size per category", loc="left", fontsize=10, color=INK)
    for ax in (ax1, ax2):
        style(ax)
    save(fig, "fig8_mondrian.png")


def fig_risk(path="results/conformal_risk.csv"):
    """T4: how often the app still answers, for each guaranteed level of wrong-type results."""
    if not os.path.exists(path):
        return
    table = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, label, colour in CONF:
        t = table[table["model"] == method]
        if t.empty:
            continue
        guaranteed = t[t["alpha"] != "no abstention"].copy()
        guaranteed["alpha"] = guaranteed["alpha"].astype(float)
        guaranteed = guaranteed.sort_values("alpha")
        free = t[t["alpha"] == "no abstention"]
        xs = list(guaranteed["alpha"]) + list(free["loss"])
        ys = list(guaranteed["answer_rate"]) + list(free["answer_rate"])
        ax.plot(xs, ys, "-o", color=colour, label=label, markersize=6)
    ax.set_xlabel("Average share of wrong-type results (guaranteed bound; last point = no abstention)",
                  color=INK2, fontsize=8.5)
    ax.set_ylabel("Share of photos the app answers", color=INK2, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Risk control on real photos: stricter guarantee = answer less often",
                 loc="left", fontsize=10, color=INK)
    style(ax)
    save(fig, "fig9_risk.png")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/{name}")


def main():
    summary = pd.read_csv("results/summary.csv")
    fig_ladder(summary)
    fig_robustness(summary)
    fig_tradeoff(summary)
    fig_resolution(summary)
    fig_finetune()
    fig_recall_vs_p5(summary)
    fig_conformal_shift()
    fig_mondrian()
    fig_risk()


if __name__ == "__main__":
    main()
