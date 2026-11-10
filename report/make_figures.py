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
  fig5_finetune.png    validation Recall@1 per fine-tuning epoch (once finetune_log.csv exists)
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


def fig_finetune(log_path="models/finetune_log.csv"):
    if not os.path.exists(log_path):
        return
    log = pd.read_csv(log_path)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(log["epoch"], log["val_recall@1"], color=BLUE, linewidth=2, marker="o", markersize=6)
    ax.axhline(log["val_recall@1"].iloc[0], color=INK2, linewidth=1, linestyle="--")
    ax.text(log["epoch"].max(), log["val_recall@1"].iloc[0], "  pretrained", va="bottom",
            ha="right", fontsize=8, color=INK2)
    ax.set_xticks(log["epoch"])
    ax.set_xlabel("Epoch (0 = before training)", color=INK2, fontsize=9)
    ax.set_ylabel("Validation Recall@1 (medium)", color=INK2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Fine-tuning: validation accuracy per epoch", loc="left", fontsize=11, color=INK)
    style(ax)
    save(fig, "fig5_finetune.png")


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


if __name__ == "__main__":
    main()
