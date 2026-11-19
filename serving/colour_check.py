"""Can the colour of the garment be judged from this photo? (black-and-white and very dark photos)

Problem: the app's colour estimate (serving/badges.py, garment_colour) is the median CIELAB colour of the
photo's centre. On a black-and-white photo it is always grey, and on a very dark photo it is almost black,
yet the app still shows confident colour badges ("colour different"). This script fixes a simple check
for the shopper's photo, using two numbers computed over the whole photo:
    colourfulness = 95th percentile of CIELAB chroma (how far pixels are from grey)
    brightness    = 95th percentile of L* (how light even the lightest parts are)
Flag "black-and-white" if colourfulness < T_c, "too dark" if brightness < T_L.

Rule (pre-registered in report/results_log.md): T_c and T_L are the 1st percentile of each number over
the DISTINCT LookBench query crops of the DEV half, so at most ~1% of normal real photos are flagged.
Check, once: TEST-half crops and 300 random catalogue images, original and degraded versions
(grayscale, brightness x0.5 / x0.3 / x0.15). Only the QUERY photo is checked; catalogue colours and the
colour bands of Finding 24 are unchanged.

Sanity check: recomputes the colour bands of Finding 24 and stops if they differ from serving/badges.json.

Run from the repo root (a few minutes, no model needed):
    python -m serving.colour_check
Writes results/colour_check.csv and serving/colour_check.json.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab

from serving.badges import garment_colour

LB = "data/lookbench/real_streetlook"
OUT_CSV = "results/colour_check.csv"
OUT_JSON = "serving/colour_check.json"
PERCENTILE = 1                       # thresholds = 1st percentile on normal real photos (dev half)
VERSIONS = ["original", "grayscale", "bright_0.5", "bright_0.3", "bright_0.15"]


def photo_numbers(image):
    """Colourfulness (95th pct chroma) and brightness (95th pct L*) over the whole photo."""
    small = image.convert("RGB")
    small.thumbnail((224, 224))                         # same scale for every photo, and fast
    lab = rgb2lab(np.asarray(small, dtype=float) / 255).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    return float(np.percentile(chroma, 95)), float(np.percentile(lab[:, 0], 95))


def degrade(image, version):
    """The photo as a black-and-white photo or a darker photo (PIL brightness factor)."""
    image = image.convert("RGB")
    if version == "original":
        return image
    if version == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    factor = float(version.split("_")[1])
    return ImageEnhance.Brightness(image).enhance(factor)


def sanity_check():
    """Recompute the colour bands exactly as serving/badges.py does (same sample, same seed)."""
    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    rng = np.random.default_rng(0)
    sample = catalog.sample(n=1500, random_state=0)
    colour_of = {i: garment_colour(Image.open(p)) for i, p in zip(sample["item_id"], sample["image_path"])}
    same = []
    for label, part in sample.groupby("color"):
        ids = list(part["item_id"])
        for _ in range(min(300, len(ids) * 3)):
            a, b = rng.choice(len(ids), size=2, replace=False) if len(ids) > 1 else (0, 0)
            if a != b:
                same.append(float(deltaE_ciede2000(colour_of[ids[a]][None], colour_of[ids[b]][None])[0]))
    saved = json.load(open("serving/badges.json"))["colour_bands"]
    now = dict(very_close_max=float(np.percentile(same, 50)), different_min=float(np.percentile(same, 90)))
    print(f"  check: colour bands now {now['very_close_max']:.2f} / {now['different_min']:.2f}, "
          f"saved {saved['very_close_max']:.2f} / {saved['different_min']:.2f}")
    if abs(now["very_close_max"] - saved["very_close_max"]) > 1e-6 or \
            abs(now["different_min"] - saved["different_min"]) > 1e-6:
        raise SystemExit("PROBLEM: colour bands do not match serving/badges.json; stopping.")
    return saved


def distinct_crops(side):
    """Paths of the distinct query crops of one split side (identical crops counted once, Finding 20)."""
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str).merge(pd.read_csv(f"{LB}/split.csv"), on="query_id")
    paths = queries.loc[queries["side"] == side, "image"]
    md5 = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in paths]
    return list(pd.Series(list(paths), index=md5).groupby(level=0).first())


def measure(photos, source, labels, thresholds, different_min):
    """One row per (photo, version): the two numbers, the flags, and how far the colour estimate moved."""
    rows = []
    for path, label in zip(photos, labels):
        original = Image.open(path).convert("RGB")
        original_colour = garment_colour(original)
        for version in VERSIONS:
            image = degrade(original, version)
            colourful, bright = photo_numbers(image)
            moved = float(deltaE_ciede2000(original_colour[None], garment_colour(image)[None])[0])
            rows.append(dict(source=source, path=path, label=label, version=version,
                             colourfulness=colourful, brightness=bright,
                             flag_bw=colourful < thresholds["colourfulness_min"],
                             flag_dark=bright < thresholds["brightness_min"],
                             estimate_moved=moved > different_min))
    return pd.DataFrame(rows)


def main():
    bands = sanity_check()

    # thresholds from normal real photos, DEV half only
    dev = [photo_numbers(Image.open(p)) for p in distinct_crops("dev")]
    thresholds = dict(colourfulness_min=float(np.percentile([c for c, _ in dev], PERCENTILE)),
                      brightness_min=float(np.percentile([b for _, b in dev], PERCENTILE)),
                      rule=f"{PERCENTILE}st percentile over {len(dev)} distinct LookBench dev crops")
    print(f"  thresholds from {len(dev)} distinct dev crops: colourfulness < {thresholds['colourfulness_min']:.2f}, "
          f"brightness < {thresholds['brightness_min']:.2f}")

    # check, once: test-half crops and catalogue images
    test = distinct_crops("test")
    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str}).sample(n=300, random_state=1)
    neutral = np.where(catalog["color"].isin(["Black", "White", "Grey"]), "neutral listed", "colour listed")
    rows = pd.concat([measure(test, "lookbench_test", ["real crop"] * len(test), thresholds, bands["different_min"]),
                      measure(catalog["image_path"], "catalogue", neutral, thresholds, bands["different_min"])])

    rows["flag_any"] = rows["flag_bw"] | rows["flag_dark"]
    table = rows.groupby(["source", "label", "version"], sort=False).agg(
        n=("path", "size"), flag_bw=("flag_bw", "mean"), flag_dark=("flag_dark", "mean"),
        flag_any=("flag_any", "mean"), estimate_moved=("estimate_moved", "mean")).reset_index()
    table.to_csv(OUT_CSV, index=False)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    # among degraded photos whose colour estimate moved past the "different" band: how many are flagged?
    degraded = rows[rows["version"] != "original"]
    moved = degraded[degraded["estimate_moved"]]
    print(f"  degraded photos whose estimate moved > {bands['different_min']:.1f}: {len(moved)}, "
          f"flagged {moved['flag_any'].mean():.3f}")

    test_rows = rows[rows["source"] == "lookbench_test"]
    false_alarm = test_rows.loc[test_rows["version"] == "original", "flag_any"].mean()
    bw_found = test_rows.loc[test_rows["version"] == "grayscale", "flag_bw"].mean()
    ship = bool(false_alarm <= 0.02 and bw_found >= 0.95)
    print(f"  ship rule: false alarms on test originals {false_alarm:.3f} (<= 0.02), "
          f"grayscale found {bw_found:.3f} (>= 0.95) -> {'SHIP' if ship else 'DO NOT SHIP'}")

    with open(OUT_JSON, "w") as f:
        json.dump(dict(thresholds, test_false_alarm=float(false_alarm), test_grayscale_found=float(bw_found),
                       ship=ship), f, indent=2)
    print(f"saved {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()