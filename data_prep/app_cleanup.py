"""Clean the APP catalogue of items a shopper should never see (the evaluation catalogue is untouched).

Problems found in the app (pre-registered in report/results_log.md, "app catalogue clean-up"):
  1. detail shots: some H&M items are close-ups of the fabric, useless as a search result.
  2. wrong images: some Kaggle rows show a different product than their label (item 13378 is listed as
     "Timberland Unisex Infant Pink Casual Shoes" but its image is a black Fossil wallet).
  3. kids' items: the app hides items whose Kaggle gender is Boys/Girls, but many kids' items are listed
     as Men/Women/Unisex; their product NAME still says Kids, Infant, Boys, ...

Three rules, all fixed before running (no value is tried twice):
  closeup     : P(close-up) >= 0.5, from a 2-way zero-shot choice of the PRETRAINED FashionSigLIP between
                "a photo of a close-up of fabric" and the item's own type prompt ("a photo of jeans", ...)
  wrong_image : the zero-shot group (app vocabulary, 7 groups, same as the app's group badge) differs
                from the item's listed group, with group probability >= 0.9 (the badge's own threshold)
  kids_name   : Kaggle product name contains a kids' word (KIDS_WORDS); H&M items come from adult sections
The items hit by any rule are written to serving/app_hidden.csv; the app hides them (next step, after the
contact sheets are checked). The same rules are COUNTED on the evaluation catalogue, never applied there.

Sanity checks (stop if they fail):
  - the pretrained embeddings of the first 6,197 app rows (= the evaluation catalogue) equal the saved
    models/index/fashion_siglip.npy;
  - for 5 random app items, the batch zero-shot group and probability equal TypePredictor("app").analyse.

Run from the repo root (the first run embeds ~54k images with the pretrained model on MPS, ~15-30 min):
    python -m data_prep.app_cleanup
Writes serving/app_hidden.csv, results/app_cleanup.csv and contact sheets report/figures/cleanup_*.png.
"""
import os

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, ImageDraw

from evaluation.run_eval import RETRIEVERS
from serving.badges import APP_GROUP_OF_TYPE, APP_TYPE_WORDS, SIGLIP, TypePredictor

CLOSEUP_PROMPT = "a photo of a close-up of fabric"
CLOSEUP_MIN = 0.5
WRONG_MIN = 0.9
KIDS_WORDS = r"\b(?:kid|kids|kid's|kidswear|infant|infants|baby|babies|toddler|boy|boys|boy's|girl|girls|girl's)\b"
N_EVALUATION = 6197                          # the first rows of the app catalogue are the evaluation catalogue
OUT_HIDDEN = "serving/app_hidden.csv"
OUT_COUNTS = "results/app_cleanup.csv"
SHEET_SIZE = 24                              # items per contact sheet (4 rows of 6)


def text_embeddings(model, prompts):
    """Prompts -> normalised text embeddings (needs the tokenizer, i.e. the transformers package)."""
    tokenizer = open_clip.get_tokenizer(SIGLIP)
    device = next(model.parameters()).device
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy()


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def zero_shot_groups(features, scale):
    """Same computation as TypePredictor.analyse: softmax over the 72 app types, summed per group."""
    types = list(APP_TYPE_WORDS)
    type_text = np.load("serving/type_text_app.npy")             # saved prompt embeddings, same order
    probs = softmax(scale * features @ type_text.T)              # (items, 72)
    groups = sorted(set(APP_GROUP_OF_TYPE.values()))
    member = np.array([[APP_GROUP_OF_TYPE[t] == g for g in groups] for t in types], dtype=float)
    group_probs = probs @ member                                 # (items, 7)
    best = group_probs.argmax(axis=1)
    return np.array(groups)[best], group_probs[np.arange(len(best)), best]


def closeup_probability(features, scale, article_types, closeup_text):
    """2-way softmax: close-up of fabric vs the item's own type prompt."""
    type_text = np.load("serving/type_text_app.npy")
    own = type_text[[list(APP_TYPE_WORDS).index(t) for t in article_types]]   # (items, d)
    s_own = (features * own).sum(axis=1)
    s_close = features @ closeup_text[0]
    logits = scale * np.stack([s_close, s_own], axis=1)
    return softmax(logits)[:, 0]


def kids_in_name(catalog):
    """True where the Kaggle product name contains a kids' word (H&M items: always False)."""
    styles = pd.read_csv("data/raw/styles.csv", dtype=str, on_bad_lines="skip")
    names = dict(zip(styles["id"], styles["productDisplayName"]))
    name = catalog["item_id"].map(names)
    is_kaggle = catalog["source"] != "hm"
    print(f"  Kaggle items without a product name: {int((is_kaggle & name.isna()).sum())} (counted as not kids)")
    return (is_kaggle & name.str.contains(KIDS_WORDS, case=False, regex=True, na=False)).to_numpy(), name


def contact_sheet(rows, path, caption):
    """Grid of item images with a short caption under each (for checking a rule by eye)."""
    w, h, pad = 180, 240, 34
    sheet = Image.new("RGB", (6 * w, ((len(rows) + 5) // 6) * (h + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (_, row) in enumerate(rows.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((w - 6, h - 6))
        x, y = (k % 6) * w, (k // 6) * (h + pad)
        sheet.paste(image, (x + 3, y + 3))
        draw.text((x + 3, y + h), caption(row), fill="black")
    sheet.save(path)
    print(f"  saved {path}")


def main():
    catalog = pd.read_csv("data/app_catalog.csv", dtype={"item_id": str})
    print(f"app catalogue: {len(catalog)} items")

    # pretrained FashionSigLIP embeddings of the app catalogue (saved once, like every other index)
    retriever = RETRIEVERS["fashion_siglip"]()
    retriever.index_name = "fashion_siglip_app"
    retriever.build_index(catalog)
    features = retriever.features
    scale = float(retriever.model.logit_scale.exp())

    # check 1: the evaluation rows reproduce the saved pretrained index
    saved = np.load("models/index/fashion_siglip.npy")
    diff = float(np.abs(features[:N_EVALUATION] - saved).max())
    print(f"  check: first {N_EVALUATION} rows vs saved fashion_siglip.npy, max difference {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("PROBLEM: the app embeddings do not reproduce the saved evaluation embeddings; stopping.")

    group, group_prob = zero_shot_groups(features, scale)

    # check 2: the batch computation equals the app's own TypePredictor on 5 random items
    predictor = TypePredictor("app")
    for i in np.random.default_rng(0).choice(len(catalog), size=5, replace=False):
        seen = predictor.analyse(Image.open(catalog.loc[i, "image_path"]))
        print(f"  check: item {catalog.loc[i, 'item_id']}: batch {group[i]} {group_prob[i]:.3f}, "
              f"app {seen['group']} {seen['group_prob']:.3f}")
        if seen["group"] != group[i] or abs(seen["group_prob"] - group_prob[i]) > 1e-2:
            raise SystemExit("PROBLEM: batch zero-shot groups differ from the app's TypePredictor; stopping.")

    # the three rules
    closeup_text = text_embeddings(retriever.model, [CLOSEUP_PROMPT])
    catalog["p_closeup"] = closeup_probability(features, scale, catalog["article_type"], closeup_text)
    catalog["listed_group"] = catalog["article_type"].map(APP_GROUP_OF_TYPE)
    catalog["zero_shot_group"], catalog["group_prob"] = group, group_prob
    catalog["closeup"] = catalog["p_closeup"] >= CLOSEUP_MIN
    catalog["wrong_image"] = (catalog["zero_shot_group"] != catalog["listed_group"]) & (catalog["group_prob"] >= WRONG_MIN)
    catalog["kids_name"], catalog["name"] = kids_in_name(catalog)
    rules = ["closeup", "wrong_image", "kids_name"]
    catalog["any_rule"] = catalog[rules].any(axis=1)
    catalog["part"] = np.where(np.arange(len(catalog)) < N_EVALUATION, "evaluation rows", catalog["source"])

    # counts per rule and source (evaluation rows: counted only, never hidden in the evaluation catalogue)
    counts = catalog.groupby("part")[rules + ["any_rule"]].agg(["sum"]).droplevel(1, axis=1)
    counts.insert(0, "items", catalog.groupby("part").size())
    counts.loc["all app"] = [len(catalog)] + [int(catalog[r].sum()) for r in rules + ["any_rule"]]
    counts.to_csv(OUT_COUNTS)
    print(counts.to_string())
    already = catalog["gender"].isin(["Boys", "Girls"])
    print(f"  kids_name items not already hidden by gender: {int((catalog['kids_name'] & ~already).sum())}")

    # hidden list for the app
    hit = catalog[catalog["any_rule"]].copy()
    hit["reasons"] = hit[rules].apply(lambda r: ";".join(x for x in rules if r[x]), axis=1)
    hit[["item_id", "reasons"]].to_csv(OUT_HIDDEN, index=False)
    print(f"saved {OUT_HIDDEN} ({len(hit)} items) and {OUT_COUNTS}")

    # contact sheets: random dropped and random kept items for each rule (seed 0)
    os.makedirs("report/figures", exist_ok=True)
    pools = {"closeup": catalog["source"] == "hm", "wrong_image": catalog["source"] == catalog["source"],
             "kids_name": catalog["source"] != "hm"}
    captions = {"closeup": lambda r: f"{r['item_id']} {r['article_type']}\nP(close-up) {r['p_closeup']:.2f}",
                "wrong_image": lambda r: f"{r['item_id']} {r['article_type']}\nlooks {r['zero_shot_group']} {r['group_prob']:.2f}",
                "kids_name": lambda r: f"{r['item_id']} {r['gender']}\n{str(r['name'])[:28]}"}
    for rule in rules:
        dropped = catalog[rule] & ~already if rule == "kids_name" else catalog[rule]   # kids: the NEW drops only
        for kind, chosen in [("dropped", dropped), ("kept", pools[rule] & ~catalog[rule])]:
            part = catalog[chosen]
            sample = part.sample(n=min(SHEET_SIZE, len(part)), random_state=0)
            contact_sheet(sample, f"report/figures/cleanup_{rule}_{kind}.png", captions[rule])


if __name__ == "__main__":
    main()