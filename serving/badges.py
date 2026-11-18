"""E3: match badges for the app. They DESCRIBE measured differences; they are not the model's reasons.

Type badge
  The query's garment type is predicted zero-shot with the PRETRAINED FashionSigLIP (image and
  text encoders): similarity of the photo with "a photo of a t-shirt", "a photo of jeans", ...
  for the 14 catalogue types, softmax -> (type, probability). It is independent of the ranking
  (which uses the fine-tuned model), so the badge is not circular. A result gets "same type" if
  its catalogue type equals the predicted type.

Colour badge
  Garment colour = median colour, in CIELAB (a colour space where distance ~ how different two
  colours look), of the central part of the image, ignoring near-white background pixels.
  Difference = CIEDE2000 Delta-E between query and result. Bands are set from the catalogue's own
  colour labels: over random pairs of catalogue items with the SAME colour label,
      very close      : Delta-E <= median of same-label pairs
      different       : Delta-E >  90th percentile of same-label pairs
      slightly different in between.
  So "different" is only claimed for a gap larger than 90% of same-label pairs show.

Evaluation (synthetic test photos, true type and true item known; finetuned_random saved top 5):
  - type accuracy of the zero-shot prediction (clean catalogue photos as a sanity check,
    medium and strong phone photos as the real test);
  - how often the "same type" badge on a shown result agrees with the truth;
  - how often a TRUE match (query vs its own catalogue item) gets the "different" colour badge
    (robustness to phone lighting).
Saves serving/badges.json (bands + evaluation).

Sanity checks: CIELAB of pure white is (100, 0, 0); catalogue items labelled Black are darker
(median L) than items labelled White.

Run from the repo root (needs the pretrained model, about a minute):
    python -m serving.badges
"""
import json
import os

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from retrieval.clip_retriever import pick_device

SIGLIP = "hf-hub:Marqo/marqo-fashionSigLIP"
TYPE_WORDS = {"Tshirts": "a t-shirt", "Shirts": "a shirt", "Tops": "a top", "Kurtas": "a kurta",
              "Sweatshirts": "a sweatshirt", "Jackets": "a jacket", "Dresses": "a dress",
              "Jeans": "jeans", "Trousers": "trousers", "Track Pants": "track pants",
              "Shorts": "shorts", "Casual Shoes": "casual shoes", "Sports Shoes": "sports shoes",
              "Heels": "heels"}
# coarse groups for the type badge (the 14 fine types are confusable even on clean photos)
GROUP_OF_TYPE = {"Tshirts": "tops", "Shirts": "tops", "Tops": "tops", "Kurtas": "tops",
                 "Sweatshirts": "tops", "Jackets": "outerwear", "Dresses": "dresses",
                 "Jeans": "bottoms", "Trousers": "bottoms", "Track Pants": "bottoms", "Shorts": "bottoms",
                 "Casual Shoes": "footwear", "Sports Shoes": "footwear", "Heels": "footwear"}
CONFIDENCE = [0.0, 0.5, 0.7, 0.9]     # show the group badge only if its probability >= this
BADGE_MIN_CONFIDENCE = 0.9     # chosen Nov 18 after seeing the table above (UI decision; D9 checks it)
OUT = "serving/badges.json"
TEXT_EMB = "serving/type_text.npy"      # the 14 prompt embeddings, computed once (needs transformers)


class TypePredictor:
    """Zero-shot garment type with the pretrained FashionSigLIP."""

    def __init__(self):
        self.device = pick_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(SIGLIP)
        self.model = self.model.to(self.device).eval()
        self.types = list(TYPE_WORDS)
        if not os.path.exists(TEXT_EMB):        # first run only: the app then needs no tokenizer
            tokenizer = open_clip.get_tokenizer(SIGLIP)
            with torch.no_grad():
                text = tokenizer([f"a photo of {TYPE_WORDS[t]}" for t in self.types]).to(self.device)
                t = self.model.encode_text(text)
                t = t / t.norm(dim=-1, keepdim=True)
            np.save(TEXT_EMB, t.float().cpu().numpy())
            print(f"  saved the {len(self.types)} prompt embeddings to {TEXT_EMB}")
        self.text = torch.from_numpy(np.load(TEXT_EMB)).to(self.device)

    @torch.no_grad()
    def predict(self, image):
        """PIL image -> (predicted catalogue type, probability)."""
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        v = self.model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
        scale = self.model.logit_scale.exp()            # the model's own temperature
        probs = (scale * v @ self.text.T).softmax(dim=-1)[0].float().cpu().numpy()
        best = int(probs.argmax())
        return self.types[best], float(probs[best])

    def predict_group(self, image):
        """PIL image -> (predicted group, probability) = sum of the type probabilities in each group."""
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
            probs = (self.model.logit_scale.exp() * v @ self.text.T).softmax(dim=-1)[0].float().cpu().numpy()
        by_group = {}
        for t, p in zip(self.types, probs):
            by_group[GROUP_OF_TYPE[t]] = by_group.get(GROUP_OF_TYPE[t], 0.0) + float(p)
        best = max(by_group, key=by_group.get)
        return best, by_group[best]


def garment_colour(image):
    """Median CIELAB colour of the image centre, ignoring near-white background pixels."""
    small = np.asarray(image.convert("RGB").resize((96, 128)), dtype=float) / 255
    centre = small[19:109, 19:77]                       # middle ~70% height, ~60% width
    lab = rgb2lab(centre).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = ~((lab[:, 0] > 90) & (chroma < 8))           # drop white background
    if keep.sum() < 20:                                 # (almost) all white: it IS a white garment
        keep[:] = True
    return np.median(lab[keep], axis=0)


def lab_to_hex(lab):
    """CIELAB colour -> '#rrggbb' (for showing the estimated garment colour as a swatch)."""
    rgb = np.clip(lab2rgb(np.asarray(lab, dtype=float)[None, None, :])[0, 0], 0, 1)
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def colour_band(lab_a, lab_b, bands):
    """'very close' / 'slightly different' / 'different' from the CIEDE2000 difference."""
    d = float(deltaE_ciede2000(lab_a[None], lab_b[None])[0])
    if d <= bands["very_close_max"]:
        return "very close", d
    if d > bands["different_min"]:
        return "different", d
    return "slightly different", d


def main():
    white = rgb2lab(np.ones((1, 1, 3)))[0, 0]
    print(f"  check: CIELAB of pure white = ({white[0]:.1f}, {white[1]:.1f}, {white[2]:.1f}), expect (100, 0, 0)")
    red = rgb2lab(np.array([[[1.0, 0.0, 0.0]]]))[0, 0]
    print(f"  check: pure red -> CIELAB -> hex = {lab_to_hex(red)}, expect #ff0000")

    catalog = pd.read_csv("data/catalog.csv", dtype={"item_id": str})
    rng = np.random.default_rng(0)
    sample = catalog.sample(n=1500, random_state=0)
    colour_of = {i: garment_colour(Image.open(p)) for i, p in zip(sample["item_id"], sample["image_path"])}
    light = {lab: np.median([colour_of[i][0] for i in sample.loc[sample["color"] == lab, "item_id"]])
             for lab in ["Black", "White"]}
    print(f"  check: median lightness L of Black items {light['Black']:.0f} < White items {light['White']:.0f}"
          f" -> {'OK' if light['Black'] < light['White'] else 'PROBLEM'}")

    # colour bands from same-label pairs
    same = []
    for label, part in sample.groupby("color"):
        ids = list(part["item_id"])
        for _ in range(min(300, len(ids) * 3)):
            a, b = rng.choice(len(ids), size=2, replace=False) if len(ids) > 1 else (0, 0)
            if a != b:
                same.append(float(deltaE_ciede2000(colour_of[ids[a]][None], colour_of[ids[b]][None])[0]))
    bands = dict(very_close_max=float(np.percentile(same, 50)), different_min=float(np.percentile(same, 90)),
                 n_same_label_pairs=len(same))
    print(f"  colour bands from {len(same)} same-label pairs: very close <= {bands['very_close_max']:.1f}, "
          f"different > {bands['different_min']:.1f} (CIEDE2000)")

    # evaluation on synthetic test photos
    predictor = TypePredictor()
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    type_of = dict(zip(catalog["item_id"], catalog["article_type"]))
    path_of = dict(zip(catalog["item_id"], catalog["image_path"]))
    evaluation = {}
    for condition in ["clean", "phone_medium", "phone_strong"]:
        part = queries[queries["condition"] == condition]
        top5 = pd.read_csv(f"results/finetuned_random_{condition}.csv", dtype={"item_id": str})
        correct, agree, true_pairs, group_rows = [], [], [], []
        for q in part.itertuples():
            image = Image.open(q.image_path)
            predicted, _ = predictor.predict(image)
            correct.append(predicted == q.article_type)
            group, group_prob = predictor.predict_group(image)
            group_rows.append((group == GROUP_OF_TYPE[q.article_type], group_prob))
            for item in top5.loc[top5["query_id"] == q.query_id, "item_id"]:
                agree.append((type_of[item] == predicted) == (type_of[item] == q.article_type))
            if condition != "clean":
                band, _ = colour_band(garment_colour(image), garment_colour(Image.open(path_of[q.source_item_id])), bands)
                true_pairs.append(band)
        result = dict(n=len(part), type_accuracy=float(np.mean(correct)),
                      same_type_badge_agrees=float(np.mean(agree)))
        ok = np.array([g for g, _ in group_rows])
        conf = np.array([c for _, c in group_rows])
        result["group_badge"] = {f"conf>={t}": dict(shown=float((conf >= t).mean()),
                                                     accuracy_when_shown=float(ok[conf >= t].mean()))
                                 for t in CONFIDENCE}
        print(f"  {condition:13s} group badge: " + "  ".join(
            f"conf>={t}: shown {(conf >= t).mean():.0%}, correct {ok[conf >= t].mean():.3f}" for t in CONFIDENCE))
        if true_pairs:
            result["true_match_colour_bands"] = {b: float(np.mean([t == b for t in true_pairs]))
                                                 for b in ["very close", "slightly different", "different"]}
        evaluation[condition] = result
        print(f"  {condition:13s} type accuracy {result['type_accuracy']:.3f}   'same type' badge agrees with "
              f"truth {result['same_type_badge_agrees']:.3f}"
              + (f"   true match colour: {result['true_match_colour_bands']}" if true_pairs else ""))
    print(f"  (chance type accuracy: {1 / len(TYPE_WORDS):.3f})")

    with open(OUT, "w") as f:
        json.dump(dict(type_prompts=TYPE_WORDS, type_groups=GROUP_OF_TYPE,
                       type_badge=dict(level="group", min_confidence=BADGE_MIN_CONFIDENCE),
                       colour_bands=bands, evaluation=evaluation), f, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()