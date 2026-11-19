"""E4: the DAYFIT web API.

  POST /search          body = the photo's raw bytes (any common image format, max 10 MB)
                        -> JSON: top 5 catalogue items with match badges, the conformal
                           shortlist size, and a "not sure" flag
  GET  /catalog/{id}    the catalogue image of one item (for the UI)
  GET  /health          is the service up, which model, how big the catalogue is

Privacy: the uploaded photo is NEVER written to disk. We read the request body ourselves,
in chunks, into memory (FastAPI's usual UploadFile would silently spill uploads over 1 MB
into a temporary FILE), decode it with PIL from memory, and drop it when the request ends.
Nothing about the photo is logged. tests/test_api.py checks that no file is written.

Uncertainty (constants from serving/calibration.json, calibrated on LookBench real photos):
  - not_sure: the best similarity is below lambda (risk control at alpha 0.1)
  - shortlist_size: number of catalogue items within q of the best similarity (conformal, alpha 0.1)
  Neither guarantee formally holds here (different gallery); the response says so.

Colour (Finding 35): if the photo has almost no colour anywhere (black-and-white photo, or a black,
white or grey garment on a plain background), the colour estimate means nothing, so query.colour_judged
is false and no colour badges are given (colour_badge = None). Threshold from serving/colour_check.py.

Type model (latency, Finding 45): DAYFIT_TYPE_MODEL=off skips the second model (pretrained FashionSigLIP).
Then there is no type badge, no group bonus (plain similarity order) and no out-of-catalogue warning.

Catalogue: DAYFIT_CATALOG=evaluation (default, 6,197 items) or app (54,178 items, demo only); the badge
vocabulary and the out-of-catalogue threshold follow the catalogue.

Run from the repo root:
    uvicorn serving.api:app --port 8000
    DAYFIT_CATALOG=app uvicorn serving.api:app --port 8000
"""
import io
import json
import os
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from serving.badges import VOCAB, TypePredictor, colour_band, garment_colour, lab_to_hex
from serving.colour_check import photo_numbers
from serving.engine import SearchEngine

try:  # lets PIL open iPhone .heic photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MAX_BYTES = 10 * 1024 * 1024
K = 5
KIDS = ["Boys", "Girls"]                        # app catalogue is for adults (data_prep/app_catalog.py)
HIDDEN_FILE = "serving/app_hidden.csv"          # data_prep/app_cleanup.py; only the kids_name rule ships (Finding 36)
SAME_PRODUCT = 0.97                            # results this similar to an earlier result are the same product photo
OUTSIDE_FILE = {"evaluation": "serving/outside.json", "app": "serving/outside_app.json"}
RERANK_FILE = "serving/rerank.json"           # type-aware ranking weight w (serving/type_rerank.py), app only

app = FastAPI(title="DAYFIT visual clothing search")
engine = SearchEngine()                        # fine-tuned model + catalogue index, loaded once
TYPE_MODEL = os.environ.get("DAYFIT_TYPE_MODEL", "on") != "off"
types = TypePredictor(engine.catalogue) if TYPE_MODEL else None   # second model: type badge, bonus, warning
TYPE_NAMES = list(VOCAB[engine.catalogue]["types"])   # same order as the type model's prompts
GROUP_OF = VOCAB[engine.catalogue]["groups"]
with open("serving/calibration.json") as f:
    calibration = json.load(f)
with open("serving/badges.json") as f:
    badges = json.load(f)
with open("serving/colour_check.json") as f:
    # only the black-and-white threshold is used; the darkness flag was dropped (Finding 35)
    colourfulness_min = json.load(f)["colourfulness_min"]
with open(OUTSIDE_FILE[engine.catalogue]) as f:
    outside = json.load(f)                     # out-of-catalogue warning threshold (outside_check.py / app_badges_check.py)
rerank_w = 0.0                                 # evaluation catalogue: plain similarity, exactly as evaluated
if engine.catalogue == "app" and os.path.exists(RERANK_FILE):
    with open(RERANK_FILE) as f:
        rerank_w = json.load(f)["w"]
# position of each catalogue item's type in the type predictor's list (for P(item's type | photo))
type_index = np.array([TYPE_NAMES.index(engine.item.loc[i, "article_type"]) for i in engine.retriever.item_ids])
# GROUP-level bonus (Finding 32: tested on held-out real photos, group P@5 0.871 -> 0.905, R@1 unchanged):
# P(item's group | photo) = sum of the type probabilities of the types in the item's group
GROUP_NAMES = sorted(set(GROUP_OF.values()))
group_member = np.array([[GROUP_OF[t] == g for g in GROUP_NAMES] for t in TYPE_NAMES], dtype=float)
group_index = np.array([GROUP_NAMES.index(GROUP_OF[TYPE_NAMES[k]]) for k in type_index])
# app catalogue only: hide kids' items, by Kaggle gender (Boys/Girls) and by product name (Finding 36:
# many kids' items are listed as Men/Women/Unisex but their name says Kids, Infant, Boys, ...)
hidden = np.zeros(len(engine.retriever.item_ids), dtype=bool)
if engine.catalogue == "app":
    hidden = engine.item.loc[engine.retriever.item_ids, "gender"].isin(KIDS).to_numpy()
    flagged = pd.read_csv(HIDDEN_FILE, dtype=str)
    kids_by_name = set(flagged.loc[flagged["reasons"].str.contains("kids_name"), "item_id"])
    hidden = hidden | np.array([i in kids_by_name for i in engine.retriever.item_ids])   # new array (pandas gives a read-only one)


def pick_results(ranking):
    """The K best catalogue items by ranking, skipping hidden items and (app catalogue only) items that are
    nearly identical to a result already chosen (the same product photographed again)."""
    features = engine.retriever.features
    chosen = []
    for i in np.argsort(-ranking):
        if hidden[i]:
            continue
        if engine.catalogue == "app" and any(features[i] @ features[j] >= SAME_PRODUCT for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) == K:
            return chosen
    return chosen


async def read_photo(request):
    """Read the body into memory in chunks, refusing anything above MAX_BYTES."""
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="photo too large (max 10 MB)")
    try:
        image = Image.open(io.BytesIO(bytes(data)))
        image = ImageOps.exif_transpose(image).convert("RGB")   # respect the phone's rotation
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="the body is not an image we can read")
    return image


def search_photo(image):
    """Everything the app shows for one photo. Pure function of the image (no files written)."""
    start = time.perf_counter()
    retriever = engine.retriever
    query = retriever.embed_query(image)
    scores = retriever.features @ query                 # similarity with every catalogue item
    if types is not None:
        seen = types.analyse(image)                     # group badge + out-of-catalogue check, one embedding
        # order: similarity + w * P(item's group | photo); w = 0 for the evaluation catalogue
        group_probs = seen["type_probs"] @ group_member
        ranking = scores + rerank_w * group_probs[group_index]
    else:                                               # type model off: plain similarity, no type information
        seen = dict(group=None, group_prob=0.0, outside_type=None, outside_prob=0.0)
        ranking = scores
    order = pick_results(ranking)
    visible = scores[~hidden]
    best = float(visible.max())                         # "not sure" and shortlist use the plain similarity
    group, group_prob = seen["group"], seen["group_prob"]
    not_sold = types is not None and seen["outside_prob"] >= outside["tau"]
    show_group = group_prob >= badges["type_badge"]["min_confidence"]
    query_colour = garment_colour(image)
    colourfulness, _ = photo_numbers(image)
    colour_judged = colourfulness >= colourfulness_min  # False: almost no colour anywhere in the photo

    results = []
    for rank, i in enumerate(order, start=1):
        item_id = retriever.item_ids[i]
        row = engine.item.loc[item_id]
        item_colour = garment_colour(Image.open(row["image_path"]))
        band, delta_e = colour_band(query_colour, item_colour, badges["colour_bands"])
        if not colour_judged:                           # no colour in the photo: no colour badge
            band, delta_e = None, None
        item_group = GROUP_OF[row["article_type"]]
        # name shown to the shopper: H&M's own product type if the item comes from H&M (data_prep/app_add_hm.py),
        # otherwise the catalogue type; badges and ranking always use article_type
        shop_type = row.get("hm_type")
        display_type = shop_type if isinstance(shop_type, str) else row["article_type"]
        results.append(dict(
            rank=rank, item_id=item_id, score=float(scores[i]),
            article_type=row["article_type"], display_type=display_type, color=row["color"],
            image_url=f"/catalog/{item_id}",
            type_badge=("same type" if item_group == group else "different type") if show_group else None,
            colour_badge=band, colour_delta_e=None if delta_e is None else round(delta_e, 1),
            colour_hex=lab_to_hex(item_colour)))

    q = calibration["shortlist"]["q"]
    lam = calibration["not_sure"]["lambda"]
    return dict(
        results=results,
        query=dict(predicted_group=group if show_group else None, group_confidence=round(group_prob, 3),
                   colour_hex=lab_to_hex(query_colour) if colour_judged else None,
                   colour_judged=bool(colour_judged),
                   not_in_catalogue=seen["outside_type"] if not_sold else None,
                   not_in_catalogue_prob=round(seen["outside_prob"], 3)),
        not_sure=best < lam,
        best_similarity=best,
        type_bonus_w=rerank_w if types is not None else 0.0,
        shortlist_size=int((visible >= best - q).sum()),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        note=("Order: visual similarity plus a small bonus for the kind of garment (group) the photo looks like. "
              if rerank_w and types is not None else "")
             + ("Type model off: no type badge, no group bonus, no out-of-catalogue warning. " if types is None else "")
             + "Badges describe measured differences; the ranking comes from overall visual similarity. "
             + calibration["limitation"])


@app.post("/search")
async def search(request: Request):
    image = await read_photo(request)
    return search_photo(image)


@app.get("/catalog/{item_id}")
def catalog_image(item_id: str):
    if item_id not in engine.item.index:
        raise HTTPException(status_code=404, detail="unknown item")
    return FileResponse(engine.item.loc[item_id, "image_path"])


@app.get("/health")
def health():
    return dict(status="ok", model=engine.method, catalogue=engine.catalogue, catalogue_items=len(engine.item))