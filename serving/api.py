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

Catalogue: DAYFIT_CATALOG=evaluation (default, 6,197 items) or app (~33k items, demo only); the badge
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from serving.badges import TypePredictor, colour_band, garment_colour, lab_to_hex
from serving.engine import SearchEngine

try:  # lets PIL open iPhone .heic photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MAX_BYTES = 10 * 1024 * 1024
K = 5
OUTSIDE_FILE = {"evaluation": "serving/outside.json", "app": "serving/outside_app.json"}
RERANK_FILE = "serving/rerank.json"           # type-aware ranking weight w (serving/type_rerank.py), app only

app = FastAPI(title="DAYFIT visual clothing search")
engine = SearchEngine()                        # fine-tuned model + catalogue index, loaded once
types = TypePredictor(engine.catalogue)        # pretrained model for the type badge, loaded once
with open("serving/calibration.json") as f:
    calibration = json.load(f)
with open("serving/badges.json") as f:
    badges = json.load(f)
with open(OUTSIDE_FILE[engine.catalogue]) as f:
    outside = json.load(f)                     # out-of-catalogue warning threshold (outside_check.py / app_badges_check.py)
rerank_w = 0.0                                 # evaluation catalogue: plain similarity, exactly as evaluated
if engine.catalogue == "app" and os.path.exists(RERANK_FILE):
    with open(RERANK_FILE) as f:
        rerank_w = json.load(f)["w"]
# position of each catalogue item's type in the type predictor's list (for P(item's type | photo))
type_index = np.array([types.types.index(engine.item.loc[i, "article_type"]) for i in engine.retriever.item_ids])


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
    seen = types.analyse(image)                         # group badge + out-of-catalogue check, one embedding
    # order: similarity + w * P(item's type | photo); w = 0 for the evaluation catalogue
    ranking = scores + rerank_w * seen["type_probs"][type_index]
    order = np.argsort(-ranking)[:K]
    best = float(scores.max())                          # "not sure" and shortlist use the plain similarity
    group, group_prob = seen["group"], seen["group_prob"]
    not_sold = seen["outside_prob"] >= outside["tau"]
    show_group = group_prob >= badges["type_badge"]["min_confidence"]
    query_colour = garment_colour(image)

    results = []
    for rank, i in enumerate(order, start=1):
        item_id = retriever.item_ids[i]
        row = engine.item.loc[item_id]
        item_colour = garment_colour(Image.open(row["image_path"]))
        band, delta_e = colour_band(query_colour, item_colour, badges["colour_bands"])
        item_group = types.group_of[row["article_type"]]
        results.append(dict(
            rank=rank, item_id=item_id, score=float(scores[i]),
            article_type=row["article_type"], color=row["color"], image_url=f"/catalog/{item_id}",
            type_badge=("same type" if item_group == group else "different type") if show_group else None,
            colour_badge=band, colour_delta_e=round(delta_e, 1), colour_hex=lab_to_hex(item_colour)))

    q = calibration["shortlist"]["q"]
    lam = calibration["not_sure"]["lambda"]
    return dict(
        results=results,
        query=dict(predicted_group=group if show_group else None, group_confidence=round(group_prob, 3),
                   colour_hex=lab_to_hex(query_colour),
                   not_in_catalogue=seen["outside_type"] if not_sold else None,
                   not_in_catalogue_prob=round(seen["outside_prob"], 3)),
        not_sure=best < lam,
        best_similarity=best,
        type_bonus_w=rerank_w,
        shortlist_size=int((scores >= best - q).sum()),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        note="Badges describe measured differences; the ranking comes from overall visual similarity. "
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
