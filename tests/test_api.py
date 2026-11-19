"""Tests for the API (serving/api.py). Loads the models once, so they take a few seconds.

Run from the repo root:
    python -m pytest tests/test_api.py -v
"""
import builtins
import io
import os
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from serving.api import app

client = TestClient(app)
QUERIES = pd.read_csv("data/queries.csv", dtype={"source_item_id": str}).set_index("query_id")
SAVED = pd.read_csv("results/finetuned_random_phone_medium.csv", dtype={"item_id": str})


def photo_bytes(query_id):
    with open(QUERIES.loc[query_id, "image_path"], "rb") as f:
        return f.read()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model"] == "finetuned_random"


def test_search_reproduces_saved_results():
    """The API must return the same top 5 as the evaluated model (results/...csv)."""
    for query_id in SAVED["query_id"].unique()[:3]:
        r = client.post("/search", content=photo_bytes(query_id))
        assert r.status_code == 200
        got = [x["item_id"] for x in r.json()["results"]]
        expected = list(SAVED[SAVED["query_id"] == query_id].sort_values("rank")["item_id"])
        assert got == expected


def test_search_writes_no_file(monkeypatch):
    """Privacy: during a search nothing is opened for writing and no temp file appears."""
    real_open = builtins.open
    writes = []

    def watching_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    temp_before = set(os.listdir(tempfile.gettempdir()))
    monkeypatch.setattr(builtins, "open", watching_open)
    big = photo_bytes(SAVED["query_id"].iloc[0]) + b"\0" * (2 * 1024 * 1024)   # > 1 MB upload
    r = client.post("/search", content=big)
    monkeypatch.setattr(builtins, "open", real_open)
    assert r.status_code == 200
    assert writes == []
    assert set(os.listdir(tempfile.gettempdir())) - temp_before == set()


def test_black_and_white_photo_gets_no_colour_badges():
    """Finding 35: a grayscale version of a normal photo -> colour not judged, no colour badges;
    the colour original still gets them."""
    original = photo_bytes(SAVED["query_id"].iloc[0])
    gray = ImageOps.grayscale(Image.open(io.BytesIO(original))).convert("RGB")
    buffer = io.BytesIO()                               # in memory only
    gray.save(buffer, format="JPEG", quality=95)
    colour_answer = client.post("/search", content=original).json()
    gray_answer = client.post("/search", content=buffer.getvalue()).json()
    assert colour_answer["query"]["colour_judged"] is True
    assert all(r["colour_badge"] is not None for r in colour_answer["results"])
    assert gray_answer["query"]["colour_judged"] is False
    assert gray_answer["query"]["colour_hex"] is None
    assert all(r["colour_badge"] is None for r in gray_answer["results"])


def test_too_large_is_refused():
    r = client.post("/search", content=b"\0" * (11 * 1024 * 1024))
    assert r.status_code == 413


def test_not_an_image_is_refused():
    r = client.post("/search", content=b"this is not a photo")
    assert r.status_code == 400


@pytest.mark.parametrize("item_id", ["does-not-exist"])
def test_unknown_catalogue_item(item_id):
    assert client.get(f"/catalog/{item_id}").status_code == 404