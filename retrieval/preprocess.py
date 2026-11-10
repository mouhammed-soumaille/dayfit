"""Garment cropping for query photos.

crop_garment(image):
  1. rembg (a pretrained segmentation network) cuts out the main object
     and makes everything else transparent
  2. we find the bounding box of the non-transparent pixels and crop to it
     (plus a small margin)
  3. we paste the garment on a WHITE background, because catalog images
     are product shots on white: this makes the query look like the catalog

If nothing is found (e.g. rembg removes everything), the original image is returned.

To look at some examples:
    python -m retrieval.preprocess        # saves before/after pairs to data/crop_examples/
"""
import os

import numpy as np
from PIL import Image

_session = None  # the rembg model, loaded once on first use


def _remove_background(image):
    global _session
    from rembg import new_session, remove  # imported here so other code works without rembg
    if _session is None:
        _session = new_session("u2net")    # downloads ~170 MB the first time
    return remove(image, session=_session)  # RGBA image, background transparent


def crop_to_alpha(rgba, margin=0.05, min_alpha=128):
    """Crop an RGBA image to its visible pixels and put it on white.
    Returns None if (almost) nothing is visible."""
    alpha = np.asarray(rgba)[:, :, 3]
    ys, xs = np.where(alpha >= min_alpha)
    if len(xs) < 0.01 * alpha.size:  # less than 1% of the image left -> give up
        return None

    garment = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # Put the garment in the middle of a white SQUARE with a margin around it.
    # Why square: CLIP-style models centre-crop every image to a square, so a
    # tall, tight crop would get its top and bottom cut off.
    gw, gh = garment.size
    side = int(max(gw, gh) * (1 + 2 * margin))
    white = Image.new("RGB", (side, side), (255, 255, 255))
    white.paste(garment, ((side - gw) // 2, (side - gh) // 2), garment)  # alpha = paste mask
    return white


def crop_garment(image):
    """PIL image -> PIL image cropped to the garment, on a white background."""
    image = image.convert("RGB")
    cropped = crop_to_alpha(_remove_background(image))
    return cropped if cropped is not None else image


def cached_crop(image_path, cache_dir="data/crop_cache"):
    """crop_garment for an image FILE, saving the result so each image is only
    cropped once (rembg is slow, ~0.25 s per image). Used for catalog images."""
    safe_name = image_path.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(cache_dir, safe_name)
    if os.path.exists(out_path):
        return Image.open(out_path).convert("RGB")
    os.makedirs(cache_dir, exist_ok=True)
    cropped = crop_garment(Image.open(image_path))
    cropped.save(out_path, quality=95)
    return cropped


# ---- extra "views" of a catalog image, for multi-view embeddings ----

def pad_to_square(image, fill=(255, 255, 255)):
    """Put the whole image in the middle of a white square, so the model's
    centre crop to a square does not cut off the top and bottom."""
    image = image.convert("RGB")
    side = max(image.size)
    square = Image.new("RGB", (side, side), fill)
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


def center_crop(image, keep=0.8):
    """Keep the central `keep` fraction of the width and height (a zoomed-in view)."""
    image = image.convert("RGB")
    w, h = image.size
    cw, ch = int(w * keep), int(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


if __name__ == "__main__":
    import pandas as pd

    queries = pd.read_csv("data/queries.csv")
    phone = queries[queries["condition"] == "phone_medium"].head(8)
    out_dir = "data/crop_examples"
    os.makedirs(out_dir, exist_ok=True)
    for q in phone.itertuples():
        before = Image.open(q.image_path).convert("RGB")
        after = crop_garment(before).resize(before.size)
        pair = Image.new("RGB", (before.width * 2, before.height), "white")
        pair.paste(before, (0, 0))
        pair.paste(after, (before.width, 0))
        pair.save(os.path.join(out_dir, f"{q.query_id}.jpg"))
    print(f"saved {len(phone)} before/after pairs to {out_dir}/")