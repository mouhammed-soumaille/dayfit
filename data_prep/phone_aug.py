"""Turn a clean shop photo into a fake "phone photo".

Kaggle product images are a garment on a white background, so we:
  1. paste the garment onto a random background (replace near-white pixels)
  2. rotate it a bit
  3. crop it a bit (like an off-centre phone shot)
  4. change brightness and contrast (indoor lighting)
  5. blur it (hand shake / out of focus)
  6. save as a low-quality JPEG (phone compression)

Three severity levels are defined so we can later plot accuracy vs severity.
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SEVERITY = {
    #          max rotation, min crop keep, brightness range, max blur, jpeg quality
    "mild":   dict(rotate=5,  crop=0.95, bright=(0.85, 1.15), blur=0.5, jpeg=85),
    "medium": dict(rotate=15, crop=0.85, bright=(0.65, 1.35), blur=1.5, jpeg=70),
    "strong": dict(rotate=30, crop=0.75, bright=(0.45, 1.55), blur=3.0, jpeg=50),
}


def random_background(size, rng):
    """A random coloured gradient plus noise, standing in for a room/bed/floor."""
    w, h = size
    c1 = rng.integers(0, 256, 3)
    c2 = rng.integers(0, 256, 3)
    t = np.linspace(0, 1, h)[:, None, None]  # vertical gradient
    bg = (1 - t) * c1 + t * c2
    bg = np.broadcast_to(bg, (h, w, 3)) + rng.normal(0, 12, (h, w, 3))
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))


def paste_on_background(img, rng, white_threshold=235):
    """Replace the (near-)white background of a product photo.
    Note: white garments partly get replaced too; that's a known limitation."""
    arr = np.asarray(img)
    is_background = (arr > white_threshold).all(axis=2)
    mask = Image.fromarray(((~is_background) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.GaussianBlur(1))  # soften the edges
    bg = random_background(img.size, rng)
    bg.paste(img, (0, 0), mask)
    return bg


def make_phone_version(img, rng, severity="medium"):
    """img: PIL RGB image. rng: numpy Generator (for reproducibility)."""
    s = SEVERITY[severity]
    img = img.convert("RGB")
    w, h = img.size

    img = paste_on_background(img, rng)

    angle = rng.uniform(-s["rotate"], s["rotate"])
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False,
                     fillcolor=tuple(int(x) for x in rng.integers(0, 256, 3)))

    keep = rng.uniform(s["crop"], 1.0)
    cw, ch = int(w * keep), int(h * keep)
    left = rng.integers(0, w - cw + 1)
    top = rng.integers(0, h - ch + 1)
    img = img.crop((left, top, left + cw, top + ch))

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(*s["bright"]))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(*s["bright"]))
    # blur is defined in pixels at the models' 224-pixel input size, so the
    # amount of blur the model sees is the same whatever the photo's resolution
    blur_px = rng.uniform(0, s["blur"]) * max(w, h) / 224
    img = img.filter(ImageFilter.GaussianBlur(blur_px))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=s["jpeg"])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")