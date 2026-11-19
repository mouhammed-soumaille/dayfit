"""E5: the DAYFIT web page (Streamlit). The shopper uploads a photo and sees similar catalogue items.

The page only talks to the API (serving/api.py) over HTTP, like a real front end: the API is the
single place where searching, badges and uncertainty are computed. The uploaded photo stays in
memory (Streamlit keeps uploads in memory) and is sent to the API as raw bytes; it is never saved.

What the shopper sees:
  - a "not sure" banner when the best match is weak (risk control, calibrated on real photos)
  - the top 5 results, each with a type badge (only when the type prediction is confident)
    and a colour badge (very close / slightly different / different); no colour badges when the photo
    has almost no colour (black-and-white photo, or a black/white/grey garment), see Finding 35
  - the shortlist size: how many catalogue items are about as similar as the best one, shown only
    when it is at most SHORTLIST_SHOW_MAX; above that the page only says that many items are about
    as similar (the API still returns the exact number)
  - two optional filters that HIDE results without changing the ranking:
    "same type only" and "hide clearly different colours"
  - one honest line: badges describe differences, the ranking is overall visual similarity
  - an optional crop box (two sliders): the shopper marks the garment, so the background and other
    clothes in the photo do not count. The crop is done in memory; with the default box (the whole
    photo) the original bytes are sent unchanged.

Run (API first, in another terminal):
    uvicorn serving.api:app --port 8000
    streamlit run serving/ui.py
"""
import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

try:                                       # iPhone photos (HEIC), as in the API
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

API = os.environ.get("DAYFIT_API", "http://localhost:8000")
PLURAL = {"skirt": "skirts", "jumpsuit": "jumpsuits", "handbag": "handbags", "hat": "hats",
          "scarf": "scarves", "belt": "belts", "socks": "socks", "sunglasses": "sunglasses",
          "swimsuit": "swimsuits", "watch": "watches", "necklace": "necklaces",
          "earrings": "earrings", "wallet": "wallets", "umbrella": "umbrellas"}
# A shortlist only helps if a shopper could actually look through it. 50 items is about one or two pages
# of a shop's product grid; a number like "6304" gives the shopper nothing they can act on.
SHORTLIST_SHOW_MAX = 50


def crop_box(image, x_range, y_range):
    """Slider percentages (0-100) -> pixel box (left, top, right, bottom)."""
    w, h = image.size
    return (round(w * x_range[0] / 100), round(h * y_range[0] / 100),
            round(w * x_range[1] / 100), round(h * y_range[1] / 100))


def swatch(hex_code):
    """A small coloured square followed by its hex code (estimated from the photo)."""
    return (f'<span style="display:inline-block;width:14px;height:14px;background:{hex_code};'
            f'border:1px solid #888;vertical-align:middle"></span> <code>{hex_code}</code>')

st.set_page_config(page_title="DAYFIT", layout="wide")
st.title("DAYFIT: find a garment from a photo")
st.caption("Your photo is only used for this search. It is kept in memory and never stored.")

photo = st.file_uploader("Photo of the garment", type=["jpg", "jpeg", "png", "webp", "heic"])
# Finding 37: on real street photos, searching only the garment instead of the whole outfit photo finds the
# exact item first 51% instead of 41% of the time, and the right kind of garment far more often
st.caption("Tip: photograph the garment alone, or mark it with the box after uploading. On real street "
           "photos this finds the exact item first about 51% of the time, against 41% for a whole-outfit photo.")
col_a, col_b = st.columns(2)
same_type_only = col_a.toggle("Same type only")
hide_other_colours = col_b.toggle("Hide clearly different colours")

if photo is not None:
    left, right = st.columns([1, 3])
    # optional crop box: two sliders in percent of the width and height (default: the whole photo)
    with left.expander("Mark the garment (recommended)", expanded=True):
        x_range = st.slider("Left - right (%)", 0, 100, (0, 100))
        y_range = st.slider("Top - bottom (%)", 0, 100, (0, 100))
    whole_photo = x_range == (0, 100) and y_range == (0, 100)
    if whole_photo:
        left.image(photo.getvalue(), caption="Your photo")
        body = photo.getvalue()                      # exactly the uploaded bytes, as before
    else:
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(photo.getvalue()))).convert("RGB")
        except OSError:
            right.error("This photo could not be opened for cropping; switch the box back to the whole photo.")
            st.stop()
        box = crop_box(image, x_range, y_range)
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            right.error("The box is too small. Make it bigger.")
            st.stop()
        preview = image.copy()
        ImageDraw.Draw(preview).rectangle(box, outline="red", width=max(3, image.width // 150))
        left.image(preview, caption="Your photo (only the red box is searched)")
        buffer = io.BytesIO()                        # in memory only, never written to disk
        image.crop(box).save(buffer, format="JPEG", quality=95)
        body = buffer.getvalue()
    with st.spinner("Searching the catalogue..."):
        response = requests.post(f"{API}/search", data=body, timeout=60)
    if response.status_code != 200:
        right.error(response.json().get("detail", "The search failed."))
        st.stop()
    answer = response.json()
    if "colour_judged" not in answer["query"]:          # an API started before the colour check (Finding 35)
        right.error("The API is an older version than this page. Stop it and start it again "
                    "(uvicorn serving.api:app --port 8000), then search again.")
        st.stop()

    not_sold = answer["query"]["not_in_catalogue"]
    if not_sold:
        right.info(f"This looks like a {not_sold}. Our catalogue has no {PLURAL[not_sold]}, so the results "
                   "below can only be loose suggestions of other garments.")
    if answer["not_sure"]:
        right.warning("Not sure: none of these looks like a strong match. Treat them as loose "
                      "suggestions, or try a clearer photo of the garment alone.")
    group = answer["query"]["predicted_group"]
    right.write(f"Looks like: **{group}**" if group else "Garment type: not sure")
    if answer["query"]["colour_judged"]:
        right.markdown(f"Estimated colour of your garment: {swatch(answer['query']['colour_hex'])} "
                       "(from the photo; lighting can shift it; marking the garment makes it more reliable)",
                       unsafe_allow_html=True)
    else:
        right.info("This photo has almost no colour (a black-and-white photo, or a black, white or grey "
                   "garment), so the colour can't be judged and colour badges are hidden.")
    size = answer["shortlist_size"]
    if size <= SHORTLIST_SHOW_MAX:
        items = "item is" if size == 1 else "items are"
        right.write(f"About **{size}** catalogue {items} about as similar as the best match "
                    "(90% shortlist, calibrated on other photos; see note below).")
    else:
        right.write("Many catalogue items are about as similar as the best match, so treat the "
                    "results below as similar options rather than the exact item.")

    shown = [r for r in answer["results"]
             if not (same_type_only and r["type_badge"] == "different type")
             and not (hide_other_colours and r["colour_badge"] == "different")]
    hidden = len(answer["results"]) - len(shown)
    if not shown:
        right.info("Your filters hide all 5 results. Switch a filter off to see them.")
    cols = right.columns(5)
    for col, r in zip(cols, shown):
        image = requests.get(f"{API}{r['image_url']}", timeout=30).content
        col.image(image)
        col.markdown(f"**{r['rank']}. {r.get('display_type', r['article_type'])}**")
        col.caption(f"listed as: {r['color']}")      # the shop's own colour label; the swatch below is measured
        if r["type_badge"]:
            col.caption(("✅ " if r["type_badge"] == "same type" else "⚠️ ") + r["type_badge"])
        if r["colour_badge"]:
            col.caption({"very close": "🎨 colour very close", "slightly different": "🎨 colour slightly different",
                         "different": "🎨 colour different"}[r["colour_badge"]])
        col.markdown(swatch(r["colour_hex"]), unsafe_allow_html=True)
        col.caption(f"similarity {r['score']:.2f}")
    if hidden:
        right.caption(f"{hidden} result(s) hidden by your filters (the ranking itself is unchanged).")
    st.caption(f"{answer['note']} Search time: {answer['latency_ms']:.0f} ms.")