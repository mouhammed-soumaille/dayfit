"""E5: the DAYFIT web page (Streamlit). The shopper uploads a photo and sees similar catalogue items.

The page only talks to the API (serving/api.py) over HTTP, like a real front end: the API is the
single place where searching, badges and uncertainty are computed. The uploaded photo stays in
memory (Streamlit keeps uploads in memory) and is sent to the API as raw bytes; it is never saved.

What the shopper sees:
  - a "not sure" banner when the best match is weak (risk control, calibrated on real photos)
  - the top 5 results, each with a type badge (only when the type prediction is confident)
    and a colour badge (very close / slightly different / different)
  - the shortlist size: how many catalogue items are about as similar as the best one
  - two optional filters that HIDE results without changing the ranking:
    "same type only" and "hide clearly different colours"
  - one honest line: badges describe differences, the ranking is overall visual similarity

Run (API first, in another terminal):
    uvicorn serving.api:app --port 8000
    streamlit run serving/ui.py
"""
import os

import requests
import streamlit as st

API = os.environ.get("DAYFIT_API", "http://localhost:8000")


def swatch(hex_code):
    """A small coloured square followed by its hex code (estimated from the photo)."""
    return (f'<span style="display:inline-block;width:14px;height:14px;background:{hex_code};'
            f'border:1px solid #888;vertical-align:middle"></span> <code>{hex_code}</code>')

st.set_page_config(page_title="DAYFIT", layout="wide")
st.title("DAYFIT: find a garment from a photo")
st.caption("Your photo is only used for this search. It is kept in memory and never stored.")

photo = st.file_uploader("Photo of the garment", type=["jpg", "jpeg", "png", "webp", "heic"])
col_a, col_b = st.columns(2)
same_type_only = col_a.toggle("Same type only")
hide_other_colours = col_b.toggle("Hide clearly different colours")

if photo is not None:
    left, right = st.columns([1, 3])
    left.image(photo.getvalue(), caption="Your photo")
    with st.spinner("Searching the catalogue..."):
        response = requests.post(f"{API}/search", data=photo.getvalue(), timeout=60)
    if response.status_code != 200:
        right.error(response.json().get("detail", "The search failed."))
        st.stop()
    answer = response.json()

    if answer["not_sure"]:
        right.warning("Not sure: none of these looks like a strong match. Treat them as loose "
                      "suggestions, or try a clearer photo of the garment alone.")
    group = answer["query"]["predicted_group"]
    right.write(f"Looks like: **{group}**" if group else "Garment type: not sure")
    right.markdown(f"Estimated colour of your garment: {swatch(answer['query']['colour_hex'])} "
                   "(from the photo; lighting can shift it)", unsafe_allow_html=True)
    right.write(f"About **{answer['shortlist_size']}** catalogue items are about as similar as the "
                f"best match (90% shortlist, calibrated on other photos; see note below).")

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
        col.markdown(f"**{r['rank']}. {r['article_type']}** ({r['color']})")
        if r["type_badge"]:
            col.caption(("✅ " if r["type_badge"] == "same type" else "⚠️ ") + r["type_badge"])
        col.caption({"very close": "🎨 colour very close", "slightly different": "🎨 colour slightly different",
                     "different": "🎨 colour different"}[r["colour_badge"]])
        col.markdown(swatch(r["colour_hex"]), unsafe_allow_html=True)
        col.caption(f"similarity {r['score']:.2f}")
    if hidden:
        right.caption(f"{hidden} result(s) hidden by your filters (the ranking itself is unchanged).")
    st.caption(f"{answer['note']} Search time: {answer['latency_ms']:.0f} ms.")
