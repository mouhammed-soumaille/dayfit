"""Baseline: compare images by their colour histograms.

For each image:
  1. convert to HSV (hue = which colour, saturation = how vivid, value = how bright)
  2. count pixels in 8 x 4 x 4 = 128 colour bins
  3. normalise the 128 counts to length 1, so cosine similarity = dot product

Search = dot product between the query histogram and every catalog histogram.
It knows nothing about shape, so a red T-shirt and red shoes look the same to it.
"""
import numpy as np
from PIL import Image

from common.interfaces import Retriever

BINS = (8, 4, 4)  # hue, saturation, value


def color_histogram(image):
    """PIL image -> normalised 128-number vector."""
    hsv = np.asarray(image.convert("RGB").convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=BINS, range=[(0, 256)] * 3)
    hist = hist.flatten()
    return hist / (np.linalg.norm(hist) + 1e-12)


class ColorHistRetriever(Retriever):
    def __init__(self, crop_queries=False):
        self.crop_queries = crop_queries
        self.name = "color_hist_crop" if crop_queries else "color_hist"

    def build_index(self, catalog_df):
        self.item_ids = list(catalog_df["item_id"])
        self.features = np.stack([
            color_histogram(Image.open(path)) for path in catalog_df["image_path"]
        ])  # shape: (number of catalog images, 128)

    def search(self, image, k=5):
        if self.crop_queries:
            from retrieval.preprocess import crop_garment
            image = crop_garment(image)
        query = color_histogram(image)
        scores = self.features @ query          # cosine similarity with every catalog image
        best = np.argsort(-scores)[:k]          # indices of the k highest scores
        return [(self.item_ids[i], float(scores[i])) for i in best]