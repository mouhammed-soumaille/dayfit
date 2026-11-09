"""A retriever that ignores the image and returns random items.
Used to test the evaluation harness: it should score at chance level."""
import numpy as np

from common.interfaces import Retriever


class RandomRetriever(Retriever):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.item_ids = []

    def build_index(self, catalog_df):
        self.item_ids = list(catalog_df["item_id"])

    def search(self, image, k=5):
        picks = self.rng.choice(len(self.item_ids), size=k, replace=False)
        scores = sorted(self.rng.random(k), reverse=True)
        return [(self.item_ids[i], float(s)) for i, s in zip(picks, scores)]
