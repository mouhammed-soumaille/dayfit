"""Ensemble: join the embeddings of two (or more) models.

For each image we take each model's normalised embedding and glue them end to
end (concatenate), then divide by sqrt(number of models) so the result has
length 1 again. The dot product of two such vectors is then simply the AVERAGE
of the models' cosine similarities. If the models make different mistakes,
the average is more often right than either model alone.

Each part keeps its own saved catalog index, so nothing is re-embedded.
"""
import numpy as np

from common.interfaces import Retriever


class EnsembleRetriever(Retriever):
    def __init__(self, name, parts):
        """parts: list of OpenClipRetriever objects."""
        self.name = name
        self.parts = parts

    def build_index(self, catalog_df):
        for part in self.parts:
            part.build_index(catalog_df)
        self.item_ids = self.parts[0].item_ids
        self.features = np.hstack([p.features for p in self.parts]) / np.sqrt(len(self.parts))

    def search(self, image, k=5):
        query = np.concatenate([p.embed_query(image) for p in self.parts]) / np.sqrt(len(self.parts))
        scores = self.features @ query
        best = np.argsort(-scores)[:k]
        return [(self.item_ids[i], float(scores[i])) for i in best]
