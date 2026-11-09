"""The one shared interface every search method follows.

Colour histogram, CLIP, FashionSigLIP, ensembles... are all just classes with
these two methods. The evaluation harness only ever calls build_index and
search, so it works for any retriever (even a random one).
"""


class Retriever:
    name = "base"  # short id used in result file names, e.g. "color_hist", "clip"

    def build_index(self, catalog_df):
        """Precompute features for every catalog image.

        catalog_df: pandas DataFrame with at least columns item_id, image_path.
        """
        raise NotImplementedError

    def search(self, image, k=5):
        """Find the k most similar catalog items.

        image: a PIL.Image (the raw query photo).
        Returns: list of (item_id, score) tuples, best match first.
        """
        raise NotImplementedError
