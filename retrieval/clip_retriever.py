"""Retriever for any open_clip model (plain CLIP, FashionSigLIP, FashionCLIP...).

How it works:
  1. a pretrained network turns each image into an embedding (a vector of 512+ numbers)
  2. we normalise every embedding to length 1, so cosine similarity = dot product
  3. catalog embeddings are computed once and saved to models/index/<index_name>.npy,
     so later runs (and the app) just load them
  4. search = embed the query, dot product with all catalog embeddings, take the top k

Options (each can be switched on/off, so the results table can show its effect):
  crop_queries : run crop_garment on each query photo before embedding it
  crop_catalog : also crop every catalog image the same way, so queries and
                 catalog are framed the same (crops are saved in data/crop_cache/)
  multiview    : embed 3 versions of every catalog image (original, padded to a
                 white square, centre crop) and average the 3 embeddings
"""
import os

import numpy as np
import open_clip
import torch
from PIL import Image

from common.interfaces import Retriever
from retrieval.preprocess import cached_crop, center_crop, crop_garment, pad_to_square


def pick_device():
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    if torch.cuda.is_available():          # NVIDIA GPU (e.g. Colab)
        return "cuda"
    return "cpu"


class OpenClipRetriever(Retriever):
    def __init__(self, name, model_name, pretrained=None, crop_queries=False,
                 crop_catalog=False, multiview=False, index_name=None,
                 checkpoint_path=None, index_dir="models/index", batch_size=64):
        """index_name: file name for the saved catalog embeddings (defaults to name).
        Variants that only change the QUERIES (e.g. crop_queries) can share an index.
        checkpoint_path: optional fine-tuned weights (Contract 4); empty = pretrained."""
        self.name = name
        self.crop_queries = crop_queries
        self.crop_catalog = crop_catalog
        self.multiview = multiview
        self.index_name = index_name or name
        self.index_dir = index_dir
        self.batch_size = batch_size
        self.device = pick_device()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state.get("state_dict", state))
            print(f"[{self.name}] loaded fine-tuned weights from {checkpoint_path}")
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed(self, images):
        """list of PIL images -> numpy array (n, d) of normalised embeddings."""
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def catalog_views(self, image_path):
        """The image(s) we embed for one catalog item."""
        image = cached_crop(image_path) if self.crop_catalog else Image.open(image_path).convert("RGB")
        if self.multiview:
            return [image, pad_to_square(image), center_crop(image)]
        return [image]

    def embed_catalog(self, paths):
        """Embed a list of catalog image paths (averaging the views if multiview)."""
        n_views = 3 if self.multiview else 1
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            views = []
            for p in paths[start:start + self.batch_size]:
                views.extend(self.catalog_views(p))
            emb = self.embed(views)                                     # (n_images * n_views, d)
            emb = emb.reshape(-1, n_views, emb.shape[1]).mean(axis=1)   # average the views
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)      # back to length 1
            chunks.append(emb)
            print(f"  {min(start + self.batch_size, len(paths))}/{len(paths)}", end="\r")
        return np.concatenate(chunks)

    def build_index(self, catalog_df):
        self.item_ids = list(catalog_df["item_id"])
        emb_path = os.path.join(self.index_dir, f"{self.index_name}.npy")
        # The saved embeddings are only reused if they were made from exactly the
        # same image FILES. Checking item_ids alone is not enough: swapping the
        # catalog to high-res images keeps the ids but changes every image.
        paths_path = os.path.join(self.index_dir, f"{self.index_name}_paths.txt")
        image_paths = list(catalog_df["image_path"])

        if os.path.exists(emb_path) and os.path.exists(paths_path):
            with open(paths_path) as f:
                saved_paths = f.read().split("\n")
            if saved_paths == image_paths:
                self.features = np.load(emb_path)
                print(f"[{self.name}] loaded saved catalog embeddings {self.features.shape}")
                return

        what = " + ".join(x for x, on in [("cropped", self.crop_catalog),
                                          ("multi-view", self.multiview)] if on) or "plain"
        print(f"[{self.name}] embedding {len(image_paths)} catalog images ({what}) on {self.device} ...")
        self.features = self.embed_catalog(image_paths)

        os.makedirs(self.index_dir, exist_ok=True)
        np.save(emb_path, self.features)
        with open(paths_path, "w") as f:
            f.write("\n".join(image_paths))
        print(f"\n[{self.name}] saved {emb_path}")

    def embed_query(self, image):
        """One query photo -> one normalised embedding (cropped first if crop_queries)."""
        if self.crop_queries:
            image = crop_garment(image)
        return self.embed([image])[0]

    def search(self, image, k=5):
        query = self.embed_query(image)
        scores = self.features @ query
        best = np.argsort(-scores)[:k]
        return [(self.item_ids[i], float(scores[i])) for i in best]
