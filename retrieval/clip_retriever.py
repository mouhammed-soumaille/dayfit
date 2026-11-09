"""Retriever for any open_clip model (plain CLIP now, fashion models in week 3).

How it works:
  1. a pretrained network turns each image into an embedding (a vector of 512+ numbers)
  2. we normalise every embedding to length 1, so cosine similarity = dot product
  3. catalog embeddings are computed once and saved to models/index/<name>.npy,
     so later runs (and the app) just load them
  4. search = embed the query, dot product with all catalog embeddings, take the top k
"""
import os

import numpy as np
import open_clip
import torch
from PIL import Image

from common.interfaces import Retriever


def pick_device():
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    if torch.cuda.is_available():          # NVIDIA GPU (e.g. Colab)
        return "cuda"
    return "cpu"


class OpenClipRetriever(Retriever):
    def __init__(self, name, model_name, pretrained=None, index_dir="models/index", batch_size=64):
        self.name = name
        self.index_dir = index_dir
        self.batch_size = batch_size
        self.device = pick_device()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed(self, images):
        """list of PIL images -> numpy array (n, d) of normalised embeddings."""
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def build_index(self, catalog_df):
        self.item_ids = list(catalog_df["item_id"])
        emb_path = os.path.join(self.index_dir, f"{self.name}.npy")
        ids_path = os.path.join(self.index_dir, f"{self.name}_ids.txt")

        # reuse saved embeddings if they were made for exactly this catalog
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            with open(ids_path) as f:
                saved_ids = f.read().split("\n")
            if saved_ids == self.item_ids:
                self.features = np.load(emb_path)
                print(f"[{self.name}] loaded saved catalog embeddings {self.features.shape}")
                return

        print(f"[{self.name}] embedding {len(self.item_ids)} catalog images on {self.device} ...")
        paths = list(catalog_df["image_path"])
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            images = [Image.open(p) for p in paths[start:start + self.batch_size]]
            chunks.append(self.embed(images))
            print(f"  {min(start + self.batch_size, len(paths))}/{len(paths)}", end="\r")
        self.features = np.concatenate(chunks)

        os.makedirs(self.index_dir, exist_ok=True)
        np.save(emb_path, self.features)
        with open(ids_path, "w") as f:
            f.write("\n".join(self.item_ids))
        print(f"\n[{self.name}] saved {emb_path}")

    def search(self, image, k=5):
        query = self.embed([image])[0]
        scores = self.features @ query
        best = np.argsort(-scores)[:k]
        return [(self.item_ids[i], float(scores[i])) for i in best]