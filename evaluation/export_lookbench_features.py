"""Save LookBench embeddings once, so every fusion experiment can run in numpy (no model needed).

Saves to models/index_lookbench/real_streetlook/:
  finetuned_random_queries.npy   query embeddings of the recommended model (gallery ones already exist)
  siglip_queries.npy             query embeddings of the PRETRAINED FashionSigLIP (for zero-shot type
                                 probabilities with the app prompts, serving/type_text_app.npy)
  siglip_logit_scale.txt         the pretrained model's temperature (softmax scale)
  dinov2_gallery.npy / dinov2_queries.npy
                                 DINOv2 ViT-B/14 (self-supervised, 518 px input). DINOv2 is trained to
                                 tell apart individual images, not captions, and sees the photo at higher
                                 resolution than our CLIP models, so it may keep more design detail.

Sanity check: the finetuned_random query embeddings must reproduce the saved rank-1 picture
(results/rankings/finetuned_random_..._dedup.csv) for every query.

Run from the repo root (first run downloads DINOv2, ~350 MB):
    python -m evaluation.export_lookbench_features
"""
import hashlib
import os

import numpy as np
import pandas as pd
import timm
import timm.data
import torch
from PIL import Image

from evaluation.run_eval import RETRIEVERS
from evaluation.run_lookbench import use_index_dir
from retrieval.clip_retriever import pick_device
from serving.badges import TypePredictor

LB = "data/lookbench/real_streetlook"
OUT = "models/index_lookbench/real_streetlook"
DINO = "vit_base_patch14_dinov2.lvd142m"


def open_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


def embed_all(embed, paths, batch=32):
    chunks = []
    for start in range(0, len(paths), batch):
        chunks.append(embed(open_images(paths[start:start + batch])))
        print(f"  {min(start + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(chunks)


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)
    device = pick_device()

    # 1. recommended model: gallery from the saved index, queries embedded now
    model = RETRIEVERS["finetuned_random"]()
    use_index_dir(model, OUT)
    model.build_index(gallery)
    q_ft = embed_all(model.embed, list(queries["image"]))
    np.save(f"{OUT}/finetuned_random_queries.npy", q_ft)

    # check: reproduces the saved rank-1 picture of every query (dedup ranking)
    picture = np.array([hashlib.md5(open(p, "rb").read()).hexdigest() for p in gallery["image_path"]])
    top1 = picture[(q_ft @ model.features.T).argmax(axis=1)]
    saved = pd.read_csv("results/rankings/finetuned_random_lookbench_real_streetlook_image_dedup.csv")
    saved = saved[saved["rank"] == 1].set_index("query_id").loc[queries["query_id"], "picture"].to_numpy()
    agree = (top1 == saved).mean()
    print(f"check: rank-1 picture reproduced for {agree:.1%} of queries")
    if agree < 0.99:
        raise SystemExit("sanity check failed, stopping")
    del model

    # 2. pretrained FashionSigLIP query embeddings (zero-shot type probabilities)
    types = TypePredictor("app")
    def siglip(images):
        x = torch.stack([types.preprocess(i) for i in images]).to(device)
        with torch.no_grad():
            v = types.model.encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
        return v.float().cpu().numpy()
    np.save(f"{OUT}/siglip_queries.npy", embed_all(siglip, list(queries["image"])))
    with open(f"{OUT}/siglip_logit_scale.txt", "w") as f:
        f.write(str(float(types.model.logit_scale.exp())))
    del types

    # 3. DINOv2 (CLS embedding, normalised to length 1)
    dino = timm.create_model(DINO, pretrained=True, num_classes=0).to(device).eval()
    config = timm.data.resolve_data_config({}, model=dino)
    transform = timm.data.create_transform(**config)
    print(f"DINOv2 input size: {config['input_size']}")
    def dinov2(images):
        x = torch.stack([transform(i) for i in images]).to(device)
        with torch.no_grad():
            v = dino(x)
            v = v / v.norm(dim=-1, keepdim=True)
        return v.float().cpu().numpy()
    np.save(f"{OUT}/dinov2_gallery.npy", embed_all(dinov2, list(gallery["image_path"])))
    np.save(f"{OUT}/dinov2_queries.npy", embed_all(dinov2, list(queries["image"])))
    print("saved all embeddings to", OUT)


if __name__ == "__main__":
    main()
