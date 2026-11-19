"""Features for the learned re-ranker (task 5, step 1; pre-registered in report/results_log.md).

Problem (Finding 31): most misses are the right kind of garment but the wrong design, and one fingerprint
(the fine-tuned embedding) decides on its own what matters. Stage 1 = finetuned_random's top 20 distinct
pictures on the official 61,553-image corpus (saved by evaluation/lookbench_full.py). For every
(photo, candidate) pair this script measures several separate things, so that a model (rerank_model.py)
can LEARN from real photos how much each should count:
  ft_sim       similarity of the fine-tuned model (stage 1 score)
  ft_gap       ft_sim of rank 1 minus ft_sim of this candidate
  rank         stage 1 rank (1-20)
  pre_sim      similarity of the PRETRAINED FashionSigLIP
  dino_sim     similarity of DINOv2 (self-supervised, CLS token, 224 px)
  patch_score  "do the small parts match?": DINOv2 patch tokens (16 x 16 = 256 per image); for every patch
               of the photo, the best-matching patch of the candidate; the mean of those best matches.
               Aimed at prints and details, which a single fingerprint washes out.
  colour_sim   masked HSV colour histogram cosine (same feature as the classical baseline, Finding 42)
  delta_e      CIEDE2000 difference of the estimated garment colours (serving/badges.py garment_colour)
  group_agree  P(candidate's group | photo): zero-shot group probabilities of the photo (pretrained
               FashionSigLIP, app prompts) at the candidate's zero-shot group (works for noise images too,
               which have no labels)
  label        1 if the candidate picture shows the photo's true item, else 0

Sanity checks (stop if they fail): ft_sim recomputed from the saved embeddings equals the saved stage-1 score
of every pair; the share of label = 1 at rank 1 equals the saved full-corpus R@1 (0.501).

Run from the repo root (DINOv2 on 981 x 21 images at 224 px, about 10-20 minutes on MPS):
    python -m evaluation.rerank_features
Writes results/rerank_pairs.csv (one row per pair, 981 x 20).
"""
import hashlib

import numpy as np
import pandas as pd
import timm
import timm.data
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000

from evaluation.classical_baseline import features as classical_features
from retrieval.clip_retriever import pick_device
from serving.badges import APP_GROUP_OF_TYPE, APP_TYPE_WORDS, garment_colour

LB = "data/lookbench/real_streetlook"
NOISE = "data/lookbench/noise"
IDX = "models/index_lookbench"
DINO = "vit_base_patch14_dinov2.lvd142m"
OUT = "results/rerank_pairs.csv"


def load_dino(device):
    """DINOv2 ViT-B/14 at 224 px: 1 CLS token + 16 x 16 patch tokens per image."""
    model = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    config = timm.data.resolve_data_config({}, model=model)
    config.update(input_size=(3, 224, 224), crop_pct=1.0)   # whole image at 224 px (the model's default is 518)
    return model, timm.data.create_transform(**config)


def dino_tokens(model, transform, images, device):
    """List of PIL images -> (CLS vectors, patch vectors), all normalised to length 1."""
    x = torch.stack([transform(i) for i in images]).to(device)
    with torch.no_grad():
        tokens = model.forward_features(x)                 # (n, 1 + 256, 768)
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
    return tokens[:, 0], tokens[:, 1:]


def zero_shot_groups(siglip_vectors, scale):
    """Pretrained FashionSigLIP vectors -> probabilities of the 7 app groups (as TypePredictor.analyse)."""
    text = np.load("serving/type_text_app.npy")
    logits = scale * siglip_vectors @ text.T
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    groups = sorted(set(APP_GROUP_OF_TYPE.values()))
    member = np.array([[APP_GROUP_OF_TYPE[t] == g for g in groups] for t in APP_TYPE_WORDS], dtype=float)
    return probs @ member, groups


def main():
    gallery = pd.read_csv(f"{LB}/gallery.csv", dtype=str)
    queries = pd.read_csv(f"{LB}/queries.csv", dtype=str)
    noise = pd.read_csv(f"{NOISE}/gallery.csv", dtype=str)
    corpus = pd.concat([gallery, noise], ignore_index=True)
    row_of = {item: k for k, item in enumerate(corpus["item_id"])}
    labels = corpus.assign(picture=[hashlib.md5(open(p, "rb").read()).hexdigest() for p in corpus["image_path"]]) \
                   .groupby("picture")["label"].apply(set).to_dict()
    ranks = pd.read_csv("results/rankings/finetuned_random_lookbench_real_streetlook_image_full.csv")

    # saved embeddings (nothing re-embedded for these)
    ft_q = np.load(f"{IDX}/real_streetlook/finetuned_random_queries.npy")
    ft_c = np.concatenate([np.load(f"{IDX}/real_streetlook/finetuned_random.npy"), np.load(f"{IDX}/noise/finetuned_random.npy")])
    pre_q = np.load(f"{IDX}/real_streetlook/siglip_queries.npy")
    pre_c = np.concatenate([np.load(f"{IDX}/real_streetlook/fashion_siglip.npy"), np.load(f"{IDX}/noise/fashion_siglip.npy")])
    scale = float(open(f"{IDX}/real_streetlook/siglip_logit_scale.txt").read())
    colour_c = np.load(f"{IDX}/classical_full.npz")["colour"]
    q_groups, group_names = zero_shot_groups(pre_q, scale)
    query_of = {q: k for k, q in enumerate(queries["query_id"])}

    device = pick_device()
    dino, transform = load_dino(device)
    rows = []
    for n, (query_id, part) in enumerate(ranks.groupby("query_id", sort=False), start=1):
        k = query_of[query_id]
        q_image = Image.open(queries.loc[k, "image"]).convert("RGB")
        candidates = [row_of[i] for i in part["item_id"]]
        images = [q_image] + [Image.open(corpus.loc[c, "image_path"]).convert("RGB") for c in candidates]
        cls, patches = dino_tokens(dino, transform, images, device)
        dino_sim = (cls[1:] @ cls[0]).float().cpu().numpy()
        # patch score: for each photo patch, best cosine with any candidate patch; then the mean
        patch_score = torch.einsum("pd,cqd->cpq", patches[0], patches[1:]).max(dim=2).values.mean(dim=1).float().cpu().numpy()
        q_colour_hist = classical_features(queries.loc[k, "image"])[0]
        q_lab = garment_colour(q_image)
        c_groups, _ = zero_shot_groups(pre_c[candidates], scale)
        best_ft = None
        for j, (row, c) in enumerate(zip(part.itertuples(), candidates)):
            ft_sim = float(ft_q[k] @ ft_c[c])
            best_ft = ft_sim if best_ft is None else best_ft
            rows.append(dict(
                query_id=query_id, item_id=row.item_id, saved_score=row.score, rank=row.rank,
                ft_sim=ft_sim, ft_gap=best_ft - ft_sim, pre_sim=float(pre_q[k] @ pre_c[c]),
                dino_sim=float(dino_sim[j]), patch_score=float(patch_score[j]),
                colour_sim=float(q_colour_hist @ colour_c[c]),
                delta_e=float(deltaE_ciede2000(q_lab[None], garment_colour(images[j + 1])[None])[0]),
                group_agree=float(q_groups[k, c_groups[j].argmax()]),
                label=int(queries.loc[k, "label"] in labels[row.picture])))
        print(f"  {n}/{len(queries)}", end="\r")
    print()
    pairs = pd.DataFrame(rows)

    # checks
    diff = float((pairs["ft_sim"] - pairs["saved_score"]).abs().max())
    r1 = pairs.loc[pairs["rank"] == 1, "label"].mean()
    print(f"  check: recomputed fine-tuned similarity vs saved stage-1 score, max difference {diff:.2e}")
    print(f"  check: label at rank 1 = R@1 {r1:.4f} (saved full-corpus R@1 0.5010)")
    if diff > 1e-4 or abs(r1 - 0.5010) > 0.001:
        raise SystemExit("PROBLEM: pairs do not reproduce the saved stage-1 results; stopping.")
    in_top20 = pairs.groupby("query_id")["label"].max().mean()
    print(f"  true item somewhere in the top 20 (R@1 ceiling of any re-ranker): {in_top20:.3f}")
    pairs.drop(columns="saved_score").to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
