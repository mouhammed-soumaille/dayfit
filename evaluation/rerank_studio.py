"""Re-ranker with extra training photos from LookBench real_studio_flat (task 5 follow-up; pre-registered).

Problem (Finding 43): the re-ranker gained +2.1 points R@1 on test, not significant, and its learning curve was
still rising at 500 labelled photos. LookBench has a second real subset, real_studio_flat (1,011 studio and
flat-lay photos with their own 3,951-item gallery). Its pairs are added to the TRAINING data only.

Steps:
  1. leakage check: no training photo may show a real_streetlook TEST item. A studio photo is dropped if its own
     image, or any gallery picture of its true item, is byte-identical or a near-duplicate (pretrained
     FashionSigLIP cosine >= 0.97, the near-duplicate rule of Finding 18) of a test query crop or of a gallery
     picture of a test item;
  2. studio pairs: stage 1 = finetuned_random's top 20 distinct pictures on real_studio_flat's official corpus
     (its gallery + the same 58,275 noise images); the same 9 features as evaluation/rerank_features.py;
  3. the model chosen in Finding 43 (boosted trees, same fixed settings) is trained on dev + studio pairs:
     dev 5-fold CV by cluster (studio pairs always in the training folds) for comparison, then ONCE on test.
Tests: paired cluster bootstrap vs stage 1 on test; Holm over the two re-rankers taken to test (Finding 43's
and this one); also this one vs Finding 43's, descriptive.

Sanity check (stops if it fails): the dev-only model reproduces Finding 43's test R@1 (0.520).

Run from the repo root (downloads real_studio_flat, ~330 MB; embeddings and DINOv2 pairs, ~20-30 min on MPS):
    python -m evaluation.rerank_studio
Writes results/rerank_studio_pairs.csv, results/rerank_studio_leakage.csv, results/rerank_studio.csv.
"""
import hashlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from evaluation.classical_baseline import features as classical_features
from evaluation.cluster_bootstrap import cluster_bootstrap
from evaluation.fusion_test_stats import holm
from evaluation.rerank_features import dino_tokens, load_dino, zero_shot_groups
from evaluation.rerank_model import FEATURES, hits, new_model, stage1_score
from evaluation.run_eval import RETRIEVERS
from evaluation.run_lookbench import prepare, use_index_dir
from retrieval.clip_retriever import pick_device
from serving.badges import garment_colour

IDX = "models/index_lookbench"
SL = "data/lookbench/real_streetlook"
NEAR_DUPLICATE = 0.97
CHOSEN = "gradient-boosted trees"                 # chosen on dev in Finding 43


def studio_model():
    """Finding 43's boosted trees with the SAME settings. sklearn would switch on early stopping by itself above
    10,000 training rows (dev alone has exactly 10,000, so Finding 43 had none); it is switched off here so the
    settings stay identical."""
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0,
                                          early_stopping=False)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def embed(retriever, paths):
    return np.stack([retriever.embed_query(Image.open(p).convert("RGB")) for p in paths])


def corpus_features(method, gallery, noise):
    """Studio gallery embeddings (made once, saved) + the saved noise embeddings."""
    retriever = RETRIEVERS[method]()
    use_index_dir(retriever, f"{IDX}/real_studio_flat")
    retriever.build_index(gallery)
    own = retriever.features
    use_index_dir(retriever, f"{IDX}/noise")
    retriever.build_index(noise)
    return retriever, np.concatenate([own, retriever.features])


def leakage(st_gallery, st_queries, st_gal_pre, st_q_pre):
    """Which studio photos must be dropped because they (or their item's pictures) show a streetlook TEST item."""
    sl_gallery = pd.read_csv(f"{SL}/gallery.csv", dtype=str)
    sl_queries = pd.read_csv(f"{SL}/queries.csv", dtype=str).merge(pd.read_csv(f"{SL}/split.csv"), on="query_id")
    test_q = (sl_queries["side"] == "test").to_numpy()
    test_g = sl_gallery["label"].isin(set(sl_queries.loc[test_q, "label"])).to_numpy()
    bad = np.concatenate([np.load(f"{IDX}/real_streetlook/fashion_siglip.npy")[test_g],
                          np.load(f"{IDX}/real_streetlook/siglip_queries.npy")[test_q]])
    bad_md5 = set(md5(p) for p in sl_gallery.loc[test_g, "image_path"]) | set(md5(p) for p in sl_queries.loc[test_q, "image"])
    q_hit = ((st_q_pre @ bad.T).max(axis=1) >= NEAR_DUPLICATE) | np.array([md5(p) in bad_md5 for p in st_queries["image"]])
    g_hit = ((st_gal_pre @ bad.T).max(axis=1) >= NEAR_DUPLICATE) | np.array([md5(p) in bad_md5 for p in st_gallery["image_path"]])
    bad_items = set(st_gallery.loc[g_hit, "label"])
    drop = q_hit | st_queries["label"].isin(bad_items).to_numpy()
    report = pd.DataFrame([dict(studio_queries=len(st_queries), query_image_hits=int(q_hit.sum()),
                                gallery_picture_hits=int(g_hit.sum()), items_with_a_hit=len(bad_items),
                                queries_dropped=int(drop.sum()), queries_kept=int((~drop).sum()))])
    return drop, report


def studio_pairs(st_gallery, st_queries, noise):
    """Stage 1 + the 9 features for every kept studio photo (same definitions as rerank_features.py)."""
    corpus = pd.concat([st_gallery, noise], ignore_index=True)
    corpus["picture"] = [md5(p) for p in corpus["image_path"]]
    picture = corpus["picture"].to_numpy()
    labels = corpus.groupby("picture")["label"].apply(set).to_dict()
    ft, ft_c = corpus_features("finetuned_random", st_gallery, noise)
    pre, pre_c = corpus_features("fashion_siglip", st_gallery, noise)
    ft_q, pre_q = embed(ft, st_queries["image"]), embed(pre, st_queries["image"])
    scale = float(open(f"{IDX}/real_streetlook/siglip_logit_scale.txt").read())
    noise_colour = np.load(f"{IDX}/classical_full.npz")["colour"][3278:]          # rows after the streetlook gallery
    q_groups, _ = zero_shot_groups(pre_q, scale)
    device = pick_device()
    dino, transform = load_dino(device)
    n_own = len(st_gallery)
    rows = []
    for k, q in enumerate(st_queries.itertuples()):
        scores = ft_c @ ft_q[k]
        best = np.argpartition(-scores, 299)[:300]
        candidates, seen = [], []
        for i in best[np.argsort(-scores[best])]:
            if picture[i] not in seen:
                seen.append(picture[i])
                candidates.append(i)
                if len(seen) == 20:
                    break
        q_image = Image.open(q.image).convert("RGB")
        images = [q_image] + [Image.open(corpus.loc[c, "image_path"]).convert("RGB") for c in candidates]
        cls, patches = dino_tokens(dino, transform, images, device)
        dino_sim = (cls[1:] @ cls[0]).float().cpu().numpy()
        patch_score = torch.einsum("pd,cqd->cpq", patches[0], patches[1:]).max(dim=2).values.mean(dim=1).float().cpu().numpy()
        q_hist, q_lab = classical_features(q.image)[0], garment_colour(q_image)
        c_groups, _ = zero_shot_groups(pre_c[candidates], scale)
        top_ft = float(scores[candidates[0]])
        for j, c in enumerate(candidates):
            c_hist = classical_features(corpus.loc[c, "image_path"])[0] if c < n_own else noise_colour[c - n_own]
            rows.append(dict(query_id=f"studio_{q.query_id}", rank=j + 1, ft_sim=float(scores[c]),
                             ft_gap=top_ft - float(scores[c]), pre_sim=float(pre_q[k] @ pre_c[c]),
                             dino_sim=float(dino_sim[j]), patch_score=float(patch_score[j]),
                             colour_sim=float(q_hist @ c_hist),
                             delta_e=float(deltaE_ciede2000(q_lab[None], garment_colour(images[j + 1])[None])[0]),
                             group_agree=float(q_groups[k, c_groups[j].argmax()]),
                             label=int(q.label in labels[picture[c]])))
        print(f"  studio pairs {k + 1}/{len(st_queries)}", end="\r")
    print()
    return pd.DataFrame(rows)


def main():
    st_gallery, st_queries = prepare("real_studio_flat")
    noise = pd.read_csv("data/lookbench/noise/gallery.csv", dtype=str)
    pairs = pd.read_csv("results/rerank_pairs.csv").merge(pd.read_csv(f"{SL}/split.csv"), on="query_id")
    dev, test = pairs[pairs["side"] == "dev"].reset_index(drop=True), pairs[pairs["side"] == "test"].reset_index(drop=True)

    # sanity check: the dev-only model gives Finding 43's test R@1
    base_model = new_model(CHOSEN).fit(dev[FEATURES], dev["label"])
    base_hit1, _ = hits(test, base_model.predict_proba(test[FEATURES])[:, 1])
    print(f"  check: dev-only re-ranker test R@1 {base_hit1.mean():.4f} (Finding 43: 0.5198)")
    if abs(base_hit1.mean() - 0.5198) > 0.001:
        raise SystemExit("PROBLEM: Finding 43 not reproduced; stopping.")

    # 1. leakage check (pretrained FashionSigLIP embeddings of the studio photos and gallery)
    pre = RETRIEVERS["fashion_siglip"]()
    use_index_dir(pre, f"{IDX}/real_studio_flat")
    pre.build_index(st_gallery)
    drop, report = leakage(st_gallery, st_queries, pre.features, embed(pre, st_queries["image"]))
    report.to_csv("results/rerank_studio_leakage.csv", index=False)
    print(report.to_string(index=False))
    del pre

    # 2. studio training pairs
    studio = studio_pairs(st_gallery, st_queries[~drop].reset_index(drop=True), noise)
    studio.to_csv("results/rerank_studio_pairs.csv", index=False)
    print(f"  studio: {studio['query_id'].nunique()} photos, true item in top 20 for "
          f"{studio.groupby('query_id')['label'].max().mean():.3f}")

    # 3a. dev CV with studio pairs always in training (for comparison with Finding 43's 0.508)
    oof = np.zeros(len(dev))
    for train, held in GroupKFold(n_splits=5).split(dev, groups=dev["cluster"]):
        both = pd.concat([dev.iloc[train][FEATURES + ["label"]], studio[FEATURES + ["label"]]])
        model = studio_model().fit(both[FEATURES], both["label"])
        oof[held] = model.predict_proba(dev.iloc[held][FEATURES])[:, 1]
    dev_cv = hits(dev, oof)[0].mean()

    # 3b. TEST, once
    both = pd.concat([dev[FEATURES + ["label"]], studio[FEATURES + ["label"]]])
    model = studio_model().fit(both[FEATURES], both["label"])
    hit1, hit5 = hits(test, model.predict_proba(test[FEATURES])[:, 1])
    s1_hit1, s1_hit5 = hits(test, stage1_score(test))
    idx = hit1.index
    clusters = test.groupby("query_id")["cluster"].first().loc[idx].to_numpy()
    new_vs_s1 = cluster_bootstrap(hit1.to_numpy(), s1_hit1.loc[idx].to_numpy(), clusters)
    old_vs_s1 = cluster_bootstrap(base_hit1.loc[idx].to_numpy(), s1_hit1.loc[idx].to_numpy(), clusters)
    new_vs_old = cluster_bootstrap(hit1.to_numpy(), base_hit1.loc[idx].to_numpy(), clusters)
    p_holm = holm(np.array([old_vs_s1[3], new_vs_s1[3]]))
    table = pd.DataFrame([
        dict(method="stage 1 (finetuned_random)", dev_cv_R1=hits(dev, stage1_score(dev))[0].mean(), test_R1=s1_hit1.mean(), test_R5=s1_hit5.mean()),
        dict(method="re-ranker, dev only (Finding 43)", test_R1=base_hit1.mean(), diff_vs_stage1=old_vs_s1[0],
             lo=old_vs_s1[1], hi=old_vs_s1[2], p=old_vs_s1[3], p_holm=p_holm[0]),
        dict(method="re-ranker, dev + studio", dev_cv_R1=dev_cv, test_R1=hit1.mean(), test_R5=hit5.mean(),
             diff_vs_stage1=new_vs_s1[0], lo=new_vs_s1[1], hi=new_vs_s1[2], p=new_vs_s1[3], p_holm=p_holm[1]),
        dict(method="dev + studio minus dev only", diff_vs_stage1=new_vs_old[0], lo=new_vs_old[1], hi=new_vs_old[2], p=new_vs_old[3])])
    table.to_csv("results/rerank_studio.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print("saved results/rerank_studio_pairs.csv, rerank_studio_leakage.csv, rerank_studio.csv")


if __name__ == "__main__":
    main()
