# DAYFIT results log

Running record of results and what we concluded, for writing the report.
Numbers are copied from the terminal outputs; the raw files are
`results/summary.csv` (with 95% CIs) and `results/comparisons.csv` (paired tests).

## Setup (as of Nov 10)

- Catalog: 6,197 Kaggle Fashion Product Images, 14 article types, near-duplicates removed (dHash + coarse mean colour).
- Images: high-res versions of the same items (max side 512 px), swapped in from the full Kaggle dataset.
- Queries: 294 items (21 per type), each as clean + synthetic phone photo at 3 severities (mild / medium / strong).
  All severities of one item share the same random draws, only scaled, so the robustness curve is a fair comparison.
- Metrics: exact-item Recall@1 / Recall@5 (phone queries), Precision@5 same article type (source item removed).
  95% CIs by bootstrap (2,000 resamples of the queries); method comparisons by paired bootstrap on the same queries.
- Clean-query Recall is always 1.0 by construction (the query is the catalog image), so it is not reported.

## Finding 1: resolution matters a lot

Same models, same queries, 60x80 px images vs high-res (phone_medium, Recall@1):

| method | low-res (60x80) | high-res |
|---|---|---|
| CLIP | 2.4% | 24.8% |
| FashionSigLIP | 6.1% | 79.3% |
| FashionCLIP | 15.3% | 79.3% |

(Low-res runs used a single "phone" severity equivalent to medium; results in `results/lowres/`.)

## Finding 2: the results ladder (phone queries, Recall@1 with 95% CI)

| row | method | mild | medium | strong | latency |
|---|---|---|---|---|---|
| 0 | colour histogram | 2.4% | 1.0% [0.0, 2.0] | 0.3% | 4 ms |
| 1 | CLIP ViT-B/32 | 53.1% | 24.8% [20, 30] | 10.2% [7, 14] | 11 ms |
| 2 | FashionSigLIP | 94.9% | 79.3% [74, 84] | 51.0% [45, 56] | 19 ms |
| 2b | FashionCLIP | 91.8% | 79.3% [74, 84] | 49.7% [44, 55] | 18 ms |
| 3 | + garment cropping (query only) | 93.2% | 73.8% [69, 79] | 44.2% [38, 50] | 252 ms |
| 3b | + cropping query and catalog | 93.9% | 74.5% [69, 80] | 39.1% [34, 45] | 250 ms |
| 4 | FashionSigLIP + multi-view catalog | 96.9% | 82.3% [78, 87] | 54.4% [49, 60] | 19 ms |
| 5 | ensemble (SigLIP + FashionCLIP) | 95.6% | 84.0% [80, 88] | 55.8% [50, 61] | 37 ms |
| 5b | ensemble, both multi-view | 96.3% | **85.4%** [81, 89] | **58.2%** [52, 64] | 37 ms |
| 7 | fine-tuned | | | | (pending) |

Precision@5 (same article type), phone_medium: histogram 0.08, CLIP 0.58, FashionSigLIP 0.89, ensemble_mv 0.88 (chance 0.07).

## Finding 3: which differences are real (paired bootstrap, Recall@1)

| change | medium | strong | significant? |
|---|---|---|---|
| colour histogram -> CLIP | +23.8 | +9.9 | yes (all severities) |
| CLIP -> FashionCLIP | +54.4 | +39.5 | yes (all severities) |
| FashionCLIP -> FashionSigLIP | +0.0 (p=1.0) | +1.4 (p=0.61) | no: the two fashion models are tied |
| FashionSigLIP -> + crop (query) | -5.4 | -6.8 | yes, worse (mild: no difference) |
| FashionSigLIP -> + crop (both) | -4.8 | -11.9 | yes, worse (mild: no difference) |
| FashionSigLIP -> + multi-view | +3.1 | +3.4 | yes (all severities, p<0.03) |
| FashionSigLIP -> ensemble | +4.8 | +4.8 | yes (mild: no difference) |
| ensemble -> ensemble multi-view | +1.4 (p=0.11) | +2.4 (p=0.035) | strong only |

## Interpretation (draft wording)

- Fashion-specific pretraining is by far the biggest single gain (CLIP -> fashion model: +40 to +54 points).
- Which fashion model is used does not matter on our data (SigLIP and FashionCLIP tie); we use SigLIP as the
  base for fine-tuning because it is marginally ahead on strong photos and Precision@5.
- Combining the two models helps (+5 points), consistent with them making different mistakes, at 2x search cost.
- Multi-view catalog embeddings give a small but consistent gain for free at search time.
- Garment cropping (rembg) hurts on synthetic phone photos, and cropping the catalog too does not fix it, so the
  cause is not a framing mismatch between query and catalog. Likely the segmentation itself fails on our synthetic
  backgrounds (TODO: check data/crop_examples/). Synthetic photos have no person or room, which is what cropping
  is meant to remove, so this test cannot show cropping's intended benefit; real photos are needed.
- Mild photos are nearly solved (95-97%); medium and strong are where methods separate.
- Cropping costs ~230 ms per query, about 12x the embedding time.

## Caveats to state in the report

- Synthetic phone photos only so far; real-photo evaluation (own photos, LookBench) pending.
- 294 queries, 6,197 catalog images: small but every result has a CI.
- Fine-tuning will train on the same kind of synthetic augmentation used for testing, so part of any gain may be
  learning the augmentation; real photos are the check.

## Pending

- [ ] Fine-tuning (training/finetune.py) -> row 7, then ensemble_ft_mv
- [ ] ~30 real photos of own clothes (condition "real", Precision@5 only)
- [ ] LookBench real_streetlook (download failed on slow connection; retry on faster network with HF_HUB_DISABLE_XET=1)
- [ ] Look at crop failures (python -m retrieval.preprocess)
- [ ] Per-category breakdown, failure gallery, manual check of ~30 queries
- [ ] FastAPI + Streamlit app, Docker
