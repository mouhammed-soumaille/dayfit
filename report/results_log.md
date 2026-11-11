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

## Update Nov 11: proposal alignment

Approved proposal: user = online shoppers; main metric = Precision@5 (same type); constraint = robustness to
imperfect user photos + response time. We keep that framing; exact-item Recall is our added, harder question.

## Finding 4: fine-tuning (row 7)

Training: 5 epochs, last 4/12 blocks, same-type hard batches, lr 1e-5, temperature 0.05, batch 64.
Validation Recall@1: 0.807 (pretrained) -> 0.957 -> 0.973 -> 0.977 (best, epoch 3) -> 0.977 -> 0.977.

| | mild | medium | strong |
|---|---|---|---|
| Recall@1 FashionSigLIP | 94.9 | 79.3 | 51.0 |
| Recall@1 fine-tuned | 99.0 | 96.9 [95, 99] | 84.7 [80, 89] |
| paired difference | +4.1* | +17.7* | +33.7* (all p<0.0005) |
| P@5 FashionSigLIP | 0.897 | 0.886 | 0.773 |
| P@5 fine-tuned | 0.795 | 0.798 | 0.744 |
| paired difference | -10.3* | -8.8* | -2.9* (p=0.047, borderline) |

Fine-tuning wins exact-item Recall but loses the proposal's main metric. Hypothesis: same-type hard batches
push each item away from its same-type neighbours. Test: --hard_batches 0 ablation; also try ensemble_ft_mv.

## Finding 5: real photos (LookBench real_streetlook, 981 queries, 3,278 gallery, query = LookBench crop)

| | Recall@1 | Recall@5 | category P@5 |
|---|---|---|---|
| FashionSigLIP | 42.8 [39.8, 45.8] | 64.1 | 0.714 |
| fine-tuned | 45.5 [42.6, 48.4] | 67.3 | 0.691 |
| paired | +2.7 [+0.5, +4.8] p=0.014 | | -2.3 p<0.0005 |

- Synthetic fine-tuning gain (+17.7) mostly does not transfer to real photos (+2.7): it largely learned our augmentation.
- Synthetic photos are easier than real ones (79% vs 43%).
- Paper (arXiv 2601.14706) reports FashionSigLIP fine R@1 55.2%, coarse R@1 73.4% on a 61,553-image corpus with a
  different correctness rule (category + all attributes). Not comparable to our item_ID metric; our category P@5
  (71.4%) is roughly consistent with their coarse 73.4%.


## Finding 6: does adding the pretrained model fix the fine-tuning trade-off? (A1)

| | synth R@1 med | synth P@5 med | real R@1 | real cat P@5 | ms |
|---|---|---|---|---|---|
| ensemble_mv (pretrained) | 85.4 | 0.882 | 44.0 [40.9, 47.1] | 0.720 | ~28 |
| finetuned | 96.9 | 0.798 | 45.5 | 0.691 | ~14 |
| finetuned_mv | 96.9 | 0.808 | 45.6 [42.6, 48.7] | 0.700 | ~14 |
| ensemble_ft_mv | 97.6 | 0.850 | 47.1 [44.1, 50.2] | 0.706 | ~28 |

Paired tests, ensemble_ft_mv vs finetuned: R@1 not different anywhere (real +1.6, p=0.146); P@5 better everywhere
(synthetic +3.8 to +6.5, real +1.5, p=0.001). So ensemble_ft_mv dominates finetuned.
ensemble_ft_mv vs ensemble_mv: R@1 better (synthetic +3 to +25, real +3.1, p=0.002); P@5 worse (synthetic -3.1/-3.5,
strong tie; real -1.4, p=0.003). Trade-off persists but is small on real photos.
Decision rule: find-similar (P@5) -> ensemble_mv (no training); find-exact-item -> ensemble_ft_mv; both ~28 ms.
All significant real-photo results p <= 0.003, so they survive a Holm correction.

## Finding 7: the cause of the P@5 drop is same-type hard batches (ablation)

Random-batch fine-tuning (same seed, split, 5 epochs; only the batching changed), val R@1 best 0.963 vs 0.977.
vs hard-batch finetuned: real R@1 +0.1 (p=0.95), real cat P@5 +3.5 (p<0.0005); synthetic P@5 +7.0/+6.9/+7.8,
synthetic R@1 -1.4/-2.0/-3.4 (all significant). Hard batches only buy synthetic Recall.
vs FashionSigLIP (same architecture, 14 ms): real R@1 +2.8 (p=0.01), real cat P@5 +1.2 (p=0.003): better on both.
vs ensemble_mv (28 ms): real R@1 +1.5 (p=0.16), real P@5 +0.6 (p=0.26): no significant difference at half latency.
Synthetic P@5 vs pretrained: -2 to -3 on mild/medium (medium p=0.042, borderline), +5 on strong.
New decision rule: finetuned_random (14 ms) for both find-similar and find-exact; ensemble_mv as training-free option.

## Finding 8: mechanism (T2) - hard batches push same-type items apart

Catalog embeddings (6,197 items), cosine similarity. gap_std = (within-type - between-type mean) / std of all pairs.
nn_same_type = share of each item's 5 nearest catalog neighbours with the same type.
| model | within | between | gap_std | nn_same_type |
| FashionSigLIP | 0.690 | 0.531 | 1.68 | 0.905 |
| hard batches | 0.250 | 0.127 | 1.20 | 0.836 |
| random batches | 0.373 | 0.179 | 1.78 | 0.893 |
Hard batches cut type separation by 29%; random batches keep it. Same ordering as synthetic P@5.
Fine-tuning lowers all similarities (embeddings spread out), so raw gaps are not comparable; use gap_std.
Caveat: catalog includes training items; descriptive, no CIs.

## Finding 9: conformal on real models (B1, synthetic photos, alpha = 0.1, 200 splits)

In-distribution coverage >= 0.90 for all models and severities (guarantee holds).
Relative variant, mean set size (all severities / strong): SigLIP 8.6 / 102, ensemble_mv 3.9 / 54,
finetuned_random 1.0 / 1.8 (72% of strong queries get a single answer, 95% correct). Synthetic favours fine-tuning.
Threshold variant: exact coverage (0.901-0.907, inside the T6 bound 0.907) but huge sets on strong photos (912 items).
Relative variant over-covers (0.95-0.98): most calibration scores are tied at 0 (top-1 correct), so the
no-ties assumption of the upper bound fails; coverage = R@1. Fix: randomized tie-breaking.
Shift mild -> strong: coverage collapses (relative 0.51 / 0.58 / 0.81; threshold 0.17 / 0.10 / 0.61);
threshold sets often empty. Guarantee needs calibration data like deployment data -> T1.