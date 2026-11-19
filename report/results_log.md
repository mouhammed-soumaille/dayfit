# DAYFIT results log

## How to read this log (one-page index, Nov 19)

This is my lab notebook, in the order things were done. Later findings can correct earlier ones (corrections:
Findings 15, 19, 21); the README is the current summary. Every experiment from Nov 19 on has a pre-registration
paragraph written BEFORE it was run, then a Finding with the result, including negative results.

**Codes used below**
- **Finding N**: a result. Findings 1-30 are the first phase (synthetic photos, first real-photo runs, the app);
  31-44 are the Nov 19 phase (frozen dev/test split, official corpus, baselines, re-ranker).
- **A1**: two-model ensembles. **B1**: conformal shortlist on synthetic photos. **C1 / C2**: the old menu-filter
  baselines (random shop order / menu + our ranking).
- **D8**: data-quality checks on LookBench (duplicates, near-duplicates, manual check). **D9**: own-wardrobe test,
  rules fixed in advance (not run). **D10**: per-category breakdown and failure gallery.
- **E1-E8**: the app (engine, calibration, badges, API, page, Docker, latency, README).
- **T1-T6**: analysis questions: conformal under the synthetic-to-real shift (T1), why hard batches hurt (T2),
  per-category and grouped conformal (T3), risk control / "not sure" (T4), hubness (T5), the coverage bound (T6).
- **Tasks 1-6** (Nov 19 plan): 1 demo fixes (F35-36), 2 the app's real setting (F37-39), 3 bad photos (F40),
  4 credible baselines (F41-42), 5 learned re-ranker (F43-44), 6 production and documentation.

**Protocols**
- *Main protocol (from Finding 38 on)*: LookBench real_streetlook, 981 real street photos, searched against the
  OFFICIAL corpus of 61,553 images (3,278 gallery + 58,275 noise), identical image files merged ("dedup").
  Exact-item metrics: R@1, R@5.
- *Gallery protocol*: the same photos against the 3,278 gallery only. Needed for group P@5 (noise images have no
  category) and for anything that needs item text or labels (keyword search, menus). Earlier findings use it.
- *Dev / test split*: 500 / 481 photos, 621 clusters (same crop, street photo or item stay together), seed 490.
  Methods are tuned on dev and run ONCE on test; paired cluster bootstrap; Holm across methods taken to test.
- *Synthetic photos*: catalogue images degraded with the training augmentation; they mostly measure that
  augmentation (Finding 5), so they are a secondary result.

**Current headline numbers and their source (test half, 481 real photos, LookBench crop)**
| claim | number | source |
|---|---|---|
| exact item first, recommended model, official corpus | R@1 0.499, R@5 0.688 | Finding 38, results/perquery/finetuned_random_*_image_full.csv |
| pretrained FashionSigLIP, same setting | R@1 0.476 | Finding 38 |
| classical image search (colour + texture + shape) | R@1 0.123 | Finding 42, results/classical_baseline.csv |
| keyword search, short / idealised detailed (gallery) | R@1 0.312 / 0.486 | Finding 41, results/keyword_baseline.csv |
| whole photo instead of crop (official corpus) | R@1 0.391 | Finding 37/38, results/perquery/*_raw_image_full.csv |
| fine-tuning gain on blurred photos (all 981) | +6 to +21 points R@1 | Finding 40, results/robustness*.csv |
| learned re-ranker | R@1 0.520 (+2.1, n.s.) | Finding 43, results/rerank_test.csv |
| right group in top 5 with the group bonus (gallery) | 0.905 (sanity floor) | Finding 32, results/fusion_test.csv |

Raw numbers of the first phase are in `results/summary.csv` (with 95% CIs) and `results/comparisons.csv`.

---

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

## Finding 10: conformal under real-photo shift (T1), LookBench, alpha = 0.1, 200 paired splits

Synthetic -> real coverage (threshold / relative): SigLIP 0.84 / 0.67, ensemble_mv 0.84 / 0.64,
finetuned_random 0.73 / 0.52 (relative mean set 1.2: confident single answers, right half the time).
Real -> real restores coverage 0.899-0.903; relative median set size SigLIP 164, ensemble_mv 143,
finetuned_random 59 (of 3,278); means 310 / 288 / 151 (skewed). Threshold sets 2-4x larger.
T1 bound coverage >= 0.9 - TV(synthetic scores, real scores): holds in 14/18 (model x variant x bins).
Misses are small and at coarse bins: binning underestimates TV (data processing), so the plug-in bound is
optimistic. TV largest for the fine-tuned model (0.31-0.37 vs 0.23-0.29): consistent with "learned our augmentation".
Implication: calibrate on real photos; on real photos a 90% guarantee needs large sets -> app should flag "unsure".

## Finding 11: per-category conformal (T3) and risk control (T4), LookBench real -> real

T3 (alpha 0.1, relative): standard conformal hides categories at 0.52-0.82 (jumpsuit, denim skirt, cape,
pleated skirt, cardigan, coat) while easy ones are at 1.00. Mondrian: every category >= 0.90.
Cost = redistribution: easy categories get smaller sets (finetuned_random: shirt 65 -> 10, bag 63 -> 10),
hard ones bigger (coat 84 -> 588). Categories with < ~18 queries (< 9 calibration) get the whole gallery
(3,278): need ceil((n+1)(1-alpha)) <= n, i.e. n >= 9. Fix: merge rare categories into coarser groups.
T4 (answer only if top similarity >= lambda; loss = wrong-category share of shown top 5, 0 if abstain):
measured loss <= alpha at every level (0.198-0.200 / 0.096-0.097 / 0.049). Answer rate at alpha 0.2:
SigLIP 80%, ensemble 81%, finetuned_random 85%; at 0.1: 41 / 46 / 46%. P@5 when answered only 0.73 -> ~0.79:
the guarantee is met mostly by abstaining; top similarity is a weak confidence signal.

## Finding 12: grouped Mondrian conformal (T3 follow-up, evaluation/conformal_extra.py, "group" variant)
Setup: 7 category groups fixed BEFORE seeing any grouped result (tops 296, outerwear 216, bottoms 229,
dresses 40, footwear 55, bags 49, accessories 96 queries; sock -> accessories, cardigan -> tops).
Calibrate the relative threshold within each group; real -> real, alpha = 0.1, 200 random 50/50 splits.
Standard (one shared threshold) under-covers the same three groups for all three models:
outerwear 0.86-0.88, bottoms 0.85-0.87, dresses 0.83-0.86. Grouped calibration brings every group to
0.90-0.93 for every model (bottoms 0.899-0.902) and reduces over-coverage of easy groups (bags 1.00 -> 0.92).
Set sizes (finetuned_random): small categories no longer get the whole gallery (per-category Mondrian 3278 ->
grouped: belt 177, tie 3, glasses 2, clutch bag 4, sneaker 3).
Cost: the guarantee is per group, not per category, so hard categories inside an easier group are hidden
again (finetuned_random: sock 0.80, cardigan 0.77, jumpsuit 0.74, denim skirt 0.70); some large
categories in a hard group get larger sets than standard (coat 84 -> 221).
Standard and per-category numbers and T4 are unchanged (same seed and splits; checked: max difference 0.0).
Implication: grouping is the practical middle ground: a guarantee for every group with usable sets.
Per-category guarantees need >= 9 calibration queries per category, which the smallest categories lack.

## Finding 13: menu-filter baseline, "how shoppers search today" (C1, evaluation/filter_baseline.py)
Shopper filters by garment type (and colour on synthetic data; LookBench has no colour labels); the shop
shows matches in an order unrelated to the photo (modelled as uniformly random). Exact formulas, no
simulation: with r pictures of the item among m shown, P(hit in top k) = 1 - C(m-r,k)/C(m,k), expected
position of the first picture = (m+1)/(r+1). Precision@5 (same type) = 1 by construction, so we compare on
exact-item recall and scroll depth.
Synthetic (294 test items, 6,197 catalogue): type filter shows median 490 items, R@1 0.002, R@5 0.012,
median scroll 246; type + colour shows median 56, R@1 0.045, R@5 0.183, median scroll 28.5.
LookBench real (981 queries, 3,278 gallery): type filter shows median 154, R@1 0.026, R@5 0.113, median
scroll 48; 49 queries (5%) are filtered out because their gallery pictures carry a different category label
(counted as misses; also caps any model's category P@5 on LookBench).
Comparison: finetuned_random R@1 0.949 (synthetic medium) and 0.456 (real) vs 0.045 / 0.026 for the best
menu filter: about 20x / 17x. Optimistic for menus: assumes the shopper can name type and colour,
which the user story says they cannot.

## Finding 14: menu filter + visual re-ranking (C2, evaluation/filter_rerank.py, finetuned_random)
Filter by type (and colour, synthetic only), then sort the survivors by model similarity.
Filter "none" reproduces run_eval (synthetic medium R@1 0.949) and the dedup LookBench protocol (0.520).
Synthetic R@1 none / type / type+colour: medium 0.949 / 0.949 / 0.963, strong 0.813 / 0.823 / 0.874,
clean 1.0 everywhere: the model almost never confuses garment types, so a type filter adds nothing;
colour helps on the hardest photos.
LookBench real R@1 none 0.520 -> type 0.554, R@5 0.677 -> 0.704, mean rank of the item 85.7 -> 16.9
(filtering removes the catastrophic misses), but 49 items (5%) are lost to inconsistent category labels.
Theory: if the item survives the filter, its rank can only improve (filtering only removes competitors).
Implication: an optional type filter is a useful app feature for real photos; the model, not the menu,
does the work of finding the exact item.

## Finding 15: LookBench gallery duplicates; dedup protocol (correction, evaluation/run_lookbench.py)
real_streetlook gallery: 3,278 entries but only 2,537 distinct image files; 633 files (outfit photos)
are listed 2-5 times under different item_IDs (94 of them under different categories); 515 of 981
queries have an item with a shared picture. The official protocol keeps whichever copy the sort puts
first, so retrieving the right picture can count as a miss. New "dedup" protocol: merge copies (md5)
before the top 5; a picture is correct if any copy has the query's item_ID / category. Both are saved;
official numbers reproduce exactly.
Dedup raises R@1 by +5.8 to +7.0 points for every model (finetuned_random 0.456 -> 0.520,
fashion_siglip 0.428 -> 0.486, ensemble_mv 0.440 -> 0.505, ensemble_ft_mv 0.471 -> 0.539), R@5 by ~2,
category P@5 essentially unchanged. Ranking unchanged; all Finding 7 conclusions hold (paired, dedup):
finetuned -> finetuned_random P@5 +0.036 (p<0.0005), R@1 -0.005 (p=0.31); fashion_siglip -> finetuned_random
R@1 +0.034 (p<0.0005), P@5 +0.011 (p=0.009); ensemble_mv -> finetuned_random n.s. (R@1 p=0.09, P@5 p=0.40).
C1 LookBench re-counted in distinct pictures (supersedes the Finding 13 LookBench line): type filter
shows median 119 pictures (was 154 entries), R@1 0.031 (was 0.026), R@5 0.131 (was 0.113), median scroll
36.7 (was 48); 49 queries still filtered out. Conclusion unchanged (model R@1 0.520 dedup, ~17x).
C2 LookBench: R@1 counts ties in the item's favour, so it equals the dedup protocol; R@5 and ranks still
count duplicate entries of OTHER pictures (slightly pessimistic).
Not re-run: conformal set sizes and risk control count gallery entries, not pictures (set sizes inflated
at most 3278/2537 = 1.29x; coverage unaffected) -> stated as a limitation.


## Finding 16: per-category breakdown (D10a, evaluation/per_category.py, finetuned_random)
Sanity check: query-weighted per-category means reproduce every saved overall number (4 synthetic
conditions + LookBench dedup R@1 0.520 / P@5 0.725). Saved in results/per_category.csv (97 rows).
LookBench real (dedup), 7 groups fixed for F12, R@1 / category P@5 (95% CI):
dresses 0.65 / 0.32 [0.24, 0.41] (n=40); bags 0.63 / 0.52 (n=49); accessories 0.52 / 0.66 (n=96);
tops 0.57 / 0.72 (n=296); footwear 0.62 / 0.73 (n=55); outerwear 0.38 [0.32, 0.44] / 0.80 (n=216);
bottoms 0.52 / 0.80 (n=229).
Low dress/bag P@5 is a base-rate effect, not a failure to find the item: dresses are 47 of 2,537 distinct
gallery pictures (1.9%; chance P@5 ~0.02, model ~17x) and bags 169; the exact item is often found
(R@1 0.65), but its neighbours are mostly other categories (tops/skirts for dresses); P@5 also needs the
exact category (dress vs jumpsuit counts as wrong). Outerwear is the opposite: type almost always right,
exact item hardest (601 near-identical coat/jacket pictures).
Synthetic strong photos (n=21 per type): lowest P@5 Jackets 0.69, Sports Shoes 0.72, Tops 0.74;
lowest R@1 Trousers 0.48 (P@5 still 0.86: right type, wrong pair), Track Pants 0.62.
Caveats: descriptive only (no correction across 7 groups / 14 types; n=21 gives CIs of about +-0.15).
Category P@5 depends on how common the category is in the gallery -> state as a metric limitation.
Implication: the "unreliable match" warning matters most for rare categories (dresses, bags) and for
exact-item claims on outerwear; feeds D10b (failure gallery) and F (per-category fairness).

## Finding 17: failure gallery (D10b, evaluation/failure_gallery.py, fig10_failures_*.png)
run_lookbench now also saves each query's top-5 distinct pictures (results/rankings/); re-run of
finetuned_random reproduced every saved number exactly, and hit@1 / hit@5 / P@5 recomputed from the
rankings match the per-query files for all 981 queries.
Correction found by the gallery's sanity check: our synthetic P@5 removes the exact item (top-6 minus
source; "are the alternatives sensible?") while LookBench category P@5 (LookBench's code) counts it.
Model-vs-model comparisons within a dataset are unaffected; synthetic and real P@5 must not be compared
directly (now stated in the report's metrics and LookBench sections).
Gallery = 6 random rows (seed 0) per case, qualitative only:
- Real dresses with P@5 <= 0.4 (31 queries): in 3/6 rows the "dress" crop shows a top and/or shorts
  (label/crop mismatch); the model returns visually matching tops/shorts with high scores (up to 0.86).
  In the other 3 the exact item is rank 1 and neighbours share colour/fabric (black skirts, satin).
- Real outerwear, top-1 not exact (134): all results are near-identical coats/jackets (fine-grained),
  scores 0.74-0.89; one row shows apparent near-duplicate gallery pictures not merged by md5; a cape row
  marks wrap coats as "wrong" (exact-category P@5).
- Synthetic Trousers, strong photos (11): all results right type, exact pair often rank 2-3; some black
  trousers are labelled Track Pants/Jeans in the catalogue (label ambiguity).
Implications: (1) misses are mostly fine-grained or label ambiguity, supporting a top-5 shortlist over a
single answer; (2) confident misses (high top similarity) explain why risk control on top similarity only
lifts answered P@5 0.73 -> 0.79 (F11); (3) D8 should quantify dress-label noise and near-duplicates.

## Finding 18: near-duplicate gallery photos (D8a/D8c, evaluation/near_duplicates.py, near_dedup.py)
Beyond exact duplicates (F15), the gallery has the same photo saved as different files (re-encoded,
resized, cropped). Candidates: pairs of distinct pictures with PRETRAINED FashionSigLIP cosine >= 0.90
(2,305 pairs; the evaluated model is not used to define duplicates in its own test set). Contact sheet,
4 random pairs per bin: same photo in 12/12 pairs with cosine >= 0.97, mixed in [0.95, 0.97), mostly
different photos below. dHash misses crops (a cropped copy had Hamming 17), so the rule uses cosine only.
Rule fixed BEFORE recomputing metrics: cosine >= 0.97 (0.95, 0.98 as sensitivity). Near-duplicates joined
into connected components (union-find): 2,537 pictures -> 2,406 groups (2,441 at 0.98; 2,303 at 0.95).
The same photo sometimes carries different categories (a leopard heel listed as "pants, shoes" and "sock").
Near-dedup protocol = dedup with "same file" replaced by "same photo"; computed from saved top-20 rankings
(every query had >= 5 groups); with no merging it reproduces all saved dedup per-query results.
R@1 dedup -> near 0.97: FashionSigLIP 0.486 -> 0.495, ensemble_mv 0.505 -> 0.516, finetuned 0.525 -> 0.535,
finetuned_random 0.520 -> 0.532 (about +1 pt for every model; 0.95: +1.6 to +2.1). Category P@5 within 0.004.
Paired tests (dedup / 0.98 / 0.97 / 0.95): fashion_siglip -> finetuned_random R@1 +0.034/+0.036/+0.037/+0.040
(all p<0.0005); P@5 +0.011 (p=0.009) / +0.011 (0.004) / +0.012 (0.002) / +0.008 (p=0.083, n.s.);
finetuned -> finetuned_random P@5 +0.036 to +0.038 (p<0.0005), R@1 n.s.; ensemble_mv -> finetuned_random
n.s. everywhere (smallest p 0.058).
Conclusion: dedup stays the main protocol; near-dedup is a sensitivity check in the report. All conclusions
hold, but "better P@5 than the pretrained base" is small (+0.01) and not significant at the loosest
threshold; state it as such. Many tests, no correction: p ~ 0.01 results are suggestive, not strong.

## Finding 19: LookBench category names are too strict; group P@5 (evaluation/group_p5.py) (correction)
Manual check (D8b) showed "wrong category" results that are the same kind of garment. Cause: category P@5
compares category STRINGS and LookBench's names overlap (bag/handbag/clutch bag, skirt/pleated/denim/silk
skirt, shirt/t-shirt, shoe/sneaker); the gallery also has 38 names on 109 of 3,278 entries that never occur
as a query category (boots, jeans, sneaker variants, "trench coat", "tights" ...), which always count as wrong.
Group P@5 = result in the query's category group (7 groups fixed for F12 before any grouped result + explicit
mapping of the 38 gallery-only names, written before computing; "product label" never counts). With exact
names the script reproduces every saved category P@5.
Exact -> group P@5: FashionSigLIP 0.714 -> 0.869, ensemble_mv 0.721 -> 0.864, finetuned 0.688 -> 0.843,
finetuned_random 0.725 -> 0.869 [0.853, 0.885].
Paired (group P@5): finetuned -> finetuned_random +0.026 [+0.020, +0.032] p<0.0005 (hard-batch drop survives);
fashion_siglip -> finetuned_random -0.001 [-0.008, +0.006] p=0.85 (the +0.011 exact-name gain VANISHES);
ensemble_mv -> finetuned_random +0.004 p=0.27 (tie holds).
By group (finetuned_random, exact -> group): bags 0.52 -> 0.85, footwear 0.73 -> 0.90, tops 0.72 -> 0.89,
outerwear 0.80 -> 0.93, bottoms 0.80 -> 0.95, accessories 0.66 -> 0.72, dresses 0.32 -> 0.32.
Corrections: (1) F7/F15 claim "finetuned_random beats its pretrained base on both real metrics" -> "better
real R@1 (+3.4, p<0.0005), tied on garment type"; the P@5 gain was in arbitrary name distinctions
(it was also fragile in F18). (2) F16: low bag P@5 was mostly naming (bag vs handbag), not base rate;
low dress P@5 is real (neighbours are tops/skirts, another group; plus label noise, F17).
Report: new section "Category names and group P@5", row in the ablation table, bullets and summary rewritten.
Limitation: groups are lenient (cardigan counts for blouse); exact and group P@5 bracket the truth.

## Finding 20: manual check of 61 LookBench queries (D8b, evaluation/manual_check.py)
Sample fixed before looking: all 31 dress/jumpsuit queries with P@5 <= 0.4 (finetuned_random, dedup) +
30 random other queries (seed 1). Contact sheet per query: full photo, LookBench crop, top 5 with borders.
Verdicts: label_check (crop shows the labelled garment: correct / wrong / ambiguous) and metric_fair (would a
shopper agree with the exact-category scoring). AI-annotated (Claude), then reviewed row by row by the author,
who agreed with all 61 (review, not independent annotation: anchoring possible). results/manual_check.csv.
Random sample (generalises, 95% Wilson): label wrong 1/30 = 3% [1, 17]; wrong or ambiguous 6/30 = 20% [10, 37];
metric unfair 9/30 = 30% [17, 48]. All 9 unfair rows are same-group results scored wrong by exact names
(coat/jacket, bag/handbag, shirt/t-shirt, sweater/cardigan) -> independent confirmation of Finding 19.
Dress failures (selected, does not generalise): label wrong 7/31 = 23% [11, 40]; wrong or ambiguous 10/31;
metric unfair 15/31 = 48% [32, 65], mostly label errors (crop shows a top or shorts) or dresses listed under
other categories; group P@5 cannot fix these (consistent with dresses 0.32 -> 0.32 in F19).
New data issue found: identical query crops. Only 700 of 981 query crops are distinct; 503 queries fall into
222 groups of identical crops, each copy asking for a DIFFERENT item (query-side twin of F15). Queries are
therefore not independent; bootstrap CIs and p-values that resample queries are likely too narrow -> next:
cluster bootstrap (resample crops).

## Finding 21: cluster bootstrap over identical query crops (evaluation/cluster_bootstrap.py) (correction)
981 LookBench queries use only 700 distinct crops (F20), so queries are not independent. New paired test
resamples whole crops (cluster bootstrap, ratio estimator sum(S_c)/sum(n_c)); with one query per cluster it
reproduces every earlier paired test exactly. CIs widen x1.06-1.36 (design effect ~1.1-1.9).
finetuned_random minus ... (naive -> cluster):
FashionSigLIP R@1 +0.034, p<0.0005 -> [+0.015, +0.052] p=0.001 (survives Holm over 4 tests: 0.001 < 0.0125);
FashionSigLIP exact P@5 +0.011, p=0.009 -> [+0.000, +0.021] p=0.047 (fails Holm: 0.047 > 0.0167);
FashionSigLIP group P@5 -0.001, n.s. both;
finetuned (hard) exact P@5 +0.036 and group P@5 +0.026: p<0.0005 both ways; R@1 n.s. both;
ensemble_mv: all n.s. (cluster p >= 0.13).
Correction to the report's ablation caption (it said both FashionSigLIP results survive Holm).
Robust real-photo claims: (1) better exact-item R@1 than the pretrained base, (2) random batches recover the
precision lost to hard batches, (3) tie with the pretrained ensemble at half the latency. Dropped: "better
garment-type precision than the base" (borderline exact, absent at group level).
Note: naive group-P@5 p here 0.898 vs 0.852 in group_p5 (different query order -> different resamples;
Monte Carlo noise, same estimate).

## D9 protocol (fixed BEFORE taking the photos, Nov 18)
Final untouched test: ~30 phone photos of the author's own clothes, used ONCE.
Model choice frozen: finetuned_random (recommended); also reported, not used for choosing: color_hist,
fashion_siglip, ensemble_mv. Metric: Precision@5 = same catalogue article type (own clothes are not in the
catalogue, so no Recall). Primary result: finetuned_random P@5 with 95% bootstrap CI. Comparisons are
descriptive (n~30 has little power). No re-shooting, no dropping photos after seeing results;
any photo that fails to load is reported.
Amendment (Nov 19, 00:55, BEFORE any photo was taken; reason: plan changed from ~30 photos to the whole
wardrobe, and the group bonus of Finding 32 became the headline method):
- Photos: every garment in the wardrobe, one photo each, taken in one session, phone camera, no editing.
  All photos are used; none dropped or re-shot after seeing results; a photo that fails to load is reported.
- Each photo is labelled BEFORE running anything: garment group (tops, outerwear, bottoms, dresses,
  footwear, bags, accessories) and, if it exists, the evaluation-catalogue article type (one of the 14).
- Primary results (frozen methods, run once):
  (1) group P@5 of finetuned_random + group bonus w = 0.05 against the APP catalogue (the app as shipped),
      with 95% bootstrap CI; this is the D9 check of the 0.905 headline;
  (2) article-type P@5 of finetuned_random against the EVALUATION catalogue, only for photos whose type is
      one of the 14 (the original protocol).
- Also reported, descriptive: color_hist, fashion_siglip, ensemble_mv on (2); the app's "not sure" rate,
  group-badge accuracy, out-of-catalogue warnings and shortlist sizes.
Clarification (Nov 19, 13:35, BEFORE any photo was taken; script evaluation/d9_own_photos.py written, not run):
- Crop box: not used. The crop box is optional in the app and its default sends the photo unchanged, so
  "the app as shipped" = the whole photo. (Harder than LookBench, whose queries are hand-drawn crops.)
  Photos are taken with the garment filling most of the frame (hanger or plain surface, or worn).
- Group "other": a garment that fits none of the 7 groups (e.g. a watch) is labelled "other"; it is listed
  and counted for out-of-catalogue warnings but is in neither primary result.
- Photos in data/real_photos/ (git-ignored); labels in data/real_photos/labels.csv (file, group, eval_type),
  written by hand and checked with --check (no model) BEFORE --run. --run refuses to run twice.

## Finding 22: hubness on LookBench (T5, evaluation/hubness.py)
k-occurrence N_k(x) = number of distinct query crops (700; identical crops counted once, F20) whose top-k
contains gallery picture x (2,537 distinct pictures). Check: sum of N_k = crops x k for every model and k.
Skewness of N_5 (random-list baseline Bin(700, 5/2537): 0.85): FashionSigLIP 2.20, ensemble_mv 2.31,
finetuned 2.34, finetuned_random 2.32 (N_10: 2.00-2.29). Never in any top 5: 35-39% (random: 25%).
Top 1% of pictures (26) fill 7.3-7.8% of all top-5 slots; max N_5 14-16.
-> hubness present but moderate, and essentially unchanged by fine-tuning.
Hubs in failures (share of hub slots that are wrong-group results vs other slots): finetuned_random 14.4% vs
14.8% (no difference); FashionSigLIP 11.3% vs 14.8%; ensemble_mv 22.4% vs 14.6%; finetuned 21.5% vs 16.9%.
Descriptive only (hub slots are not independent).
Biggest hubs (fig11_hubs.png): plain generic garments (white t-shirts, cream cardigans, beige trench),
likely near the centre of their type; tops are 30% of queries, so part of their popularity is legitimate.
Conclusion: hub reduction (e.g. CSLS, mutual proximity) is not a priority for the recommended model.

## Finding 23: app calibration (E2, serving/calibrate.py -> serving/calibration.json)
Decision 1a: app uncertainty calibrated on LookBench real photos (all 981; finetuned_random). Checks: best
similarity equals saved rank-1 score for all photos; exact-name loss 0.2736 reproduces F11.
Shortlist (relative conformal, alpha 0.1): q = 0.228; 200 random 50/50 splits: coverage 0.901, median size 59
(reproduces F10) -> too large to display in full; the UI shows the count and the first items.
"Not sure" (risk control on top similarity; loss = share of top 5 in the wrong GROUP, F19):
alpha 0.2 is VACUOUS (loss already 0.130 without abstention -> answers 99.9%), so alpha set to 0.1 (chosen
after seeing the 0.2 result; a product decision, not an accuracy claim; D9 is the independent check).
LookBench (200 splits): none: answer 100%, loss 0.130, group P@5 0.870 | 0.2: 99.9%, 0.128, 0.872 |
0.1: 91.1%, 0.098, 0.893 | 0.05: 58.3%, 0.047, 0.920. lambda(0.1) = 0.671 (exact-name lambda 0.823).
Interpretation: the guarantee is met mostly by abstaining more, not by abstaining on the right photos
(precision when answering only 0.87 -> 0.89 -> 0.92): top similarity is a weak confidence signal
(consistent with F11, F17 confident outerwear misses, F22).
Limitation: calibrated on LookBench's gallery; our catalogue may give different similarity levels, so no
formal guarantee in the app; D9 measures the app's answer rate and wrong-type rate when answering.

## Finding 24: match badges (E3, serving/badges.py -> serving/badges.json)
Badges describe measured differences, never the model's reasons. Checks: CIELAB(white) = (100, 0, 0);
catalogue Black items darker than White (median L 21 vs 76).
Colour badge: median CIELAB of image centre (near-white background dropped); CIEDE2000 bands from 3,060
same-colour-label catalogue pairs: very close <= 16.5 (median), different > 37.0 (90th pct).
True match (query vs its own item) badged "different": 3.7% medium, 7.1% strong phone photos -> shipped.
Type badge: zero-shot PRETRAINED FashionSigLIP ("a photo of ...", 14 types; prompt embeddings saved once,
so the app needs no tokenizer). Fine-type accuracy clean 0.840, medium 0.830, strong 0.769 (chance 0.071);
"same type" badge agrees with truth only 0.78-0.85, worse than the ranking's own P@5 -> fine type DROPPED.
Group level (tops/outerwear/bottoms/dresses/footwear; group prob = sum of type probs):
accuracy 0.969 / 0.973 / 0.942; shown only if group prob >= 0.9: 0.978 (94% shown) / 0.992 (89%) /
0.983 (82%) -> shipped: group badge at confidence >= 0.9 (threshold chosen after seeing this table, on
synthetic photos: a UI decision, not an accuracy claim; D9 checks it on real photos).
## Finding 25: out-of-catalogue warning (serving/outside_check.py -> serving/outside.json)
The catalogue has 14 types and no skirts, bags, hats, jumpsuits, ...; for such photos the results are
necessarily poor (manual tests: floral skirt, white skirt -> "not sure", shortlists 252-688). New zero-shot
check (pretrained FashionSigLIP, 9 out-of-catalogue prompts, same image embedding as the type badge; group
badge provably unchanged): outside_prob = total probability of the outside prompts in a 23-way softmax.
Labels fixed before running: LookBench OUT = skirts, jumpsuit, bags, hat, scarf, belt, sock, glasses, glove,
tie, umbrella, headband (251 crops); IN = categories we sell (606 crops); coat/cape excluded; + 588 synthetic
in-catalogue photos. Threshold rule fixed before running (smallest tau with false alarms <= 5% on both IN
sets) -> tau = 0.4: detects 91.6% of out-of-catalogue crops (skirts 93.9%, bags 100%, other 85.6%);
false alarms 3.8% LookBench, 0.9% synthetic; detected skirts named "skirt" 97.8%.
App: shows "This looks like a skirt. Our catalogue has no skirts, so the results are only loose suggestions."
Coverage remains a limitation (proposal: catalogue coverage); a modern demo catalogue is planned.

## Finding 26: deployment (E6/E7: Dockerfile, docker-compose.yml, serving/latency.py)
CPU-only image (python 3.14-slim, torch 2.15 CPU), API + Streamlit page as two services; API container
read-only (cannot write files). Build note: copying the checkpoint into the image changed its modification
time, so the index cache rebuilt the index on CPU once (25 min); fixed by restoring the recorded time
(index built on the Mac reused; build then reports "loaded saved catalog embeddings").
Latency, 100 synthetic medium photos after 5 warm-up (median / p95): Mac GPU (MPS) 36 / 37 ms;
Docker CPU (4 vCPU) 266 / 282 ms server time (final image; previous image 266 / 278); round trip the same within ~1 ms (clocks of VM and host
differ, so upload/JSON overhead is negligible). Top 5 equal to the evaluated results for 105/105 photos.
About half of the time is the second model (type badge); making it optional would roughly halve latency.
Caution recorded: a first measurement (36 ms) accidentally hit a local MPS server on the same port;
checked with lsof and the container log, then re-measured.
## Finding 27: learned "not sure" score (evaluation/learned_confidence.py -> results/learned_confidence.csv)
Question: can a learned confidence score answer more photos than the top similarity at the same risk level?
Seven signals per LookBench photo (top similarity, gaps to 2nd and 5th result, shortlist size, type and
out-of-catalogue probabilities, colour difference to the top 5); target "the top 5 are all in the right group".
Plan fixed before running: 50 random splits by crop (40% train, 30% calibrate lambda at alpha 0.1, 30% test);
logistic regression and gradient-boosted trees (HistGradientBoostingClassifier, fixed settings, no tuning).
Means over 50 splits (top similarity / logistic / trees):
  AUROC 0.651 / 0.758 / 0.719; answer rate at alpha 0.1: 90.2% / 89.9% / 87.7%;
  test loss 0.096 / 0.097 / 0.097 (all <= 0.1, guarantee holds); answer rate above top similarity in
  24 / 50 splits (logistic) and 15 / 50 (trees).
Negative result: the learned scores rank photos better overall but not on the worst ~10% that must be
skipped at alpha 0.1. The app keeps the top similarity (simpler, as good where it matters).

## Finding 28: bigger app catalogue (data_prep/app_catalog.py, serving/engine.py, serving/app_badges_check.py)
App only; every evaluation number stays on the 6,197-item catalogue. Selection rules fixed before building:
full Fashion Product Images zip, adult items, 72 wearable types (clothes, shoes, bags, worn accessories;
no watches, wallets, jewellery, underwear, nightwear, swimwear, personal care), all evaluation items kept,
near-duplicates removed with the build_catalog rule. Result: 33,141 items of 68 types (501 near-duplicates
removed, 5 without image); e.g. heels 1,316, sandals 845, flip flops 885, handbags 1,756, sunglasses 1,042,
skirts 98. Embedded with finetuned_random (separate index, finetuned_random_app.npy).
Checks: evaluation catalogue unchanged (same top 5, score difference 0); in the app index the 6,197
evaluation items have exactly the saved embeddings (difference 0) and no best score went down.
Search time on MPS 18 ms (was 15 ms).
Badges with the app vocabulary (72 prompts, 7 groups: + bags, accessories; groups fixed before measuring);
the evaluation badge still reproduces Finding 24 (shown 0.891, correct 0.992 on phone_medium).
Group badge at confidence >= 0.9: LookBench real shown 64.5%, correct 95.4% (accessories 86.4%, bags 100%,
bottoms incl. skirts 96.1%, dresses 91.3%, footwear 100%, outerwear 93.5%, tops 97.0%);
synthetic shown 70.2%, correct 99.3%. More groups -> badge shown less often, accuracy about the same.
Out-of-catalogue warning (still not sold: swimsuit, watch, necklace, earrings, wallet, umbrella): same rule
as Finding 25 -> tau = 0.3; false alarms 0.1% LookBench (976 crops of sold kinds), 0.2% synthetic.
Detection only testable on 5 umbrella crops (5/5) -> unmeasured in practice.
Open: "not sure" lambda was calibrated against a smaller gallery; with 33k items best scores can only rise,
so it may fire less often. To be checked on real photos (D9).

## Finding 29: type-aware ranking for the app (serving/type_rerank.py -> serving/rerank.json)
Problem found with own photo (my_photos/jeanstrial.jpg, flat lay of long light-wash jeans): top 1 = kids'
denim shorts (0.660), adult jeans lower. Diagnosis (serving/diagnose_photo.py): NOT the centre crop (the
model sees the whole garment; padding to a square changes nothing; my first guess was wrong) and not
transparency (plain RGB JPG). Cause: photo style. Every adult pair of jeans in the catalogue is photographed
on a person; only kids' denim lies flat on white, so a flat photo matches flat kids' items. Synthetic tests
cannot show this (queries are made from catalogue images, same style).
Fix: score = similarity + w * P(item's type | photo), P from the zero-shot type predictor (app vocabulary,
ignores photo style). Shown similarity, "not sure" and shortlist stay on the plain similarity; evaluation
catalogue unchanged (w = 0 there). Plan fixed before running: synthetic medium/strong photos (588) against the
33,141-item app catalogue; w in {0, 0.02, 0.05, 0.1, 0.2}; keep the highest mean type P@5 with R@1 drop
<= 1 pt on both. Check: w = 0 reproduces the app's top 5.
                      R@1 medium / strong    type P@5 medium / strong
  w = 0 (before)      0.929 / 0.769          0.845 / 0.815
  w = 0.05 (chosen)   0.922 / 0.762          0.871 / 0.828
  w = 0.1             0.912 / 0.748          0.877 / 0.835   (R@1 drop 1.7 / 2.1 pts: not allowed)
  w = 0.2             0.891 / 0.731          0.861 / 0.805
Own photos (shown only, did not choose w): jeans flat lay now Jeans first (Jeans, Shorts, Jeans, Jeans, Jeans);
skirt photo: skirts now ranks 1-4 (Trousers moved from 4 to 5), dayfittrial4 (jeans) 2 -> 3 jeans, shoe photo formal shoe replaced by casual shoe (5/5 casual);
others unchanged. Still to do: measure on real photos (LookBench, group P@5), the headline target.

## Finding 30: app clean-up (serving/api.py, data_prep/app_catalog.py; app catalogue only)
The app catalogue now keeps each item's `gender`. The app hides the 267 kids' items (Boys 123, Girls 144; all of
them came in with the evaluation catalogue) and skips a result whose cosine with an already chosen result is
>= 0.97 (same product photographed again; the same girls' jeans had appeared 3x in one top 5). "Not sure" and
the shortlist size use only visible items. Checks: 33,141 rows, no missing gender, index vectors have unit
length (so the dot product is a cosine), evaluation catalogue unaffected (nothing hidden, no skipping), 6 tests pass.
Own jeans flat lay afterwards: looks right (jeans first; checked by Mouhammed in the app).
Wide-leg jeans photo (worn, white background; after the group-level bonus, Nov 19): badge "bottoms", top 5 all
Jeans (Blue, Grey, Blue, Blue, Blue), colour "very close" for all 5, similarity 0.67 / 0.67 / 0.66 / 0.62 / 0.59,
shortlist 66. Right kind and wash, wrong CUT: all 5 are men's straight/bootcut. Cause = catalogue coverage, not
the model: none of the 609 Jeans names in Fashion Product Images (2011-12) contain wide, baggy, flare, loose,
relaxed or palazzo (names are sparse, so not a proof); slim 29, skinny 15. Similarity can only rank what the
shop sells. Re-check this photo after the H&M catalogue (2018-20, mostly women's) is merged.

## Pre-registration (Nov 19, 00:00, BEFORE any of the experiments below)
Accuracy target (agreed definition): group P@5 on LookBench real photos (dedup protocol), goal 0.90.
Secondary: exact-item R@1 on real photos (no 90% goal; any significant gain counts).
LookBench dev/test split (evaluation/lookbench_split.py -> data/lookbench/real_streetlook/split.csv): queries
joined by same crop, same street photo or same target item (union-find, 621 clusters), shuffled with seed 490;
dev 500 queries, test 481; nothing shared between the sides. Baseline finetuned_random:
  dev  R@1 0.528, group P@5 0.866      test R@1 0.511, group P@5 0.871     (all 981: 0.520 / 0.869, check OK)
Rules: every new method is tuned ONLY on dev, with its grid and choice rule written here before running.
Default choice rule: highest dev R@1 such that dev group P@5 does not drop by more than 0.5 pt (for methods
aimed at group P@5: highest dev group P@5 such that dev R@1 does not drop by more than 0.5 pt).
Each chosen method is run ONCE on test; paired cluster bootstrap (clusters = the split clusters) vs
finetuned_random, Holm over all methods taken to test. Negative results are logged. D9 own photos stay last.

## Finding 31: failure analysis on LookBench dev (finetuned_random, 500 queries, top-20 rankings)
Exact item at rank 1: 264. Misses: 236, of which 206 (87%) have the RIGHT garment group at rank 1 but the wrong
item, and 30 the wrong group. True item at rank 2-5: 80, rank 6-20: 59, not in the top 20: 97.
Meaning: the model mostly finds the right kind of garment but not the right design (print, colour blocking,
details). Re-ranking the top 20 can win at most 139 of 236 misses (R@1 ceiling 0.81 on dev); the 97 outside
the top 20 need a better embedding (DINOv2 fusion, training on real street-to-shop pairs).

## Pre-registration: colour re-ranking on real photos (evaluation/color_rerank.py)
Re-rank the top 20 of finetuned_random: score = similarity + beta * colour similarity (cosine of 128-bin HSV
histograms, retrieval/color_hist.py). Variants: "plain" (whole images) and "masked" (gallery pixels that are
near-white, S < 26 and V > 230 on 0-255, are left out; product photos have white backgrounds, street crops do not).
Grid beta in {0, 0.02, 0.05, 0.1, 0.2} x 2 variants, DEV only, default choice rule. Check: beta = 0 reproduces
the saved R@1 and group P@5.
Dev result (results/color_rerank_dev.csv; check OK, beta = 0 gives 0.528):
  plain  beta 0 / 0.02 / 0.05 / 0.1 / 0.2:  R@1 0.528 / 0.540 / 0.532 / 0.524 / 0.514; group P@5 0.866 / 0.865 / 0.864 / 0.869 / 0.862
  masked beta 0 / 0.02 / 0.05 / 0.1 / 0.2:  R@1 0.528 / 0.540 / 0.528 / 0.530 / 0.510; group P@5 0.866 / 0.864 / 0.864 / 0.870 / 0.859
Choice by the rule: plain, beta = 0.02 (tie with masked on R@1, plain keeps more group P@5). Gain +1.2 pts on dev:
small, may be noise; goes to test in the final batch.

## Pre-registration: group bonus and DINOv2 fusion on real photos (evaluation/fusion_lookbench.py)
Embeddings saved once by evaluation/export_lookbench_features.py (check: reproduces every saved rank-1 picture);
all tuning in numpy on DEV only.
(a) Group bonus (the old "step 2"): score = similarity + w * P(gallery item's group | photo); P = zero-shot
    FashionSigLIP with the app prompts (serving/type_text_app.npy), summed per group; the gallery item's group
    comes from its label (the app also knows each item's type). w in {0, 0.02, 0.05, 0.1, 0.2}; rule for
    group-aimed methods (highest dev group P@5, dev R@1 drop <= 0.5 pt).
(b) DINOv2 fusion: score = (1 - a) * finetuned_random similarity + a * DINOv2 similarity (ViT-B/14, 518 px),
    a in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1}; default rule.
(c) Combined: the chosen (b), then the chosen (a) bonus, then the chosen colour re-ranking, each with its own
    chosen value, NO re-tuning; reported on dev, then taken to test with (a), (b), colour separately.
Check: a = 0, w = 0 reproduces the saved R@1 and group P@5 of finetuned_random.

## Finding 32: group bonus, DINOv2 fusion and colour on real photos (dev choice, then test ONCE)
Embeddings: evaluation/export_lookbench_features.py (check: rank-1 picture reproduced). Tuning:
evaluation/fusion_lookbench.py on DEV (check: plain model gives the saved 0.528). results/fusion_dev.csv:
  DINOv2 a = 0 / 0.1 / 0.2 / 0.3 / 0.5 / 1:   R@1 0.528 / 0.538 / 0.510 / 0.500 / 0.448 / 0.306
                                           group P@5 0.866 / 0.876 / 0.878 / 0.873 / 0.860 / 0.791
  group bonus w = 0 / 0.02 / 0.05 / 0.1 / 0.2: R@1 0.528 / 0.536 / 0.528 / 0.520 / 0.512
                                           group P@5 0.866 / 0.883 / 0.903 / 0.913 / 0.916
Choices by the rules: DINOv2 a = 0.1; group bonus w = 0.05 (w = 0.1 drops R@1 by 0.8 pt: not allowed).
Combined (a 0.1, w 0.05, colour beta 0.02, no re-tuning) on dev: R@1 0.534, group P@5 0.904.
(Exploratory, not pre-registered: without colour 0.528 / 0.903; without DINOv2 0.536 / 0.904.)
TEST, run once (evaluation/fusion_test_stats.py -> results/fusion_test.csv; paired cluster bootstrap vs
finetuned_random over the 621 split clusters; Holm over the 4 methods per metric; check: base = saved 0.511):
                  R@1 (base 0.511)                     group P@5 (base 0.871)
  colour          0.516  +0.4 [-0.4, +1.5]  p_Holm 1   0.866  -0.5 [-0.9, -0.2]  p_Holm <0.0005
  DINOv2          0.516  +0.4 [-1.5, +2.6]  p_Holm 1   0.867  -0.5 [-1.6, +0.6]  p_Holm 0.44
  group bonus     0.509  -0.2 [-0.7,  0.0]  p_Holm 1   0.905  +3.4 [+2.4, +4.5]  p_Holm <0.0005
  combined        0.522  +1.0 [-1.0, +3.3]  p_Holm 1   0.902  +3.1 [+1.9, +4.3]  p_Holm <0.0005
Meaning: the agreed target is reached on held-out real photos: group P@5 0.905 with the group bonus alone
(w = 0.05, the same weight the app already uses at type level), exact-item R@1 unchanged. Negative results:
DINOv2 is much weaker than our model on its own (dev R@1 0.306 vs 0.528) and adds nothing significant in
fusion; colour re-ranking costs a little group P@5 and gains no significant R@1. The hope that a
higher-resolution self-supervised model keeps design detail is NOT supported. Exact-item R@1 (~0.51) stays
the open problem (Finding 31: design-level misses); only real street-to-shop training is left as a lever.
App change after F32: serving/api.py now uses the GROUP-level bonus, score = similarity + 0.05 * P(item's group |
photo) (sum of the zero-shot type probabilities in the item's group), i.e. exactly the version tested on real
photos. Same w as before; serving/rerank.json records level "group" and the evidence. Not re-measured on
synthetic photos (the type-level numbers of F29 no longer describe the app exactly; to check with D9).
Decision (Nov 19): DeepFashion2 real-pair training skipped for now (only lever left for exact-item R@1, but
costly: access form, large download, hours of training); revisit only if > 1 week remains after G.

## Finding 33: Docker with the app catalogue (Dockerfile, docker-compose.yml, serving/latency.py)
The image now ships both catalogues and serves the app catalogue by default; a build step checks that the
saved app index matches (weights line + 33,141 image paths) so it is reused, not re-embedded on the CPU.
Check: a reference top 5 was saved from the Mac's own app API (MPS) for 105 synthetic medium photos
(results/latency_reference_app.csv); the container gives the same top 5 for 105/105.
CPU in Docker (Colima, 4 vCPU), 100 photos after 5 warm-up: server median 263 ms, p95 279 ms; round trip
median 265 ms, p95 282 ms (evaluation catalogue, F26: 266 / 282 ms). Five times more items cost nothing
measurable: scoring 33k x 768 dot products is tiny next to the two image-model forward passes.
Also new (not evaluated): optional crop box in the UI (two sliders, in-memory crop; default sends the
original bytes). LookBench queries are hand-drawn crops, so the evaluated setting is the marked-garment one.

## Finding 34: H&M items added to the app catalogue (data_prep/hm_catalog.py, data_prep/app_add_hm.py)
App catalogue 33,141 -> 54,178 items: 21,037 H&M products (2018-20; adult sections; 27 product types, at most
1,000 per type; mostly flat or plain-background photos). Each H&M type was mapped, before running, to an
existing app type (e.g. Hoodie -> Sweatshirts, Boots -> Casual Shoes), so badge prompts, the
out-of-catalogue threshold and the group bonus are unchanged; the page shows H&M's own type name.
Only the new images were embedded and appended (checks: old 33,141 vectors unchanged; 20 re-embedded
images reproduce the saved vectors, cosine > 0.999; the engine loads the merged index without re-embedding).
Old files kept as *_fpi. Evaluation numbers unaffected (evaluation catalogue).
Own photo (long tiered floral skirt on a person, crop box around the skirt): top 5 all skirts, maxi
and tiered or floral in the right tones; rank 1 a black-ground floral tiered maxi (H&M) close in design.
Shortlist size 1,784: with 54k items many more are about as similar as the best match; the shortlist was
calibrated on LookBench's 3,278-item gallery, so in the app it is only indicative (as already stated).
Colour names are the shops' own labels (H&M: one main colour per item), the swatch is measured; the page
now says "listed as".

### Finding 34 addendum (Nov 19, 13:10): jeans photos after the H&M merge; photo style beats cut
Qualitative only: 5 photos, not chosen at random, shown after the app was frozen; no setting was changed.
None is a D9 wardrobe photo (web/product photos). App catalogue 54,178 items, group bonus w = 0.05, no crop box.
- F29 flat lay (my_photos/jeanstrial.jpg, light-wash straight): top 5 all adult straight light/mid-wash jeans,
  all H&M flat lays, similarity 0.81-0.75 (before: kids' denim shorts first at 0.660; note kids' items were
  also hidden in between (F30), so the H&M effect is the flat-lay adult jeans and the higher similarity).
- F30 worn wide-leg (my_photos/images.jpg): top 5 IDENTICAL to before (same 5 men's straight/bootcut jeans
  photographed on a person, 0.67/0.67/0.66/0.62/0.59); only the shortlist grew (66 -> 182).
  Yet the catalogue now has the cut: of 999 H&M Trousers, 250 descriptions say denim, 37 denim + "wide",
  14 + "loose", 8 + "flare", 5 + "relaxed".
- Three more flat-lay web photos: wide-leg light wash with stains -> 5/5 H&M wide-leg light denim (0.66-0.68);
  faded black straight -> grey/black denim, 4 H&M + 1 FPI, #3 an H&M "Leggings/Tights" (still bottoms);
  dark greenish vintage wash, wide straight -> 5/5 H&M jeans (0.72-0.75), cut roughly right but the
  vintage tone is missed (3/5 "colour slightly different", all brighter blue).
Reading: with a flat-lay photo, the H&M flat-lay items with the right cut (incl. wide-leg) come first; with a
worn photo, the old worn-on-model jeans come first even though wide-leg items now exist. The model matches HOW
the garment was photographed before WHAT the cut is, FOR JEANS. Correction to F30: catalogue coverage is
necessary but not sufficient; photo style decides (same mechanism as F29, in the other direction).
Scope (added 13:27, after the sweater rehearsal below): a worn sweater matched flat H&M sweaters without trouble,
so photo style is not a general rule; likely jeans-specific (the old catalogue has hundreds of men's jeans on a model).
Not measured: the rank of the 37 wide-leg denim items for the worn photo (would need a small script).

### Demo rehearsal (Nov 19, 13:20): the author's own brown crew-neck sweater (not in the catalogue)
CORRECTION (13:40): this sweater is the AUTHOR'S (first written down as a friend's), so it is a D9 wardrobe
garment that was searched in the app before D9. Decision, made now, before any D9 photo: it stays in D9 like
every other garment (the protocol drops nothing), and is flagged in the D9 result. The app was not changed
after seeing it (frozen since the H&M merge), so seeing it cannot have tuned the method; dropping it now,
after seeing its results, would be the biased choice.
Descriptive only; app frozen, nothing changed. Same sweater, three photos:
                                  rank 1                          sim    shortlist  colour badges (top 5)
  worn, no crop (room, face)      H&M brown knit "Greenish Khaki" 0.71   1,386      4 very close, 1 different (beige)
  worn, crop box on the sweater   same item                       0.74   1,081      5 very close
  flat on a bed (other lighting)  same item                       0.74   1,299      5 slightly different
All 15 results are sweaters (group badge "tops"), mostly H&M flat photos. The crop box raised the top similarity
and shrank the shortlist; rank 1 was the same item in all three photos. Flat photo: the estimated colour moved
from #4b3e36 to #7d6f67 (lighter, greyer) with the lighting, so the colour badges changed while the ranking barely
did (as the README's lighting caveat says). Demo recipe chosen: worn photo + crop box around the garment.

## Pre-registration: "colour can't be judged" check for the app (serving/colour_check.py)
Problem: on black-and-white or very dark photos the colour estimate (median CIELAB of the image centre) is
meaningless but the app still shows confident colour badges. The check uses the QUERY photo only; catalogue
colours and the colour bands of Finding 24 are unchanged.
Two numbers per photo, over the whole photo (longer side 224 px): colourfulness = 95th percentile of CIELAB
chroma, brightness = 95th percentile of L*. Flag "black-and-white" if colourfulness < T_c; flag "too dark"
if brightness < T_L. Rule fixed now: T_c and T_L = the 1st percentile of each number over the DISTINCT
LookBench query crops of the DEV half (identical crops counted once), so at most ~1% of normal real photos
are flagged by construction. No other value is tried.
Check, once, on the distinct TEST-half crops and on 300 random catalogue images (seed 1): flag rate on the
original photos (false alarms; catalogue split into items listed Black/White/Grey vs the rest), and on
grayscale, brightness x0.5 (inside the training range 0.45-1.55), x0.3 and x0.15 (outside) versions; and how
often the colour estimate moves by more than the "different" band (Delta-E > 37.0) from the original.
Ship rule: ship if false alarms on test originals <= 2% and grayscale detection >= 95% on test crops;
the darkness flag is reported (no ground truth for "too dark" beyond the estimate moving).
Sanity check: the script first recomputes the colour bands of Finding 24 and stops if they differ from
serving/badges.json.


## Finding 35: "colour can't be judged" check (serving/colour_check.py -> results/colour_check.csv)
Check OK: the colour bands recomputed as 16.45 / 37.00, equal to serving/badges.json. Thresholds by the
pre-registered rule (1st percentile over 362 distinct dev crops): colourfulness < 3.64, brightness < 54.41.
Test (338 distinct test crops), flag rate black-and-white / dark / either:
  original 0.012 / 0.015 / 0.027; grayscale 1.000 / 0.015 / 1.000; brightness x0.5 0.021 / 1.000 / 1.000;
  x0.3 0.092 / 1.000 / 1.000; x0.15 0.467 / 1.000 / 1.000.
Catalogue (300 random items, seed 1), black-and-white flag on the originals: 0.006 for items listed in a
colour (172), 0.227 for items listed Black / White / Grey (128); grayscale versions flagged 1.000 in both.
Pre-registered ship rule FAILED: 2.7% false alarms on test originals (limit 2%). Cause: two flags, each set to
fire on ~1% of normal photos, together fire on ~2.7%; the rule left no room for dev-to-test variation
(a design error in the rule, not bad luck).
Dark flag: fires on 100% of x0.5 photos (inside the training range 0.45-1.55), although there the colour
estimate never moved past the "different" band (0%); it moved only at x0.3 (39%) and x0.15 (48%).
Brightness alone does not tell whether colour can be judged.
Black-and-white flag alone: catches 100% of grayscale photos, 1.2% false alarms on real test crops, 0.6% on
colourful catalogue items, but 22.7% on plain-background photos of black / white / grey items (no colour
anywhere: the photo cannot tell this apart from a black-and-white photo).
Also: grayscale never moved the estimate past Delta-E 37 (0%), so "moved past the different band" is a poor
yardstick for black-and-white photos; the harm there is a meaningless grey swatch and confident badges.
Decision (made AFTER seeing the test numbers, so the test half is used up for this check; a display feature,
not an accuracy claim): ship the black-and-white flag only (it meets both ship conditions on its own) and
drop the dark flag. App: below colourfulness 3.64, no swatch and no colour badges, and the message "This
photo has almost no colour (a black-and-white photo, or a black, white or grey garment), so the colour
can't be judged". Darkness moves to task 3 (colour-badge errors on darkened real photos).
Not fixed by this check: on whole photos the estimate can pick up the background (a wallet on a beige
backdrop gave #9c7f68, the backdrop, not the dark-brown wallet); the page now suggests marking the garment.

## Pre-registration: app catalogue clean-up (data_prep/app_cleanup.py)
Problems found in the app: H&M fabric close-ups reach the top 5; some Kaggle rows show another product
than their label (item 13378, "Timberland Unisex Infant Pink Casual Shoes", is a black Fossil wallet);
kids' items listed as Men/Women/Unisex are not hidden (~444 by product name). App catalogue only; the same
rules are COUNTED on the evaluation catalogue (first 6,197 rows) but never applied there.
Rules, fixed now, no value tried twice (pretrained FashionSigLIP, zero-shot):
  closeup     : 2-way choice "a photo of a close-up of fabric" vs the item's own type prompt; hide if P >= 0.5.
  wrong_image : zero-shot group (app prompts, 7 groups, as the group badge) differs from the listed group
                with group probability >= 0.9 (the badge's own threshold); hide.
  kids_name   : Kaggle product name matches kid(s), kid's, kidswear, infant(s), baby, babies, toddler, boy(s),
                boy's, girl(s), girl's (whole words, any case); hide. H&M items come from adult sections.
Check, by eye: for each rule a contact sheet of 24 random items it drops and 24 random items it keeps
(seed 0; kids: only drops not already hidden by gender). A rule ships only if the author judges >= 20/24
of its dropped items to be correct drops (judged by the author, not independent). Report how many items
each rule drops, per source.
Caveat: wrong_image uses the same zero-shot model as the group badge and the group bonus, so hiding its
disagreements makes the app look more self-consistent; later app measurements of the group badge are
slightly flattered by it.
Sanity checks: the pretrained embeddings of the first 6,197 app rows equal the saved fashion_siglip.npy;
the batch zero-shot group equals TypePredictor("app").analyse on 5 random items.

## Finding 36: app catalogue clean-up (data_prep/app_cleanup.py -> serving/app_hidden.csv, results/app_cleanup.csv)
Checks OK: pretrained embeddings of the first 6,197 app rows equal the saved fashion_siglip.npy; batch
zero-shot groups equal TypePredictor("app").analyse on 5 random items. Items each rule would hide
(evaluation rows / Kaggle extra / H&M / whole app): closeup 1 / 24 / 154 / 179; wrong_image 30 / 32 / 413 / 475;
kids_name 395 / 411 / 0 / 806 (542 not already hidden by gender).
Contact sheets (report/figures/cleanup_*_dropped.png, 24 random drops each; counted by Claude, counts
accepted by the author, as in Finding 20; not an independent annotation), correct drops:
- kids_name 24/24 -> SHIPPED. No kids' item seen among the 24 kept. The app now hides 809 items
  (267 by gender, 542 more by name); "not sure" and the shortlist use only visible items, as before.
- closeup 11/24 real fabric close-ups (15/24 counting 4 items whose image is another product) -> NOT shipped.
  The rest are whole garments with odd H&M labels (scarves listed as "Hat/beanie"). One close-up was missed
  among the 24 kept (a partial bag shot).
- wrong_image about 0/24 -> NOT shipped. It flags H&M's own label noise on normal garments, not broken
  items: parkas and hooded jackets listed as "Hoodie" (mapped to Sweatshirts; 163 of the 413 H&M hits),
  jumpsuits and playsuits listed as "Trousers"/"Shorts", shirts listed as "Jacket". It did not catch the
  wallet (13378); the wallet is hidden anyway by kids_name (its name says "Infant").
Meaning: (1) fabric close-ups and wrong-image rows stay a stated limitation of the app catalogue; no second
rule was tried (re-tuning after seeing the sheets would be fishing). (2) H&M's own type labels are noisy, so
the app's group badge and group bonus inherit that noise on H&M items. (3) 395 of the 6,197 evaluation
items (6.4%) are kids' items by name; harmless for the evaluation (nothing is hidden there), noted as label noise.
Limitation: kids' items are hidden, not detected in the photo; a photo of a kids' garment gets the most
similar ADULT items, with no warning.

## Pre-registration: whole photo vs crop on LookBench (task 2a, evaluation/whole_photo.py)
Question: every real-photo number so far uses LookBench's hand-drawn crop, but the app sends the WHOLE
photo by default (crop box optional). How much worse is the app's default setting?
Runs (no tuning, nothing chosen): evaluation/run_lookbench.py --query_image raw_image for finetuned_random
and fashion_siglip, same 3,278 gallery and dedup protocol; plain similarity, no group bonus.
Reported on all 981 queries and on the TEST half: R@1, R@5, group P@5 for crop and whole photo; paired
cluster bootstrap over the split clusters for (1) finetuned_random whole - crop, (2) fashion_siglip
whole - crop, (3) on whole photos finetuned_random - fashion_siglip; Holm over these 3 per metric and side.
Also: the R@1 ceiling on whole photos (one photo is the query for several items but gets one ranking),
and R@1 for photos with one query vs photos shared by several queries.
Sanity check: the crop numbers reproduce the saved R@1 (lookbench_summary.csv) and group P@5 (group_p5.csv).
Expectation written now: whole photo is worse; the size of the drop is the result.

## Finding 37: whole photo vs crop on LookBench (task 2a, evaluation/whole_photo.py -> results/whole_photo.csv)
Checks OK: crop R@1 and group P@5 reproduce the saved values for both models (0.520 / 0.869 and 0.486 / 0.869).
981 queries come from 621 distinct whole photos; 589 queries share their photo with other queries. R@1 ceiling
on whole photos (one ranking per photo): 0.902, so the shared photos are NOT what limits the score (many
gallery pictures are outfit photos that show several of the photo's items).
                         all 981: R@1 / R@5 / group P@5      test 481: R@1 / R@5 / group P@5
  finetuned_random crop   0.520 / 0.694 / 0.869               0.511 / 0.701 / 0.871
  finetuned_random whole  0.406 / 0.549 / 0.668               0.407 / 0.559 / 0.684
  fashion_siglip   crop   0.486 / 0.657 / 0.869               0.489 / 0.665 / 0.874
  fashion_siglip   whole  0.376 / 0.498 / 0.649               0.378 / 0.509 / 0.664
Paired cluster bootstrap, Holm over 3 tests per metric and side (test half):
  finetuned_random whole - crop: R@1 -0.104 [-0.145, -0.063], group P@5 -0.187 [-0.232, -0.142], p_Holm < 0.001
  fashion_siglip whole - crop:   R@1 -0.110 [-0.156, -0.069], group P@5 -0.210 [-0.256, -0.163], p_Holm < 0.001
  whole photos, finetuned_random - fashion_siglip: R@1 +0.029 [+0.006, +0.054] p_Holm 0.020;
                                                   group P@5 +0.020 [+0.005, +0.035] p_Holm 0.013
(All 981 gives the same picture: -0.114 / -0.200 and +0.030 / +0.019, all p_Holm <= 0.001.)
Meaning: (1) the app's default (whole photo) is much worse than the evaluated setting: about 11 points of
exact-item R@1 and about 20 points of group P@5. The headline group P@5 (0.905 with the group bonus) is a
CROP number; the app's default is nearer 0.67-0.68 (without the bonus; the bonus was not measured on whole
photos). (2) Marking the garment matters more than any model change tried so far (crop vs whole: +10 to +11
points R@1; fine-tuning vs pretrained: +3). (3) Fine-tuning still helps on whole photos, and there it also
gives a small significant group P@5 gain (+2.0 points), which it did not on crops (Finding 21).
(4) Photos shared by several queries are harder in BOTH settings (whole 0.28 vs 0.60, crop 0.38 vs 0.72):
outfit photos with several garments, not only the whole-photo ambiguity.
Not measured here: the 54k app catalogue (task 2b) and the group bonus on whole photos.

## Pre-registration: LookBench against the official corpus (task 2b, evaluation/lookbench_full.py)
Checked on Hugging Face (srpone/look-bench, version_info.json): the official corpus of each subset is its
own gallery plus the 58,275 shared "noise" images; for real_streetlook 3,278 + 58,275 = 61,553 entries,
the number in the paper's Table 1. Noise images have no category or attributes (category "noise data"),
so on this corpus only exact-item metrics are defined (R@1, R@5); group P@5 stays a 3,278-gallery metric.
(The paper lists 1,000 real_streetlook queries; the release has 981, the set we use.)
Runs, nothing tuned: finetuned_random and fashion_siglip, crop and whole photo; noise embedded once per
model, gallery embeddings = the saved ones; dedup protocol (identical image files merged). Reported:
R@1, R@5 (95% CIs), share of noise images in the top 5, and paired cluster bootstrap finetuned_random -
fashion_siglip (split clusters) on all 981 and on the test half.
Sanity check: ranking only the 3,278 gallery entries reproduces the saved R@1 of every query.
Not in this step: re-calibrating "not sure" and the shortlist at this scale (next step), ensemble_mv
(three times the embedding cost), and near-duplicates between noise and gallery (only exact copies merged).
Decision to take after the results, with the author: which corpus later experiments use.

## Finding 38: LookBench against the official 61,553-image corpus (task 2b, evaluation/lookbench_full.py)
Checks OK: 0 noise images are byte-identical to a gallery picture; for both models and both views,
ranking only the 3,278 gallery entries reproduces the saved rank-1 result for 981/981 queries.
R@1 / R@5 on the full corpus (all 981; 95% CI), with the 3,278-gallery R@1 and the share of noise in the top 5:
  finetuned_random crop   R@1 0.501 [0.471, 0.531]  R@5 0.677   (gallery 0.520)  noise in top 5: 16%
  finetuned_random whole  R@1 0.382 [0.353, 0.412]  R@5 0.521   (gallery 0.406)  noise in top 5: 23%
  fashion_siglip   crop   R@1 0.472 [0.441, 0.504]  R@5 0.643   (gallery 0.486)  noise in top 5: 14%
  fashion_siglip   whole  R@1 0.353 [0.323, 0.383]  R@5 0.474   (gallery 0.376)  noise in top 5: 21%
finetuned_random - fashion_siglip, paired cluster bootstrap (split clusters):
  crop  all 981: R@1 +0.029 [+0.011, +0.046] p=0.001; R@5 +0.034 p<0.001
        test 481: R@1 +0.023 [+0.000, +0.047] p=0.070; R@5 +0.025 [-0.004, +0.056] p=0.113
  whole all 981: R@1 +0.030 [+0.010, +0.050] p=0.004; R@5 +0.047 p<0.001
        test 481: R@1 +0.025 [+0.002, +0.050] p=0.044; R@5 +0.052 [+0.029, +0.079] p<0.001
Meaning: (1) a corpus 19 times bigger costs only about 2 points of R@1 (crop 0.520 -> 0.501, whole photo
0.406 -> 0.382): the 58k distractors rarely beat the true item, although they fill 14-23% of the top-5 slots.
(2) The fine-tuning gain is about +2.5 to +3 points R@1 at this scale too; on the test half alone it is
borderline for crops (p = 0.07) and significant for whole photos (p = 0.044); with whole photos the R@5 gain
is larger (+5 points). These p-values are not Holm-corrected. (3) Not comparable to the paper's FashionSigLIP
55.2% "fine R@1" (different correctness rule, Finding 5).
Only exact-item metrics exist on this corpus (noise images have no category); group P@5 stays a
3,278-gallery metric.

## Decision (Nov 19, after Finding 38): protocol for all later experiments
Main protocol for exact-item metrics (R@1, R@5): the OFFICIAL corpus (3,278 gallery + 58,275 noise =
61,553 entries), dedup protocol. Group P@5 stays on the 3,278 gallery (noise has no category). The old
3,278-gallery exact-item numbers are kept as a clearly labelled secondary protocol. Later re-rankers work
on the top 20 of the full corpus. Same dev/test split, same paired cluster bootstrap, Holm across methods.

## Pre-registration: shortlist re-calibration at the official-corpus scale (serving/calibrate_full.py)
The shortlist q (relative split conformal, alpha 0.1) is re-calibrated with the best similarity taken over
the 61,553-image corpus instead of the 3,278 gallery, for crops and for whole photos separately (all 981
photos, finetuned_random). Rule fixed now: the app uses the LARGER of the two q values, so the 90% holds on
LookBench for both kinds of photo the app can receive. Reported: q for each view, coverage over 200
random 50/50 splits, median shortlist size on the corpus, coverage of the OLD q at the new scale, and the
coverage of each view's q on the other view. "Not sure" is NOT re-calibrated (its loss needs garment
groups, which noise images lack); reported only: its answer rate at the current lambda on the gallery vs
the full corpus. Sanity checks: gallery-only crops give back the saved q; the best full-corpus similarity
equals the saved rank-1 scores of lookbench_full.py.

## Finding 39: shortlist re-calibrated at the official-corpus scale (serving/calibrate_full.py -> results/calibration_full.csv)
Checks OK: gallery-only crops give back the saved q (0.228381); the best full-corpus similarity equals the
saved rank-1 score of lookbench_full.py for every photo.
                     q (61,553 corpus)  coverage (200 50/50 splits)  median shortlist  coverage of the OLD q
  crop               0.2297             0.902                        116 entries       0.897
  whole photo        0.2363             0.902                        878 entries       0.888
Crop q on whole photos covers 0.889; whole-photo q on crops covers 0.905. By the pre-registered rule the
app now uses q = 0.2363 (was 0.2284; old value kept in serving/calibration.json as q_gallery_3278).
"Not sure" (lambda 0.671, unchanged) answers 92% of crops on the gallery and 93% on the full corpus, but
only 49% of whole photos (both corpora).
Meaning: (1) a 19 times bigger corpus barely moves the shortlist constant (0.228 -> 0.230 for crops): the
old value still covers about 89-90%, consistent with Finding 38 (distractors rarely beat the true item).
(2) The honest 90% shortlist is long: about 116 items for a marked garment and about 880 for a whole
photo, out of 61,553; the page shows the number only when it is at most 50, so in practice it says "many".
(3) With a whole photo, "not sure" fires on about half the photos, which fits Finding 37 (whole photos
are much harder): the warning reacts to the harder setting even though it was calibrated on crops.

## Pre-registration: robustness to bad photos on REAL LookBench crops (task 3, evaluation/robustness.py)
Question: how much do imperfect photos hurt, inside vs outside the training augmentation? Our only
robustness numbers so far are synthetic (same augmentation as training).
Filters, fixed now, applied to the 981 real crops (sizes in px at the 224 input, as in phone_aug.py):
  inside the training augmentation: Gaussian blur radius 1.5 and 3; brightness x0.5
  outside: Gaussian blur radius 6; horizontal motion blur 5, 9, 15 px; brightness x0.3 and x0.15;
           half saturation and grayscale; warm colour cast (R, G, B multiplied by 1/0.85/0.65 and 1/0.7/0.45).
Methods: finetuned_random, fashion_siglip, colour histogram (reference baseline; task 4 adds the credible ones).
Metrics: R@1, R@5 on the official corpus (main protocol), group P@5 on the 3,278 gallery; "not sure" rate of
the app rule (finetuned_random); share of photos whose TRUE item gets the "different" colour badge, before
and after the black-and-white fix (Finding 35). All 981 queries (nothing is tuned).
Tests: drop from clean with a 95% cluster-bootstrap CI for each method; finetuned_random - fashion_siglip per
filter (R@1 and group P@5), paired cluster bootstrap over the split clusters, Holm over the 14 conditions
per metric. Expectation written now: small drops inside the augmentation, larger outside; fine-tuning should
help more inside than outside.
Sanity check: with no filter both models reproduce the saved per-query full-corpus R@1.

## Finding 40: robustness to bad photos on real LookBench crops (task 3, evaluation/robustness.py)
Checks OK: with no filter both models reproduce the saved per-query full-corpus R@1. All 981 crops, R@1 on
the 61,553 corpus (drop from clean, 95% cluster CI) / group P@5 on the gallery; * = outside the training augmentation.
                    finetuned_random                  fashion_siglip                 colour histogram
  clean             0.501            / 0.869         0.472            / 0.869        0.077 / 0.327
  gauss 1.5         0.490 (-1.0, n.s.) / 0.838       0.389 (-8.3)     / 0.790        0.072
  gauss 3           0.355 (-14.6)    / 0.718         0.147 (-32.5)    / 0.429        0.058
  gauss 6 *         0.118 (-38.2)    / 0.415         0.022 (-45.0)    / 0.191        0.042
  motion 5 *        0.501 (0.0)      / 0.848         0.444 (-2.8)     / 0.843        0.075
  motion 9 *        0.454 (-4.7)     / 0.828         0.368 (-10.4)    / 0.795        0.068
  motion 15 *       0.368 (-13.3)    / 0.751         0.244 (-22.8)    / 0.659        0.066
  dark x0.5         0.501 (0.0)      / 0.865         0.461 (-1.1)     / 0.864        0.000
  dark x0.3 *       0.490 (-1.0)     / 0.860         0.451 (-2.1)     / 0.850        0.000
  dark x0.15 *      0.391 (-10.9)    / 0.829         0.358 (-11.4)    / 0.790        0.000
  half colour *     0.505 (+0.4)     / 0.866         0.472 (0.0)      / 0.864        0.024
  grayscale *       0.379 (-12.1)    / 0.846         0.388 (-8.4)     / 0.847        0.001
  warm cast mild *  0.485 (-1.5)     / 0.861         0.456 (-1.6)     / 0.861        0.000
  warm cast strong* 0.452 (-4.9)     / 0.862         0.432 (-4.0)     / 0.855        0.000
finetuned_random - fashion_siglip, R@1 (paired cluster bootstrap, Holm over the 14 conditions):
  clean +2.9 (p_Holm 0.006); gauss 1.5 / 3 / 6: +10.1 / +20.8 / +9.6; motion 5 / 9 / 15: +5.6 / +8.6 / +12.4
  (all p_Holm < 0.001); dark x0.5 / x0.3 / x0.15: +4.0 / +4.0 / +3.4 (p_Holm < 0.001 / 0.006 / 0.052);
  half colour +3.3 (< 0.001); grayscale -0.9 (0.43); warm cast mild / strong +3.0 / +1.9 (0.052 / 0.27).
  Group P@5: big gains only under blur (gauss 3 +29.0, gauss 6 +22.4, motion 15 +9.2 points); none for
  colour changes.
"Not sure" (finetuned_random): 7% of clean photos, 15% at gauss 3, 19% at gauss 6, 14% at motion 15, 22% at
dark x0.15. The colour badge calls the TRUE item "different" for 4% of clean photos, 12% / 36% / 47% at dark
x0.5 / x0.3 / x0.15 and 4% for grayscale; after the black-and-white fix: grayscale 0%, dark x0.3 33%, dark x0.15 27%.
Meaning: (1) the expectation was only partly right. Inside the augmentation, mild blur and half brightness cost
nothing, but blur radius 3 (the edge of the training range) already costs 15 points. (2) The fine-tuning's
main value is ROBUSTNESS TO BLUR: on clean real photos it adds +3 points, on blurred ones +6 to +21 points,
including motion blur, a kind of blur it never saw in training. For colour changes it never saw (grayscale,
colour casts) it adds nothing beyond its clean-photo gain. So the synthetic augmentation does transfer to
real photos, for the kinds of damage it simulates. (3) Very blurry photos break every model (gauss 6: R@1
0.12), and "not sure" catches few of them (fires on 19% while R@1 is 12%): it is a weak safety net.
(4) Dark photos are the colour badge's real problem: even after the fix, 27-33% of very dark photos get a
wrong "different" badge for the true item (the darkness flag was dropped in Finding 35).
(5) The colour histogram finds the exact item 7.7% of the time on real crops (vs 1% on synthetic photos) but
fails completely under any lighting or colour change (R@1 0.000), as expected for a pure colour method.
"Not sure" numbers are only meaningful for finetuned_random (lambda is calibrated on its similarity scale).

## Pre-registration: keyword search baseline, "how shops work today" (task 4a, evaluation/keyword_baseline.py)
Replaces the strawman baselines (colour histogram; type menu in random order). A shopper types what they see
and the shop ranks items by text match: BM25 (k1 = 1.2, b = 0.75, fixed, no tuning) over each gallery item's
LookBench text (category + main attribute + other attributes; read from Hugging Face, checked row by row
against our saved files). IDEALISED shopper: the query text is LookBench's own annotation of the photo, so it
always uses the shop's words. Two variants: keywords_basic (category + main attribute) and keywords_detailed
(+ other attributes). Ties broken at random, averaged over 50 orders.
Protocol: the 3,278 gallery only (dedup), because the noise images have no text and would be invisible to
keyword search on the full corpus; the photo model is compared on the same gallery (saved finetuned_random
results). Metrics: R@1, R@5, group P@5, on all 981 and on the test half; paired cluster bootstrap photo -
keywords for R@1 and group P@5, Holm over the 2 variants. Also reported: how often a query's annotation is
identical to its true item's annotation (how idealised the shopper is).
Expectation written now: keywords get the KIND of garment right (high group P@5) but rarely the exact item.

## Finding 41: keyword search baseline (task 4a, evaluation/keyword_baseline.py -> results/keyword_baseline.csv)
Checks OK: the Hugging Face text lines up row by row with our saved LookBench files; the saved photo-model
gallery results are reproduced (R@1 0.520, group P@5 0.869). 3,278-gallery protocol (dedup), random tie-breaking
averaged over 50 orders. 35.8% of queries have an annotation word-for-word identical to their true item's.
                                   all 981: R@1 / R@5 / group P@5      test 481: R@1 / R@5 / group P@5
  keywords_basic (category + main)  0.309 / 0.626 / 0.981              0.312 / 0.634 / 0.983
  keywords_detailed (+ other attr.) 0.490 / 0.753 / 0.986              0.486 / 0.760 / 0.986
  photo (finetuned_random)          0.520 / 0.694 / 0.869              0.511 / 0.701 / 0.871
Photo - keywords (paired cluster bootstrap, Holm over the 2 variants), test half:
  basic:    R@1 +0.199 [+0.121, +0.284] p_Holm < 0.001;  group P@5 -0.112 [-0.144, -0.085] p_Holm < 0.001
  detailed: R@1 +0.025 [-0.042, +0.096] p_Holm 0.46 (n.s.); group P@5 -0.115 [-0.145, -0.089] p_Holm < 0.001
Meaning, stated plainly because it goes against us: (1) the expectation was WRONG for the detailed variant:
an idealised shopper who types the category and every attribute in the shop's own words finds the exact item
first about as often as our photo model (0.49 vs 0.51 on the test half, no significant difference), and finds the
right kind of garment far more often (0.99 vs 0.87). (2) The photo model clearly beats the shorter, more realistic
query (category + main attribute): +20 points R@1. (3) Typing the category makes group P@5 almost trivial (0.98),
which confirms that group P@5 is a sanity floor, not a headline: a text box gets it nearly for free.
Why the idealised shopper is so strong here, and why it is optimistic: the query words are LookBench's own
annotation of the photo (identical to the target's annotation for 36% of queries); real shoppers do not know the
words "applique" or "suede", which is the problem the app exists for (the user cannot describe the garment). And
the gallery is small (2,537 distinct pictures); in a real shop many more items share the same few attribute words,
so ties would grow. This could not be measured: the noise images have no text.
Implication: the honest pitch is not "photos beat text" but "photos match a perfect description without the
shopper having to produce one"; and photo + a typed or menu-chosen category is the natural combination
(Finding 14: type menu + our ranking, R@1 0.554 on the gallery, not tested on the frozen split).

## Pre-registration: classical (non-deep-learning) image baseline (task 4b, evaluation/classical_baseline.py)
Replaces the plain colour histogram as the non-ML image baseline. Each image (resized to 128 x 256) is
described by: masked HSV colour histogram (8x4x4; near-white pixels S < 26 and V > 230 left out), masked LBP
texture histogram (8 neighbours, radius 1, uniform, 10 bins) and HOG shape (64 x 128 grayscale, 9 orientations,
16 px cells, 2x2 blocks); each normalised; score = weighted sum of the three cosines.
Weights: grid of all settings in steps of 0.1 that sum to 1 (66), chosen by the highest DEV R@1 on the official
corpus (main protocol); then run ONCE on the TEST half. Variant "classical + menu": the shopper picks the right
garment group from a menu (idealised), and only gallery items of that group are ranked (gallery only: noise has
no labels). References, not tested: colour only (masked) with the same features, and the same menu in front of
our photo model. Tests on the test half: photo - classical (full corpus) and photo (gallery, no menu) - classical
+ menu, R@1, paired cluster bootstrap, Holm over these 2.
Sanity check: saved finetuned_random query embeddings reproduce the saved gallery R@1 (0.520).
Expectation written now: far below the photo model on exact items; the menu helps it a lot.

## Finding 42: classical image baseline (task 4b, evaluation/classical_baseline.py -> results/classical_*.csv)
Check OK: saved finetuned_random query embeddings reproduce the saved gallery R@1 (0.5199).
Dev choice (66 weight settings, official corpus): best dev R@1 0.112 for colour 0.3 / texture 0.2 / shape 0.5,
tied with 0.3 / 0.1 / 0.6 (the first in the sorted table was taken; the two differ by 0.1 in two weights).
Shape matters most, colour second, texture least. TEST half (481), run once:
                                corpus           R@1 [95% CI]          R@5     group P@5 (gallery)
  photo (finetuned_random)      full 61,553      0.499                 0.688   (0.871, Finding 37)
  classical                     full 61,553      0.123 [0.096, 0.154]  0.162   0.387
  colour only (masked)          full 61,553      0.081 [0.056, 0.106]  0.123   0.315
  classical + menu (idealised)  gallery 3,278    0.237 [0.200, 0.277]  0.360   1.000
  photo + menu (reference)      gallery 3,278    0.536 [0.491, 0.580]  0.713   1.000
Photo - classical (full corpus): R@1 +0.376 [+0.323, +0.435]; photo on the gallery without menu (0.511) -
classical + menu: +0.274 [+0.220, +0.329]; both p_Holm < 0.001.
Meaning: the expectation was right. A fair classical system (colour + texture + shape, tuned on dev) finds the
exact item first 12% of the time on the official corpus, 4 times less than the photo model; even with an
idealised menu in front it reaches 24%, half of the photo model without a menu. The menu adds +2.5 points to the
photo model (0.511 -> 0.536 on the same test half), consistent with Finding 14.
Note on the code: the LBP step received a float copy of the grayscale image (a warning was printed); skimage
converts every input to float internally, so the numbers are the same as with integer input.
Baseline ladder on the real test half (exact item first): colour histogram 0.08, classical 0.12 (full corpus);
classical + menu 0.24, short keywords 0.31, detailed idealised keywords 0.49 (gallery); photo 0.50 (full corpus),
0.51 (gallery); photo + menu 0.54 (gallery).

## Pre-registration: learned re-ranker trained on real photos (task 5, evaluation/rerank_features.py + rerank_model.py)
Problem (Finding 31): most misses are the right kind of garment with the wrong design; one embedding decides on
its own what matters. Stage 1: finetuned_random's top 20 distinct pictures on the official 61,553-image corpus
(main protocol; true item in the top 20 bounds what any re-ranker can do). Stage 2: a model scores each
(photo, candidate) pair from 9 features, fixed now: fine-tuned similarity, gap to rank 1, rank, pretrained
FashionSigLIP similarity, DINOv2 similarity (CLS, 224 px), DINOv2 patch matching (mean over the photo's 256
patches of the best cosine with the candidate's patches), masked HSV colour-histogram cosine, CIEDE2000 colour
difference, and P(candidate's zero-shot group | photo). Label: candidate is the true item.
Models, fixed settings: logistic regression (standardised, C = 1) and HistGradientBoostingClassifier (200 trees,
lr 0.05, 15 leaves, seed 0), trained pointwise. Choice on DEV only: 5-fold cross-validation by split cluster
inside dev; the higher cross-validated dev R@1 wins (tie: logistic regression). The chosen model is trained on
all dev and run ONCE on TEST, whatever its dev result; paired cluster bootstrap vs stage 1 (R@1, R@5).
Descriptive, after the test run (nothing chosen with them): learning curve with 50 / 100 / 250 / 500 dev photos
(20 random subsets) -> test R@1; permutation importance on test; CPU latency of stage 2.
Not in this step (decided now, may follow): extra training data from real_studio_flat (needs a leakage check
first) and a linear adapter ablation. Group P@5 is not defined on the full corpus and is not reported here.
Sanity checks: recomputed fine-tuned similarity equals the saved stage-1 score of every pair; the stage-1 order
reproduces the saved dev and test R@1.
Expectation written now: a small gain or none (fixed-weight colour and DINOv2 fusion did not help, Finding 32).

## Finding 43: learned re-ranker on real photos (task 5, evaluation/rerank_*.py -> results/rerank_*.csv)
Checks OK: recomputed fine-tuned similarity equals the saved stage-1 score (max difference 2.4e-07); the stage-1
order reproduces the saved dev (0.502) and test (0.499) R@1 query by query. (The features script printed
"saved 0.5010": that was the rounded 0.501; the exact saved value is 491/981 = 0.5005, as computed.)
True item in the top 20 (ceiling of any re-ranker): 0.771 overall, 0.764 on dev.
Dev, 5-fold CV by cluster: stage 1 0.502, logistic regression 0.506, boosted trees 0.508 -> boosted trees chosen.
TEST, run once (481 photos, official corpus):
  stage 1 (finetuned_random)   R@1 0.499   R@5 0.688
  re-ranker (boosted trees)    R@1 0.520   R@5 0.709
  difference R@1 +0.021 [-0.004, +0.050] p = 0.12; R@5 +0.021 [-0.004, +0.048] p = 0.12 (paired cluster bootstrap)
Learning curve (descriptive; test R@1, mean of 20 random training subsets): 50 dev photos 0.486 (below stage 1),
100: 0.504, 250: 0.513, 500: 0.520; still rising at 500.
Permutation importance on test (R@1 lost when the feature is shuffled): fine-tuned similarity 0.151, gap to rank 1
0.072, pretrained similarity 0.025, rank 0.015, group agreement 0.012, colour difference 0.011, patch matching
0.009, colour histogram 0.005, DINOv2 similarity 0.003.
CPU latency of stage 2: 76 ms per photo (4 threads), IF the candidates' DINOv2 patch tokens are stored with the
catalogue (for the 54k app catalogue that is about 54k x 256 x 768 numbers, ~21 GB in half precision: not
practical as is).
Meaning: NEGATIVE / INCONCLUSIVE at the pre-registered test. The re-ranker gains +2.1 points R@1 on held-out real
photos, in the expected direction but not significant (CI includes 0). The learning curve is the more informative
result: with 50 labelled photos the re-ranker is worse than doing nothing, and it improves steadily up to the 500
available, so the method is data-limited rather than useless. The features meant to catch "right garment, wrong
design" (patch matching, DINOv2, colour) contribute little; the model mostly re-weighs the fine-tuned and
pretrained similarities and their gaps. The expectation (small gain or none) was right.

## Pre-registration: re-ranker with extra real training photos from real_studio_flat (evaluation/rerank_studio.py)
Reason: Finding 43's learning curve was still rising at 500 labelled photos. Extra data: LookBench real_studio_flat
(1,011 studio / flat-lay photos, 3,951-item gallery), used for TRAINING only.
Leakage rule, fixed now: a studio photo is dropped if its own image, or any gallery picture of its true item, is
byte-identical or a near-duplicate (pretrained FashionSigLIP cosine >= 0.97, rule of Finding 18) of a
real_streetlook TEST query crop or of a gallery picture of a TEST item. Counts are reported.
Studio pairs: stage 1 = finetuned_random top 20 on real_studio_flat's official corpus (its gallery + the 58,275
noise images); the same 9 features as Finding 43. Model: Finding 43's choice (boosted trees, same settings;
sklearn's automatic early stopping, which would switch on above 10,000 rows, is switched off so nothing changes),
trained on dev + studio pairs. Reported: dev 5-fold CV by cluster (studio always in training) next to Finding 43's
0.508; then ONCE on test: paired cluster bootstrap vs stage 1, Holm over the two re-rankers taken to test
(Finding 43's and this one); this one minus Finding 43's, descriptive.
Sanity check: the dev-only model reproduces Finding 43's test R@1 (0.520).
Expectation written now: small extra gain at most; studio photos are a different kind of photo from street crops.

## Finding 44: re-ranker with extra studio training photos (evaluation/rerank_studio.py -> results/rerank_studio*.csv)
Check OK: the dev-only model reproduces Finding 43's test R@1 (0.5198).
Leakage check: 0 of 1,011 studio photos and 0 of 3,951 studio gallery pictures are identical or near-duplicates
(cosine >= 0.97) of a streetlook test crop or test-item picture; all 1,011 studio photos kept.
Studio stage 1 is much easier: the true item is in the top 20 for 92.7% of studio photos (street dev: 76.4%).
                                     dev CV R@1   test R@1   test R@5   vs stage 1 (test)
  stage 1 (finetuned_random)         0.502        0.499      0.688
  re-ranker, dev only (Finding 43)   0.508        0.520                 +0.021 [-0.004, +0.050] p 0.120, p_Holm 0.240
  re-ranker, dev + studio            0.484        0.516      0.705      +0.017 [-0.007, +0.046] p 0.226, p_Holm 0.240
  dev + studio minus dev only:       -0.004 [-0.033, +0.027] p 0.82
Meaning: NEGATIVE. Adding 1,011 studio photos does not help: on dev it makes the re-ranker worse than doing nothing
(0.484 vs 0.502), and on test it changes nothing (-0.4 points, n.s.). After Holm neither re-ranker is significant
(p_Holm 0.24). The learning curve of Finding 43 rose with more STREET photos; studio and flat-lay photos are a
different kind of photo (and an easier task), so they do not substitute. Conclusion for task 5: a learned re-ranker
on real photos gives at most about +2 points R@1 here, not significant with 500 street photos; more labelled street
photos of the shop's own items are what it would need (Finding 43's curve).

## Pre-registration: latency budget and the optional type model (task 6, serving/latency_cpu.py)
Budget, fixed now: at most 500 ms server time per search, median, on 4 CPU threads. Reason: about 1 s is the
limit for an answer to feel immediate; a phone upload and the page take roughly the other half.
Question: Finding 26 says about half of the time is the second model (pretrained FashionSigLIP for the type
badge, the group bonus and the out-of-catalogue warning); this was stated, not measured. New API switch
DAYFIT_TYPE_MODEL=off skips that model (no type badge, plain similarity order, no warning).
Measurement: the API's own search function in-process on the Mac CPU (DAYFIT_DEVICE=cpu, 4 threads), 100
synthetic medium photos after 5 warm-up, for (evaluation catalogue, on), (app catalogue, on), (app, off).
Reported: median and 95th percentile server time, and the ratio off / on. Not Docker: the Mac's CPU is faster
than the container's, so the absolute numbers are optimistic; Docker gave 266 ms (Finding 26) on 4 vCPUs.
What switching it off costs is already measured: group P@5 0.905 -> 0.871 on the test half (Finding 32), and
no type badge or out-of-catalogue warning.
Sanity check: (evaluation, on) reproduces the saved top 5 for at least 98% of photos.

## Finding 45: latency budget and the optional type model (serving/latency_cpu.py -> results/latency_cpu.csv)
Check OK: on the CPU the evaluation catalogue gives the saved top 5 for 105/105 photos.
Server time per search, Mac CPU, 4 threads, 100 photos after 5 warm-up (median / 95th percentile):
  evaluation catalogue (6,197 items), type model on    124 / 125 ms
  app catalogue (54,178 items), type model on          129 / 131 ms
  app catalogue (54,178 items), type model off          73 /  74 ms
Meaning: (1) all settings are well inside the 500 ms budget; the Docker container (4 vCPUs, slower) measured
266 ms with the type model on (Finding 26), also inside it. (2) The second model takes 56 ms of 129 (44%):
Finding 26's "about half" is roughly right (off / on = 0.56). (3) Nine times more catalogue items cost 5 ms.
Decision: the type model stays ON by default. The app is inside the budget with it, and switching it off
loses the type badge, the out-of-catalogue warning and 3.4 points of group P@5 (Finding 32) to save ~56 ms.
DAYFIT_TYPE_MODEL=off remains for slower machines.

## README check (task 6, evaluation/readme_check.py)
Every README number that comes from a results file is recomputed from that file and must appear verbatim in the
README (39 checks). Mismatches found and fixed: (1) the synthetic fine-tuning gain was given as +17.7 points, which
is the FIRST (hard-batch) model of Finding 4; for the recommended random-batch model it is +15.6 (0.949 vs 0.793,
medium photos). (2) "confidence intervals are 4-5 points wide" was wrong: they are about +-4 to 5 points (twice
that wide). After the fixes: 0 mismatches. Numbers that exist only in this log (Docker 266 ms, MPS 36 ms, the
wide-leg jeans check, H&M label noise, the 500 / 481 split) are listed by the script for a check by eye.

## D9 amendment 2 (Nov 19, 21:25, BEFORE any wardrobe photo is taken; only the sweater of the 13:20 rehearsal has been searched)
Reason: the app changed today after the "frozen since the H&M merge" note, before any D9 photo existed:
(a) kids' items are also hidden by product name (Finding 36; changes which items can appear); (b) the shortlist
constant was re-calibrated (Finding 39; changes only the shortlist size); (c) the "colour can't be judged" flag
(Finding 35) and the type-model switch (Finding 45, default on) do not change the ranking. Decision: D9 tests the
app at the commit on which it is run (recorded with the result), i.e. with all of these; nothing else in the
protocol changes. Still the whole photo, no crop box, as pre-registered (Finding 37 now shows this is the harder
setting). Expectation written now, from Finding 37 (whole street photos, gallery, no bonus: group P@5 0.67-0.68):
own photos with the garment filling the frame should be easier than street photos; the primary group P@5 is
expected between 0.70 and 0.90. Any value is reported.
