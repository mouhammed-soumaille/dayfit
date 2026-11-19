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
