# DAYFIT results log

## How to read this log (one-page index, Nov 19)

This is my lab notebook, kept in the order I did things. Later findings sometimes correct earlier ones (the
corrections are Findings 15, 19 and 21), and the README has the current summary. Every experiment from Nov 19 on
starts with a pre-registration paragraph that I wrote BEFORE running it, followed by a Finding with the result,
negative results included.

**Codes used below**
- **Finding N**: a result. Findings 1-30 cover the first phase (synthetic photos, the first real-photo runs, the
  app); 31-45 cover the Nov 19 phase (frozen dev/test split, official corpus, baselines, re-ranker).
- **A1**: two-model ensembles. **B1**: conformal shortlist on synthetic photos. **C1 / C2**: the old menu-filter
  baselines (shop order random / menu plus our ranking).
- **D8**: data-quality checks on LookBench (duplicates, near-duplicates, manual check). **D9**: own-wardrobe test,
  with the rules fixed in advance (not run yet). **D10**: per-category breakdown and failure gallery.
- **E1-E8**: the app (engine, calibration, badges, API, page, Docker, latency, README).
- **T1-T6**: analysis questions: conformal under the synthetic-to-real shift (T1), why hard batches hurt (T2),
  per-category and grouped conformal (T3), risk control / "not sure" (T4), hubness (T5), the coverage bound (T6).
- **Tasks 1-6** (the Nov 19 plan): 1 demo fixes (F35-36), 2 the app's real setting (F37-39), 3 bad photos (F40),
  4 credible baselines (F41-42), 5 learned re-ranker (F43-44), 6 production and documentation.

**Protocols**
- *Main protocol (Finding 38 onward)*: LookBench real_streetlook, 981 real street photos searched against the
  OFFICIAL corpus of 61,553 images (3,278 gallery + 58,275 noise), with identical image files merged ("dedup").
  Exact-item metrics: R@1, R@5.
- *Gallery protocol*: the same photos against the 3,278 gallery only. I need it for group P@5 (noise images have
  no category) and for anything that uses item text or labels (keyword search, menus). Earlier findings use it.
- *Dev / test split*: 500 / 481 photos in 621 clusters (the same crop, street photo or item always stays together),
  seed 490. Methods are tuned on dev and run ONCE on test, with a paired cluster bootstrap and Holm across the
  methods taken to test. The split was frozen AFTER the model (finetuned_random) had been chosen on all 981
  photos (Findings 5-7), so the test half is clean for everything decided later but not for that choice; the
  fine-tuned vs pretrained comparison on it is slightly optimistic. D9 (own photos) is the untouched check.
- *Synthetic photos*: catalogue images degraded with the training augmentation. They mostly measure that
  augmentation (Finding 5), so I treat them as a secondary result.

**Current headline numbers and where they come from (test half, 481 real photos, LookBench crop)**
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

The raw numbers from the first phase are in `results/summary.csv` (with 95% CIs) and `results/comparisons.csv`.

---

## Setup (as of Nov 10)

- Catalogue: 6,197 Kaggle Fashion Product Images across 14 article types, with near-duplicates removed (dHash plus
  a coarse mean colour).
- Images: I swapped in the high-res versions of the same items from the full Kaggle dataset (max side 512 px).
- Queries: 294 items (21 per type). Each one appears as a clean image and as a synthetic phone photo at three
  severities (mild / medium / strong). All severities of an item share the same random draws, only scaled, so the
  robustness curve compares like with like.
- Metrics: exact-item Recall@1 / Recall@5 on the phone queries, and Precision@5 for the same article type (with the
  source item removed). 95% CIs come from a bootstrap (2,000 resamples of the queries), and method comparisons use
  a paired bootstrap on the same queries.
- I don't report clean-query Recall: it's always 1.0 by construction, because the query is the catalogue image itself.

## Finding 1: resolution matters a lot

Same models and same queries, only 60x80 px images versus high-res (phone_medium, Recall@1):

| method | low-res (60x80) | high-res |
|---|---|---|
| CLIP | 2.4% | 24.8% |
| FashionSigLIP | 6.1% | 79.3% |
| FashionCLIP | 15.3% | 79.3% |

The low-res runs used a single "phone" severity that's roughly the same as medium. Results are in `results/lowres/`.

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

Precision@5 (same article type) on phone_medium: histogram 0.08, CLIP 0.58, FashionSigLIP 0.89, ensemble_mv 0.88
(chance is 0.07).

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

- By far the biggest single gain comes from fashion-specific pretraining: going from CLIP to a fashion model adds
  40 to 54 points.
- On our data it doesn't matter which fashion model I use, because SigLIP and FashionCLIP tie. I picked SigLIP as
  the base for fine-tuning since it's slightly ahead on strong photos and on Precision@5.
- Combining the two models helps by about 5 points, which fits the idea that they make different mistakes. The
  price is twice the search cost.
- Multi-view catalogue embeddings give a small but consistent gain, and they cost nothing at search time.
- Garment cropping (rembg) hurts on synthetic phone photos, and cropping the catalogue as well doesn't fix it, so
  the problem isn't a framing mismatch between query and catalogue. My guess is that the segmentation itself fails
  on our synthetic backgrounds (TODO: check data/crop_examples/). Synthetic photos have no person or room in them,
  and removing those is the whole point of cropping, so this test can't show what cropping is meant to do. That
  needs real photos.
- Mild photos are nearly solved (95-97%). Medium and strong photos are where the methods pull apart.
- Cropping adds ~230 ms per query, about 12x the embedding time.

## Caveats to state in the report

- So far there are only synthetic phone photos; the real-photo evaluation (own photos, LookBench) is still to come.
- 294 queries and 6,197 catalogue images is small, but every result comes with a CI.
- Fine-tuning will train on the same kind of synthetic augmentation I use for testing, so part of any gain may just
  be the model learning the augmentation. Real photos are the check for that.

## Pending

- [ ] Fine-tuning (training/finetune.py) -> row 7, then ensemble_ft_mv
- [ ] ~30 real photos of my own clothes (condition "real", Precision@5 only)
- [ ] LookBench real_streetlook (the download failed on a slow connection; retry on a faster network with HF_HUB_DISABLE_XET=1)
- [ ] Look at the crop failures (python -m retrieval.preprocess)
- [ ] Per-category breakdown, failure gallery, manual check of ~30 queries
- [ ] FastAPI + Streamlit app, Docker

## Update Nov 11: proposal alignment

The approved proposal says: the users are online shoppers, the main metric is Precision@5 (same type), and the
constraints are robustness to imperfect user photos and response time. I'm keeping that framing. Exact-item Recall
is a harder question that we added on top.

## Finding 4: fine-tuning (row 7)

Training: 5 epochs, the last 4 of 12 blocks, same-type hard batches, lr 1e-5, temperature 0.05, batch 64.
Validation Recall@1 went 0.807 (pretrained) -> 0.957 -> 0.973 -> 0.977 (best, epoch 3) -> 0.977 -> 0.977.

| | mild | medium | strong |
|---|---|---|---|
| Recall@1 FashionSigLIP | 94.9 | 79.3 | 51.0 |
| Recall@1 fine-tuned | 99.0 | 96.9 [95, 99] | 84.7 [80, 89] |
| paired difference | +4.1* | +17.7* | +33.7* (all p<0.0005) |
| P@5 FashionSigLIP | 0.897 | 0.886 | 0.773 |
| P@5 fine-tuned | 0.795 | 0.798 | 0.744 |
| paired difference | -10.3* | -8.8* | -2.9* (p=0.047, borderline) |

So fine-tuning wins on exact-item Recall but loses on the proposal's main metric. My hypothesis is that same-type
hard batches push each item away from its same-type neighbours. To test it I'll run a --hard_batches 0 ablation, and
also try ensemble_ft_mv.

## Finding 5: real photos (LookBench real_streetlook, 981 queries, 3,278 gallery, query = LookBench crop)

| | Recall@1 | Recall@5 | category P@5 |
|---|---|---|---|
| FashionSigLIP | 42.8 [39.8, 45.8] | 64.1 | 0.714 |
| fine-tuned | 45.5 [42.6, 48.4] | 67.3 | 0.691 |
| paired | +2.7 [+0.5, +4.8] p=0.014 | | -2.3 p<0.0005 |

- Most of the synthetic fine-tuning gain (+17.7) doesn't carry over to real photos (+2.7). The model largely
  learned our augmentation.
- Synthetic photos are easier than real ones (79% vs 43%).
- The LookBench paper (arXiv 2601.14706) reports FashionSigLIP fine R@1 of 55.2% and coarse R@1 of 73.4% on a
  61,553-image corpus, but with a different correctness rule (category plus all attributes). That isn't comparable
  to our item_ID metric. Our category P@5 (71.4%) is roughly in line with their coarse 73.4%.


## Finding 6: does adding the pretrained model fix the fine-tuning trade-off? (A1)

| | synth R@1 med | synth P@5 med | real R@1 | real cat P@5 | ms |
|---|---|---|---|---|---|
| ensemble_mv (pretrained) | 85.4 | 0.882 | 44.0 [40.9, 47.1] | 0.720 | ~28 |
| finetuned | 96.9 | 0.798 | 45.5 | 0.691 | ~14 |
| finetuned_mv | 96.9 | 0.808 | 45.6 [42.6, 48.7] | 0.700 | ~14 |
| ensemble_ft_mv | 97.6 | 0.850 | 47.1 [44.1, 50.2] | 0.706 | ~28 |

Paired tests of ensemble_ft_mv against finetuned: R@1 isn't different anywhere (real +1.6, p=0.146), and P@5 is
better everywhere (synthetic +3.8 to +6.5, real +1.5, p=0.001). So ensemble_ft_mv dominates finetuned.
Against ensemble_mv, ensemble_ft_mv has better R@1 (synthetic +3 to +25, real +3.1, p=0.002) but worse P@5
(synthetic -3.1/-3.5 with strong tied; real -1.4, p=0.003). The trade-off is still there, but it's small on real photos.
Decision rule: for find-similar (P@5) use ensemble_mv, which needs no training; for find-the-exact-item use
ensemble_ft_mv. Both take ~28 ms.
Every significant real-photo result has p <= 0.003, so they all survive a Holm correction.

## Finding 7: the P@5 drop is caused by the same-type hard batches (ablation)

I re-ran fine-tuning with random batches (same seed, split and 5 epochs; only the batching changed). The best val
R@1 was 0.963, against 0.977 with hard batches.
Against the hard-batch model: real R@1 +0.1 (p=0.95), real cat P@5 +3.5 (p<0.0005); synthetic P@5 +7.0/+6.9/+7.8
and synthetic R@1 -1.4/-2.0/-3.4 (all significant). In other words, hard batches only buy synthetic Recall.
Against FashionSigLIP (same architecture, 14 ms): real R@1 +2.8 (p=0.01) and real cat P@5 +1.2 (p=0.003), so it's
better on both.
Against ensemble_mv (28 ms): real R@1 +1.5 (p=0.16) and real P@5 +0.6 (p=0.26), so there's no significant
difference, at half the latency.
Synthetic P@5 against the pretrained model: -2 to -3 on mild/medium (medium p=0.042, borderline), +5 on strong.
New decision rule: finetuned_random (14 ms) for both find-similar and find-exact, with ensemble_mv as the option
that needs no training.

## Finding 8: mechanism (T2) - hard batches push same-type items apart

To see why, I looked at the catalogue embeddings (6,197 items) with cosine similarity. gap_std is (mean within-type
similarity - mean between-type similarity) divided by the std over all pairs. nn_same_type is the share of each
item's 5 nearest catalogue neighbours that have the same type.

| model | within | between | gap_std | nn_same_type |
|---|---|---|---|---|
| FashionSigLIP | 0.690 | 0.531 | 1.68 | 0.905 |
| hard batches | 0.250 | 0.127 | 1.20 | 0.836 |
| random batches | 0.373 | 0.179 | 1.78 | 0.893 |

Hard batches cut type separation by 29%, while random batches keep it. That's the same ordering as synthetic P@5.
One thing to watch: fine-tuning lowers all the similarities (the embeddings spread out), so raw gaps can't be
compared across models, which is why I use gap_std.
Caveat: the catalogue includes training items, and this is descriptive, with no CIs.

## Finding 9: conformal on real models (B1, synthetic photos, alpha = 0.1, 200 splits)

In distribution, coverage is >= 0.90 for every model and severity, so the guarantee holds.
Relative variant, mean set size (all severities / strong): SigLIP 8.6 / 102, ensemble_mv 3.9 / 54, finetuned_random
1.0 / 1.8. For finetuned_random, 72% of strong queries get a single answer, and 95% of those are correct. Keep in
mind that synthetic photos favour fine-tuning.
The threshold variant gives exact coverage (0.901-0.907, inside the T6 bound of 0.907) but huge sets on strong
photos (912 items).
The relative variant over-covers (0.95-0.98). Most calibration scores are tied at 0 (top-1 correct), which breaks
the no-ties assumption behind the upper bound, so coverage ends up equal to R@1. The fix is randomized tie-breaking.
Under a shift from mild to strong, coverage collapses (relative 0.51 / 0.58 / 0.81; threshold 0.17 / 0.10 / 0.61),
and the threshold sets are often empty. The guarantee needs calibration data that looks like the deployment data,
which is what T1 is about.

## Finding 10: conformal under real-photo shift (T1), LookBench, alpha = 0.1, 200 paired splits

Calibrating on synthetic photos and testing on real ones gives coverage (threshold / relative) of SigLIP 0.84 / 0.67,
ensemble_mv 0.84 / 0.64 and finetuned_random 0.73 / 0.52. For finetuned_random the relative mean set is 1.2, so it
gives confident single answers that are right half the time.
Calibrating on real and testing on real brings coverage back to 0.899-0.903. Relative median set sizes are SigLIP
164, ensemble_mv 143, finetuned_random 59 (out of 3,278); the means are 310 / 288 / 151 (skewed). Threshold sets are
2-4x larger.
The T1 bound, coverage >= 0.9 - TV(synthetic scores, real scores), holds in 14/18 cases (model x variant x bins).
The misses are small and happen at coarse bins: binning underestimates TV (data processing inequality), so the
plug-in bound is optimistic. TV is largest for the fine-tuned model (0.31-0.37 vs 0.23-0.29), which fits with it
having learned our augmentation.
Takeaway: calibrate on real photos. On real photos a 90% guarantee needs large sets, so the app should flag "unsure".

## Finding 11: per-category conformal (T3) and risk control (T4), LookBench real -> real

T3 (alpha 0.1, relative): standard conformal hides some categories at 0.52-0.82 coverage (jumpsuit, denim skirt,
cape, pleated skirt, cardigan, coat) while the easy ones sit at 1.00. With Mondrian calibration, every category
reaches >= 0.90.
The cost is that set sizes get redistributed. Easy categories get smaller sets (finetuned_random: shirt 65 -> 10,
bag 63 -> 10) and hard ones bigger sets (coat 84 -> 588). Categories with fewer than ~18 queries (< 9 calibration)
get the whole gallery (3,278), because the method needs ceil((n+1)(1-alpha)) <= n, i.e. n >= 9. The fix is to merge
rare categories into coarser groups.
T4 (answer only if the top similarity is >= lambda; loss = share of the shown top 5 in the wrong category, 0 when we
abstain): the measured loss is <= alpha at every level (0.198-0.200 / 0.096-0.097 / 0.049). Answer rates at alpha 0.2
are SigLIP 80%, ensemble 81%, finetuned_random 85%; at 0.1 they're 41 / 46 / 46%. P@5 on the answered photos only
goes from 0.73 to ~0.79, so the guarantee is met mostly by abstaining. Top similarity is a weak confidence signal.

## Finding 12: grouped Mondrian conformal (T3 follow-up, evaluation/conformal_extra.py, "group" variant)
Before looking at any grouped result I fixed 7 category groups: tops 296, outerwear 216, bottoms 229, dresses 40,
footwear 55, bags 49, accessories 96 queries (sock goes to accessories, cardigan to tops). The relative threshold is
calibrated within each group; real -> real, alpha = 0.1, 200 random 50/50 splits.
With one shared threshold, standard conformal under-covers the same three groups for all three models: outerwear
0.86-0.88, bottoms 0.85-0.87, dresses 0.83-0.86. Grouped calibration brings every group to 0.90-0.93 for every model
(bottoms 0.899-0.902) and reduces the over-coverage of easy groups (bags 1.00 -> 0.92).
Set sizes (finetuned_random): small categories no longer get the whole gallery. Going from per-category Mondrian
(3278) to grouped: belt 177, tie 3, glasses 2, clutch bag 4, sneaker 3.
The cost is that the guarantee is per group, not per category, so hard categories inside an easier group get hidden
again (finetuned_random: sock 0.80, cardigan 0.77, jumpsuit 0.74, denim skirt 0.70). Some large categories in a hard
group also get bigger sets than with standard conformal (coat 84 -> 221).
The standard and per-category numbers and T4 didn't change (same seed and splits; I checked, max difference 0.0).
Takeaway: grouping is the practical middle ground, with a guarantee for every group and sets that are still usable.
Per-category guarantees need >= 9 calibration queries per category, and the smallest categories don't have that many.

## Finding 13: menu-filter baseline, "how shoppers search today" (C1, evaluation/filter_baseline.py)
This models a shopper who filters by garment type (and by colour on synthetic data; LookBench has no colour labels).
The shop then shows the matches in an order that has nothing to do with the photo, which I model as uniformly random.
These are exact formulas, not a simulation: with r pictures of the item among m shown, P(hit in top k) =
1 - C(m-r,k)/C(m,k), and the expected position of the first picture is (m+1)/(r+1). Precision@5 (same type) is 1 by
construction, so the comparison is on exact-item recall and scroll depth.
Synthetic (294 test items, 6,197 catalogue): the type filter shows a median of 490 items, R@1 0.002, R@5 0.012,
median scroll 246. Type + colour shows a median of 56, R@1 0.045, R@5 0.183, median scroll 28.5.
LookBench real (981 queries, 3,278 gallery): the type filter shows a median of 154, R@1 0.026, R@5 0.113, median
scroll 48. 49 queries (5%) get filtered out because their gallery pictures carry a different category label. I count
those as misses; they also cap any model's category P@5 on LookBench.
Comparison: finetuned_random gets R@1 0.949 (synthetic medium) and 0.456 (real), against 0.045 / 0.026 for the best
menu filter, so about 20x / 17x better. This is generous to the menus, because it assumes the shopper can name the
type and colour, and the user story says they can't.

## Finding 14: menu filter + visual re-ranking (C2, evaluation/filter_rerank.py, finetuned_random)
Filter by type (plus colour, synthetic only), then sort whatever survives by model similarity.
With filter "none" this reproduces run_eval (synthetic medium R@1 0.949) and the dedup LookBench protocol (0.520).
Synthetic R@1 for none / type / type+colour: medium 0.949 / 0.949 / 0.963, strong 0.813 / 0.823 / 0.874, and clean
1.0 everywhere. The model almost never confuses garment types, so a type filter adds nothing, while colour helps on
the hardest photos.
LookBench real: R@1 goes from 0.520 (none) to 0.554 (type), R@5 from 0.677 to 0.704, and the item's mean rank from
85.7 to 16.9 (filtering removes the catastrophic misses). But 49 items (5%) are lost to inconsistent category labels.
In theory, if the item survives the filter its rank can only get better, since filtering only removes competitors.
Takeaway: an optional type filter is a useful app feature for real photos, but it's the model, not the menu, that
does the work of finding the exact item.

## Finding 15: LookBench gallery duplicates; dedup protocol (correction, evaluation/run_lookbench.py)
The real_streetlook gallery has 3,278 entries but only 2,537 distinct image files. 633 files (outfit photos) are
listed 2-5 times under different item_IDs, 94 of them under different categories, and 515 of the 981 queries have an
item with a shared picture. The official protocol keeps whichever copy the sort puts first, so retrieving the right
picture can still count as a miss. My new "dedup" protocol merges the copies (by md5) before taking the top 5, and a
picture counts as correct if any copy has the query's item_ID / category. I save both, and the official numbers
reproduce exactly.
Dedup raises R@1 by +5.8 to +7.0 points for every model (finetuned_random 0.456 -> 0.520, fashion_siglip 0.428 ->
0.486, ensemble_mv 0.440 -> 0.505, ensemble_ft_mv 0.471 -> 0.539), raises R@5 by ~2, and leaves category P@5
essentially unchanged. The model ranking stays the same, and all the Finding 7 conclusions hold (paired, dedup):
finetuned -> finetuned_random P@5 +0.036 (p<0.0005), R@1 -0.005 (p=0.31); fashion_siglip -> finetuned_random R@1
+0.034 (p<0.0005), P@5 +0.011 (p=0.009); ensemble_mv -> finetuned_random n.s. (R@1 p=0.09, P@5 p=0.40).
C1 on LookBench, re-counted in distinct pictures (this replaces the LookBench line of Finding 13): the type filter
shows a median of 119 pictures (was 154 entries), R@1 0.031 (was 0.026), R@5 0.131 (was 0.113), median scroll 36.7
(was 48), with 49 queries still filtered out. The conclusion doesn't change (model R@1 0.520 dedup, ~17x).
For C2 on LookBench, R@1 counts ties in the item's favour, so it equals the dedup protocol. R@5 and the ranks still
count duplicate entries of OTHER pictures, which makes them slightly pessimistic.
Not re-run: conformal set sizes and risk control count gallery entries, not pictures. That inflates set sizes by at
most 3278/2537 = 1.29x and doesn't affect coverage, so I state it as a limitation.

## Finding 16: per-category breakdown (D10a, evaluation/per_category.py, finetuned_random)
Sanity check: the query-weighted per-category means reproduce every saved overall number (4 synthetic conditions
plus LookBench dedup R@1 0.520 / P@5 0.725). Saved in results/per_category.csv (97 rows).
LookBench real (dedup), using the 7 groups fixed for F12, R@1 / category P@5 (95% CI):
dresses 0.65 / 0.32 [0.24, 0.41] (n=40); bags 0.63 / 0.52 (n=49); accessories 0.52 / 0.66 (n=96);
tops 0.57 / 0.72 (n=296); footwear 0.62 / 0.73 (n=55); outerwear 0.38 [0.32, 0.44] / 0.80 (n=216);
bottoms 0.52 / 0.80 (n=229).
The low P@5 for dresses and bags is a base-rate effect, not a failure to find the item. Dresses are 47 of the 2,537
distinct gallery pictures (1.9%; chance P@5 ~0.02, model ~17x) and bags are 169. The exact item is often found
(R@1 0.65), but its neighbours are mostly other categories (tops and skirts for dresses), and P@5 also needs the exact
category (dress vs jumpsuit counts as wrong). Outerwear is the opposite case: the type is almost always right, but the
exact item is the hardest to find (601 near-identical coat/jacket pictures).
Synthetic strong photos (n=21 per type): the lowest P@5 is Jackets 0.69, Sports Shoes 0.72 and Tops 0.74; the lowest
R@1 is Trousers 0.48 (P@5 still 0.86: right type, wrong pair) and Track Pants 0.62.
Caveats: descriptive only (no correction across 7 groups / 14 types; n=21 gives CIs of about +-0.15). Category P@5
depends on how common the category is in the gallery, which I should state as a limitation of the metric.
Takeaway: the "unreliable match" warning matters most for rare categories (dresses, bags) and for exact-item claims on
outerwear. This feeds into D10b (failure gallery) and F (per-category fairness).

## Finding 17: failure gallery (D10b, evaluation/failure_gallery.py, fig10_failures_*.png)
run_lookbench now also saves each query's top-5 distinct pictures (results/rankings/). Re-running finetuned_random
reproduced every saved number exactly, and hit@1 / hit@5 / P@5 recomputed from the rankings match the per-query files
for all 981 queries.
The gallery's sanity check turned up a correction: our synthetic P@5 removes the exact item (top-6 minus the source,
i.e. "are the alternatives sensible?"), while LookBench's category P@5 (from LookBench's code) counts it. Model-vs-model
comparisons within one dataset aren't affected, but synthetic and real P@5 can't be compared directly. The report's
metrics and LookBench sections now say so.
The gallery has 6 random rows (seed 0) per case and is qualitative only:
- Real dresses with P@5 <= 0.4 (31 queries): in 3/6 rows the "dress" crop actually shows a top and/or shorts (a
  label/crop mismatch), and the model returns visually matching tops and shorts with high scores (up to 0.86). In the
  other 3, the exact item is at rank 1 and the neighbours share its colour or fabric (black skirts, satin).
- Real outerwear where the top-1 isn't the exact item (134): every result is a near-identical coat or jacket
  (fine-grained), with scores of 0.74-0.89. One row shows what look like near-duplicate gallery pictures that md5
  didn't merge, and a cape row marks wrap coats as "wrong" (exact-category P@5).
- Synthetic Trousers, strong photos (11): every result has the right type, and the exact pair is often at rank 2-3.
  Some black trousers are labelled Track Pants/Jeans in the catalogue (label ambiguity).
What I take from this: (1) most misses are fine-grained or caused by ambiguous labels, which supports showing a top-5
shortlist rather than a single answer; (2) confident misses (high top similarity) explain why risk control on top
similarity only lifts answered P@5 from 0.73 to 0.79 (F11); (3) D8 should measure the dress-label noise and the
near-duplicates.

## Finding 18: near-duplicate gallery photos (D8a/D8c, evaluation/near_duplicates.py, near_dedup.py)
Besides the exact duplicates (F15), the gallery contains the same photo saved as different files (re-encoded, resized,
cropped). The candidates are pairs of distinct pictures with PRETRAINED FashionSigLIP cosine >= 0.90 (2,305 pairs). I
used the pretrained model on purpose, so the model being evaluated doesn't get to define the duplicates in its own test
set. On a contact sheet with 4 random pairs per bin, 12/12 pairs with cosine >= 0.97 were the same photo, pairs in
[0.95, 0.97) were mixed, and below that they were mostly different photos. dHash misses crops (a cropped copy had a
Hamming distance of 17), so the rule uses cosine only.
I fixed the rule BEFORE recomputing any metric: cosine >= 0.97, with 0.95 and 0.98 as sensitivity checks.
Near-duplicates are joined into connected components (union-find), which turns 2,537 pictures into 2,406 groups (2,441
at 0.98; 2,303 at 0.95). The same photo sometimes carries different categories (a leopard heel is listed both as
"pants, shoes" and as "sock").
The near-dedup protocol is the dedup protocol with "same file" replaced by "same photo". I computed it from the saved
top-20 rankings (every query had >= 5 groups), and with no merging it reproduces all the saved dedup per-query results.
R@1 from dedup to near-dedup at 0.97: FashionSigLIP 0.486 -> 0.495, ensemble_mv 0.505 -> 0.516, finetuned 0.525 ->
0.535, finetuned_random 0.520 -> 0.532. That's about +1 point for every model (+1.6 to +2.1 at 0.95). Category P@5
moves by at most 0.004.
Paired tests (dedup / 0.98 / 0.97 / 0.95): fashion_siglip -> finetuned_random R@1 +0.034/+0.036/+0.037/+0.040 (all
p<0.0005); P@5 +0.011 (p=0.009) / +0.011 (0.004) / +0.012 (0.002) / +0.008 (p=0.083, n.s.). finetuned ->
finetuned_random P@5 +0.036 to +0.038 (p<0.0005), R@1 n.s. ensemble_mv -> finetuned_random n.s. everywhere (smallest
p 0.058).
Conclusion: dedup stays the main protocol, and near-dedup goes into the report as a sensitivity check. All the
conclusions hold, but "better P@5 than the pretrained base" is small (+0.01) and not significant at the loosest
threshold, so that's how I'll describe it. There are a lot of tests here and no correction, so results around
p ~ 0.01 are suggestive, not strong.

## Finding 19: LookBench category names are too strict; group P@5 (evaluation/group_p5.py) (correction)
The manual check (D8b) showed "wrong category" results that were really the same kind of garment. The reason is that
category P@5 compares category STRINGS, and LookBench's names overlap (bag/handbag/clutch bag, skirt/pleated/denim/silk
skirt, shirt/t-shirt, shoe/sneaker). On top of that, the gallery has 38 names on 109 of the 3,278 entries that never
occur as a query category (boots, jeans, sneaker variants, "trench coat", "tights" ...), and these always count as wrong.
Group P@5 counts a result as correct if it falls in the query's category group. It uses the 7 groups fixed for F12
before any grouped result, plus an explicit mapping of the 38 gallery-only names that I wrote before computing anything
("product label" never counts). With exact names, the script reproduces every saved category P@5.
Exact -> group P@5: FashionSigLIP 0.714 -> 0.869, ensemble_mv 0.721 -> 0.864, finetuned 0.688 -> 0.843,
finetuned_random 0.725 -> 0.869 [0.853, 0.885].
Paired (group P@5): finetuned -> finetuned_random +0.026 [+0.020, +0.032] p<0.0005, so the hard-batch drop survives.
fashion_siglip -> finetuned_random -0.001 [-0.008, +0.006] p=0.85, so the +0.011 gain with exact names VANISHES.
ensemble_mv -> finetuned_random +0.004 p=0.27, so the tie holds.
By group (finetuned_random, exact -> group): bags 0.52 -> 0.85, footwear 0.73 -> 0.90, tops 0.72 -> 0.89, outerwear
0.80 -> 0.93, bottoms 0.80 -> 0.95, accessories 0.66 -> 0.72, dresses 0.32 -> 0.32.
Corrections: (1) The F7/F15 claim "finetuned_random beats its pretrained base on both real metrics" becomes "better
real R@1 (+3.4, p<0.0005), tied on garment type". The P@5 gain came from arbitrary name distinctions (and it was
already fragile in F18). (2) In F16, the low bag P@5 was mostly a naming issue (bag vs handbag), not base rate. The low
dress P@5 is real: the neighbours are tops and skirts, which belong to another group, and there's label noise on top of
that (F17).
Report: a new section "Category names and group P@5", a new row in the ablation table, and the bullets and summary
rewritten.
Limitation: the groups are lenient (a cardigan counts for a blouse), so exact and group P@5 bracket the truth.

## Finding 20: manual check of 61 LookBench queries (D8b, evaluation/manual_check.py)
I fixed the sample before looking: all 31 dress/jumpsuit queries with P@5 <= 0.4 (finetuned_random, dedup) plus 30
random other queries (seed 1). Each query gets a contact sheet with the full photo, the LookBench crop and the top 5
with borders.
Each query gets two verdicts: label_check (does the crop show the labelled garment: correct / wrong / ambiguous) and
metric_fair (would a shopper agree with the exact-category scoring). Claude annotated them first, then I went through
them row by row and agreed with all 61. That makes it a review, not an independent annotation, so anchoring is possible.
Saved in results/manual_check.csv.
Random sample (this part generalises; 95% Wilson intervals): label wrong 1/30 = 3% [1, 17]; wrong or ambiguous
6/30 = 20% [10, 37]; metric unfair 9/30 = 30% [17, 48]. All 9 unfair rows are same-group results that exact names
scored as wrong (coat/jacket, bag/handbag, shirt/t-shirt, sweater/cardigan), which independently confirms Finding 19.
Dress failures (a selected sample, so it doesn't generalise): label wrong 7/31 = 23% [11, 40]; wrong or ambiguous
10/31; metric unfair 15/31 = 48% [32, 65]. These are mostly label errors (the crop shows a top or shorts) or dresses
listed under other categories. Group P@5 can't fix these, which fits with dresses staying at 0.32 -> 0.32 in F19.
I also found a new data issue: identical query crops. Only 700 of the 981 query crops are distinct. 503 queries fall
into 222 groups of identical crops, and each copy asks for a DIFFERENT item (the query-side twin of F15). So the
queries aren't independent, and bootstrap CIs and p-values that resample queries are probably too narrow. Next step:
a cluster bootstrap that resamples crops.

## Finding 21: cluster bootstrap over identical query crops (evaluation/cluster_bootstrap.py) (correction)
The 981 LookBench queries use only 700 distinct crops (F20), so the queries aren't independent. The new paired test
resamples whole crops (cluster bootstrap, ratio estimator sum(S_c)/sum(n_c)). With one query per cluster it reproduces
every earlier paired test exactly. The CIs get wider by x1.06-1.36 (design effect ~1.1-1.9).
finetuned_random minus ... (naive -> cluster):
- FashionSigLIP R@1 +0.034, p<0.0005 -> [+0.015, +0.052] p=0.001 (survives Holm over 4 tests: 0.001 < 0.0125)
- FashionSigLIP exact P@5 +0.011, p=0.009 -> [+0.000, +0.021] p=0.047 (fails Holm: 0.047 > 0.0167)
- FashionSigLIP group P@5 -0.001, n.s. both ways
- finetuned (hard) exact P@5 +0.036 and group P@5 +0.026: p<0.0005 both ways; R@1 n.s. both ways
- ensemble_mv: everything n.s. (cluster p >= 0.13)
This corrects the report's ablation caption, which said both FashionSigLIP results survive Holm.
The real-photo claims that hold up: (1) better exact-item R@1 than the pretrained base, (2) random batches win back
the precision lost to hard batches, (3) a tie with the pretrained ensemble at half the latency. Dropped: "better
garment-type precision than the base" (borderline with exact names, absent at group level).
Note: the naive group-P@5 p here is 0.898, against 0.852 in group_p5. The query order is different, so the resamples
are different; it's Monte Carlo noise around the same estimate.

## D9 protocol (fixed BEFORE taking the photos, Nov 18)
This is the final untouched test: ~30 phone photos of my own clothes, used ONCE.
The model choice is frozen: finetuned_random (recommended). color_hist, fashion_siglip and ensemble_mv are also
reported but not used for choosing. Metric: Precision@5 = same catalogue article type (my clothes aren't in the
catalogue, so there's no Recall). Primary result: finetuned_random P@5 with a 95% bootstrap CI. The comparisons are
descriptive (n~30 has little power). No re-shooting and no dropping photos after seeing results; any photo that fails
to load gets reported.
Amendment (Nov 19, 00:55, BEFORE any photo was taken). Reason: the plan changed from ~30 photos to the whole wardrobe,
and the group bonus of Finding 32 became the headline method.
- Photos: every garment in the wardrobe, one photo each, taken in one session on the phone camera with no editing.
  Every photo is used; none is dropped or re-shot after seeing results, and a photo that fails to load gets reported.
- Each photo is labelled BEFORE anything runs: its garment group (tops, outerwear, bottoms, dresses, footwear, bags,
  accessories) and, if there is one, its evaluation-catalogue article type (one of the 14).
- Primary results (frozen methods, run once):
  (1) group P@5 of finetuned_random + group bonus w = 0.05 against the APP catalogue (the app as shipped), with a 95%
      bootstrap CI. This is the D9 check of the 0.905 headline.
  (2) article-type P@5 of finetuned_random against the EVALUATION catalogue, only for photos whose type is one of the
      14 (the original protocol).
- Also reported, descriptively: color_hist, fashion_siglip and ensemble_mv on (2); the app's "not sure" rate,
  group-badge accuracy, out-of-catalogue warnings and shortlist sizes.
Clarification (Nov 19, 13:35, BEFORE any photo was taken; the script evaluation/d9_own_photos.py is written but not run):
- Crop box: not used. The crop box is optional in the app, and by default the app sends the photo unchanged, so "the
  app as shipped" means the whole photo. That's harder than LookBench, whose queries are hand-drawn crops. I take the
  photos with the garment filling most of the frame (on a hanger, on a plain surface, or worn).
- Group "other": a garment that fits none of the 7 groups (e.g. a watch) is labelled "other". It's listed and counted
  for out-of-catalogue warnings, but it isn't in either primary result.
- The photos go in data/real_photos/ (git-ignored) and the labels in data/real_photos/labels.csv (file, group,
  eval_type). I write the labels by hand and check them with --check (no model) BEFORE --run, and --run refuses to run
  twice.

## Finding 22: hubness on LookBench (T5, evaluation/hubness.py)
The k-occurrence N_k(x) is the number of distinct query crops (700; identical crops counted once, F20) whose top-k
contains gallery picture x (2,537 distinct pictures). Check: the sum of N_k equals crops x k for every model and k.
Skewness of N_5 (a random-list baseline Bin(700, 5/2537) gives 0.85): FashionSigLIP 2.20, ensemble_mv 2.31, finetuned
2.34, finetuned_random 2.32 (N_10: 2.00-2.29). 35-39% of pictures never appear in any top 5 (random: 25%). The top 1%
of pictures (26) fill 7.3-7.8% of all top-5 slots, and the max N_5 is 14-16.
So hubness is there but moderate, and fine-tuning barely changes it.
Do hubs show up in the failures? Share of hub slots that are wrong-group results vs other slots: finetuned_random 14.4%
vs 14.8% (no difference); FashionSigLIP 11.3% vs 14.8%; ensemble_mv 22.4% vs 14.6%; finetuned 21.5% vs 16.9%.
Descriptive only (hub slots aren't independent).
The biggest hubs (fig11_hubs.png) are plain, generic garments (white t-shirts, cream cardigans, a beige trench),
probably sitting near the centre of their type. Tops are 30% of the queries, so part of their popularity is legitimate.
Conclusion: hub reduction (e.g. CSLS, mutual proximity) isn't a priority for the recommended model.

## Finding 23: app calibration (E2, serving/calibrate.py -> serving/calibration.json)
Decision 1a: the app's uncertainty is calibrated on LookBench real photos (all 981; finetuned_random). Checks: the best
similarity equals the saved rank-1 score for every photo, and the exact-name loss of 0.2736 reproduces F11.
Shortlist (relative conformal, alpha 0.1): q = 0.228. Over 200 random 50/50 splits, coverage is 0.901 with a median
size of 59 (reproduces F10). That's too many to show in full, so the UI shows the count and the first few items.
"Not sure" (risk control on top similarity; loss = share of the top 5 in the wrong GROUP, F19): alpha 0.2 turns out to
be VACUOUS. The loss is already 0.130 with no abstention, so it answers 99.9% of photos. I therefore set alpha to 0.1.
I chose that after seeing the 0.2 result, so it's a product decision, not an accuracy claim; D9 is the independent check.
LookBench (200 splits):

| alpha | answer rate | loss | group P@5 |
|---|---|---|---|
| none | 100% | 0.130 | 0.870 |
| 0.2 | 99.9% | 0.128 | 0.872 |
| 0.1 | 91.1% | 0.098 | 0.893 |
| 0.05 | 58.3% | 0.047 | 0.920 |

lambda(0.1) = 0.671 (with exact names it would be 0.823).
How to read this: the guarantee is met mostly by abstaining more often, not by abstaining on the right photos
(precision when answering only goes 0.87 -> 0.89 -> 0.92). Top similarity is a weak confidence signal, consistent with
F11, the confident outerwear misses in F17, and F22.
Limitation: this is calibrated on LookBench's gallery. Our catalogue may give different similarity levels, so there's
no formal guarantee in the app. D9 measures the app's answer rate and its wrong-type rate when it does answer.

## Finding 24: match badges (E3, serving/badges.py -> serving/badges.json)
The badges describe measured differences, never the model's reasons. Checks: CIELAB(white) = (100, 0, 0), and
catalogue Black items are darker than White ones (median L 21 vs 76).
Colour badge: the median CIELAB of the image centre (near-white background left out). The CIEDE2000 bands come from
3,060 catalogue pairs with the same colour label: "very close" is <= 16.5 (the median) and "different" is > 37.0 (the
90th percentile). A true match (a query against its own item) gets the "different" badge for 3.7% of medium and 7.1%
of strong phone photos, which I accepted, so this one shipped.
Type badge: zero-shot PRETRAINED FashionSigLIP ("a photo of ...", 14 types). The prompt embeddings are saved once, so
the app doesn't need a tokenizer. Fine-type accuracy is 0.840 clean, 0.830 medium and 0.769 strong (chance 0.071).
But the "same type" badge agrees with the truth only 0.78-0.85 of the time, which is worse than the ranking's own P@5,
so I DROPPED the fine-type badge.
At group level (tops/outerwear/bottoms/dresses/footwear; group prob = sum of the type probs), accuracy is 0.969 /
0.973 / 0.942. Showing it only when group prob >= 0.9 gives 0.978 (94% shown) / 0.992 (89%) / 0.983 (82%). What
shipped is a group badge at confidence >= 0.9. I chose that threshold after seeing this table, on synthetic photos, so
it's a UI decision and not an accuracy claim; D9 checks it on real photos.
## Finding 25: out-of-catalogue warning (serving/outside_check.py -> serving/outside.json)
The catalogue has 14 types and no skirts, bags, hats, jumpsuits and so on. For photos of those, the results are bound
to be poor (manual tests: floral skirt and white skirt -> "not sure", shortlists of 252-688). The new check is
zero-shot (pretrained FashionSigLIP, 9 out-of-catalogue prompts, the same image embedding as the type badge, so the
group badge provably doesn't change): outside_prob = the total probability of the outside prompts in a 23-way softmax.
I fixed the labels before running. LookBench OUT = skirts, jumpsuit, bags, hat, scarf, belt, sock, glasses, glove, tie,
umbrella, headband (251 crops); IN = the categories we sell (606 crops); coat/cape excluded; plus 588 synthetic
in-catalogue photos. I also fixed the threshold rule before running (the smallest tau with false alarms <= 5% on both
IN sets), which gave tau = 0.4. It detects 91.6% of out-of-catalogue crops (skirts 93.9%, bags 100%, other 85.6%), with
false alarms of 3.8% on LookBench and 0.9% on synthetic photos. Detected skirts get named "skirt" 97.8% of the time.
The app shows: "This looks like a skirt. Our catalogue has no skirts, so the results are only loose suggestions."
Coverage is still a limitation (the proposal lists catalogue coverage), and a modern demo catalogue is planned.

## Finding 26: deployment (E6/E7: Dockerfile, docker-compose.yml, serving/latency.py)
The image is CPU-only (python 3.14-slim, torch 2.15 CPU), with the API and the Streamlit page as two services. The API
container is read-only (it can't write files).
Build note: copying the checkpoint into the image changed its modification time, so the index cache rebuilt the index
on the CPU once, which took 25 min. I fixed it by restoring the recorded time, so the index built on the Mac is reused
and the build reports "loaded saved catalog embeddings".
Latency on 100 synthetic medium photos after 5 warm-up (median / p95): Mac GPU (MPS) 36 / 37 ms; Docker CPU (4 vCPU)
266 / 282 ms server time (final image; the previous image gave 266 / 278). The round trip is the same within ~1 ms
(the VM and host clocks differ), so upload and JSON overhead are negligible. The top 5 equals the evaluated results for
105/105 photos.
About half of the time goes to the second model (type badge); making it optional would roughly halve latency.
A note for myself: my first measurement (36 ms) accidentally hit a local MPS server running on the same port. I caught
it with lsof and the container log, then re-measured.
## Finding 27: learned "not sure" score (evaluation/learned_confidence.py -> results/learned_confidence.csv)
Question: can a learned confidence score answer more photos than the top similarity at the same risk level?
There are seven signals per LookBench photo: top similarity, the gaps to the 2nd and 5th result, shortlist size, the
type and out-of-catalogue probabilities, and the colour difference to the top 5. The target is "the top 5 are all in
the right group".
The plan, fixed before running: 50 random splits by crop (40% train, 30% to calibrate lambda at alpha 0.1, 30% test),
with logistic regression and gradient-boosted trees (HistGradientBoostingClassifier, fixed settings, no tuning).
Means over the 50 splits (top similarity / logistic / trees):
- AUROC 0.651 / 0.758 / 0.719
- answer rate at alpha 0.1: 90.2% / 89.9% / 87.7%
- test loss 0.096 / 0.097 / 0.097 (all <= 0.1, so the guarantee holds)
- answer rate above top similarity in 24 / 50 splits (logistic) and 15 / 50 (trees)
Negative result: the learned scores rank photos better overall, but not on the worst ~10% that have to be skipped at
alpha 0.1. The app keeps the top similarity, which is simpler and just as good where it matters.

## Finding 28: bigger app catalogue (data_prep/app_catalog.py, serving/engine.py, serving/app_badges_check.py)
This is for the app only; every evaluation number stays on the 6,197-item catalogue. I fixed the selection rules
before building: the full Fashion Product Images zip, adult items, 72 wearable types (clothes, shoes, bags, worn
accessories; no watches, wallets, jewellery, underwear, nightwear, swimwear or personal care), all evaluation items
kept, and near-duplicates removed with the build_catalog rule. The result is 33,141 items of 68 types (501
near-duplicates removed, 5 without an image), e.g. heels 1,316, sandals 845, flip flops 885, handbags 1,756,
sunglasses 1,042, skirts 98. Everything is embedded with finetuned_random in a separate index (finetuned_random_app.npy).
Checks: the evaluation catalogue is unchanged (same top 5, score difference 0). In the app index, the 6,197 evaluation
items have exactly the saved embeddings (difference 0), and no best score went down.
Search time on MPS is 18 ms (it was 15 ms).
Badges with the app vocabulary (72 prompts, 7 groups, adding bags and accessories; groups fixed before measuring):
the evaluation badge still reproduces Finding 24 (shown 0.891, correct 0.992 on phone_medium). With the group badge at
confidence >= 0.9, LookBench real photos get the badge 64.5% of the time and it's correct 95.4% of the time
(accessories 86.4%, bags 100%, bottoms incl. skirts 96.1%, dresses 91.3%, footwear 100%, outerwear 93.5%, tops 97.0%).
On synthetic photos it's shown 70.2% of the time and correct 99.3%. More groups means the badge shows up less often,
with about the same accuracy.
Out-of-catalogue warning (for things still not sold: swimsuit, watch, necklace, earrings, wallet, umbrella): the same
rule as Finding 25 gives tau = 0.3. False alarms are 0.1% on LookBench (976 crops of kinds we sell) and 0.2% on
synthetic photos. Detection could only be tested on 5 umbrella crops (5/5), so in practice it's unmeasured.
Still open: the "not sure" lambda was calibrated against a smaller gallery. With 33k items the best scores can only go
up, so it may fire less often. To be checked on real photos (D9).

## Finding 29: type-aware ranking for the app (serving/type_rerank.py -> serving/rerank.json)
I found this problem with one of my own photos (my_photos/jeanstrial.jpg, a flat lay of long light-wash jeans): the
top 1 was kids' denim shorts (0.660), with adult jeans further down. Diagnosis (serving/diagnose_photo.py): it's NOT
the centre crop (the model sees the whole garment, and padding to a square changes nothing; my first guess was wrong),
and it isn't transparency either (plain RGB JPG). The cause is photo style. Every adult pair of jeans in the catalogue
is photographed on a person, and only kids' denim lies flat on white, so a flat photo matches flat kids' items.
Synthetic tests can't show this, because their queries are made from catalogue images in the same style.
Fix: score = similarity + w * P(item's type | photo), where P comes from the zero-shot type predictor (app vocabulary),
which ignores photo style. The similarity shown to the user, "not sure" and the shortlist all stay on the plain
similarity, and the evaluation catalogue is unchanged (w = 0 there). The plan, fixed before running: synthetic
medium/strong photos (588) against the 33,141-item app catalogue, w in {0, 0.02, 0.05, 0.1, 0.2}; keep the highest
mean type P@5 with an R@1 drop of <= 1 pt on both. Check: w = 0 reproduces the app's top 5.

| w | R@1 medium / strong | type P@5 medium / strong | note |
|---|---|---|---|
| 0 (before) | 0.929 / 0.769 | 0.845 / 0.815 | |
| 0.05 (chosen) | 0.922 / 0.762 | 0.871 / 0.828 | |
| 0.1 | 0.912 / 0.748 | 0.877 / 0.835 | R@1 drop 1.7 / 2.1 pts: not allowed |
| 0.2 | 0.891 / 0.731 | 0.861 / 0.805 | |

My own photos (I only looked at them; they didn't choose w): the jeans flat lay now has Jeans first (Jeans, Shorts,
Jeans, Jeans, Jeans). For the skirt photo, skirts now take ranks 1-4 (Trousers moved from 4 to 5). dayfittrial4 (jeans)
went from 2 to 3 jeans, and for the shoe photo a formal shoe was replaced by a casual one (5/5 casual). The others
didn't change. Still to do: measure it on real photos (LookBench, group P@5), which is the headline target.

## Finding 30: app clean-up (serving/api.py, data_prep/app_catalog.py; app catalogue only)
The app catalogue now keeps each item's `gender`. The app hides the 267 kids' items (Boys 123, Girls 144; all of them
came in with the evaluation catalogue), and it skips any result whose cosine with an already chosen result is >= 0.97
(the same product photographed again; the same girls' jeans had shown up 3x in one top 5). "Not sure" and the
shortlist size only use visible items. Checks: 33,141 rows, no missing gender, the index vectors have unit length (so
the dot product is a cosine), the evaluation catalogue is unaffected (nothing hidden, nothing skipped), and 6 tests pass.
Afterwards the jeans flat lay looks right (jeans first; I checked it in the app myself).
Wide-leg jeans photo (worn, white background; after the group-level bonus, Nov 19): the badge says "bottoms", the top 5
are all Jeans (Blue, Grey, Blue, Blue, Blue), the colour is "very close" for all 5, the similarities are 0.67 / 0.67 /
0.66 / 0.62 / 0.59, and the shortlist is 66. Right kind of garment and right wash, but the wrong CUT: all 5 are men's
straight or bootcut jeans. The cause is catalogue coverage, not the model. None of the 609 Jeans names in Fashion
Product Images (2011-12) contain wide, baggy, flare, loose, relaxed or palazzo (names are sparse, so this isn't proof),
while slim shows up 29 times and skinny 15. Similarity can only rank what the shop actually sells. I'll re-check this
photo after the H&M catalogue (2018-20, mostly women's) is merged in.

## Pre-registration (Nov 19, 00:00, BEFORE any of the experiments below)
Accuracy target (agreed definition): group P@5 on LookBench real photos (dedup protocol), with a goal of 0.90.
Secondary: exact-item R@1 on real photos (no 90% goal; any significant gain counts).
LookBench dev/test split (evaluation/lookbench_split.py -> data/lookbench/real_streetlook/split.csv): queries are
joined if they share a crop, a street photo or a target item (union-find, 621 clusters), then shuffled with seed 490
into dev (500 queries) and test (481). Nothing is shared between the two sides. Baseline finetuned_random:
  dev  R@1 0.528, group P@5 0.866      test R@1 0.511, group P@5 0.871     (all 981: 0.520 / 0.869, check OK)
Rules: every new method is tuned ONLY on dev, with its grid and choice rule written here before it runs. The default
choice rule is the highest dev R@1 such that dev group P@5 drops by no more than 0.5 pt. For methods aimed at group
P@5 it's the other way round: the highest dev group P@5 such that dev R@1 drops by no more than 0.5 pt.
Each chosen method is run ONCE on test, with a paired cluster bootstrap (clusters = the split clusters) against
finetuned_random and Holm over all the methods taken to test. Negative results get logged. D9 (own photos) stays last.

## Finding 31: failure analysis on LookBench dev (finetuned_random, 500 queries, top-20 rankings)
The exact item is at rank 1 for 264 queries. That leaves 236 misses: 206 of them (87%) have the RIGHT garment group at
rank 1 but the wrong item, and 30 have the wrong group. The true item is at rank 2-5 for 80, at rank 6-20 for 59, and
not in the top 20 for 97.
So the model mostly finds the right kind of garment but not the right design (print, colour blocking, details).
Re-ranking the top 20 can win back at most 139 of the 236 misses (an R@1 ceiling of 0.81 on dev). The 97 outside the
top 20 need a better embedding (DINOv2 fusion, or training on real street-to-shop pairs).

## Pre-registration: colour re-ranking on real photos (evaluation/color_rerank.py)
Re-rank the top 20 of finetuned_random with score = similarity + beta * colour similarity (cosine of 128-bin HSV
histograms, retrieval/color_hist.py). Two variants: "plain" (whole images) and "masked" (near-white gallery pixels,
S < 26 and V > 230 on a 0-255 scale, are left out, because product photos have white backgrounds and street crops don't).
Grid: beta in {0, 0.02, 0.05, 0.1, 0.2} x 2 variants, DEV only, default choice rule. Check: beta = 0 reproduces the
saved R@1 and group P@5.
Dev result (results/color_rerank_dev.csv; check OK, beta = 0 gives 0.528):

| variant, metric | beta 0 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| plain, R@1 | 0.528 | 0.540 | 0.532 | 0.524 | 0.514 |
| plain, group P@5 | 0.866 | 0.865 | 0.864 | 0.869 | 0.862 |
| masked, R@1 | 0.528 | 0.540 | 0.528 | 0.530 | 0.510 |
| masked, group P@5 | 0.866 | 0.864 | 0.864 | 0.870 | 0.859 |

The rule picks plain with beta = 0.02 (tied with masked on R@1, and plain keeps more group P@5). The gain on dev is
+1.2 pts, which is small and may be noise. It goes to test in the final batch.

## Pre-registration: group bonus and DINOv2 fusion on real photos (evaluation/fusion_lookbench.py)
The embeddings are saved once by evaluation/export_lookbench_features.py (check: it reproduces every saved rank-1
picture), and all the tuning is done in numpy on DEV only.
(a) Group bonus (the old "step 2"): score = similarity + w * P(gallery item's group | photo). P is zero-shot
    FashionSigLIP with the app prompts (serving/type_text_app.npy), summed per group, and the gallery item's group
    comes from its label (the app also knows each item's type). w in {0, 0.02, 0.05, 0.1, 0.2}, using the rule for
    group-aimed methods (highest dev group P@5, dev R@1 drop <= 0.5 pt).
(b) DINOv2 fusion: score = (1 - a) * finetuned_random similarity + a * DINOv2 similarity (ViT-B/14, 518 px),
    a in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1}, default rule.
(c) Combined: the chosen (b), then the chosen (a) bonus, then the chosen colour re-ranking, each at its own chosen
    value with NO re-tuning. Reported on dev, then taken to test along with (a), (b) and colour separately.
Check: a = 0, w = 0 reproduces the saved R@1 and group P@5 of finetuned_random.

## Finding 32: group bonus, DINOv2 fusion and colour on real photos (dev choice, then test ONCE)
The embeddings come from evaluation/export_lookbench_features.py (check: rank-1 picture reproduced), and the tuning was
done with evaluation/fusion_lookbench.py on DEV (check: the plain model gives the saved 0.528). From results/fusion_dev.csv:

| DINOv2 a | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 1 |
|---|---|---|---|---|---|---|
| R@1 | 0.528 | 0.538 | 0.510 | 0.500 | 0.448 | 0.306 |
| group P@5 | 0.866 | 0.876 | 0.878 | 0.873 | 0.860 | 0.791 |

| group bonus w | 0 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| R@1 | 0.528 | 0.536 | 0.528 | 0.520 | 0.512 |
| group P@5 | 0.866 | 0.883 | 0.903 | 0.913 | 0.916 |

By the rules, that gives DINOv2 a = 0.1 and group bonus w = 0.05 (w = 0.1 drops R@1 by 0.8 pt, which isn't allowed).
Combined (a 0.1, w 0.05, colour beta 0.02, no re-tuning) on dev: R@1 0.534, group P@5 0.904.
(Exploratory, not pre-registered: without colour 0.528 / 0.903; without DINOv2 0.536 / 0.904.)
TEST, run once (evaluation/fusion_test_stats.py -> results/fusion_test.csv; paired cluster bootstrap against
finetuned_random over the 621 split clusters; Holm over the 4 methods per metric; check: base = saved 0.511):

| method | R@1 (base 0.511) | diff [95% CI] | p_Holm | group P@5 (base 0.871) | diff [95% CI] | p_Holm |
|---|---|---|---|---|---|---|
| colour | 0.516 | +0.4 [-0.4, +1.5] | 1 | 0.866 | -0.5 [-0.9, -0.2] | <0.0005 |
| DINOv2 | 0.516 | +0.4 [-1.5, +2.6] | 1 | 0.867 | -0.5 [-1.6, +0.6] | 0.44 |
| group bonus | 0.509 | -0.2 [-0.7, 0.0] | 1 | 0.905 | +3.4 [+2.4, +4.5] | <0.0005 |
| combined | 0.522 | +1.0 [-1.0, +3.3] | 1 | 0.902 | +3.1 [+1.9, +4.3] | <0.0005 |

What this means: the agreed target is reached on held-out real photos. Group P@5 is 0.905 with the group bonus alone
(w = 0.05, the same weight the app already uses at type level), and exact-item R@1 doesn't change. Now the negative
results. DINOv2 on its own is much weaker than our model (dev R@1 0.306 vs 0.528) and adds nothing significant in
fusion, and colour re-ranking costs a bit of group P@5 without a significant R@1 gain. So my hope that a
higher-resolution self-supervised model would keep the design details is NOT supported. Exact-item R@1 (~0.51) is
still the open problem (Finding 31: the misses are at the design level), and the only lever left is real
street-to-shop training.
App change after F32: serving/api.py now uses the GROUP-level bonus, score = similarity + 0.05 * P(item's group |
photo) (the sum of the zero-shot type probabilities in the item's group), which is exactly the version I tested on real
photos. The w is the same as before, and serving/rerank.json records level "group" and the evidence. I haven't
re-measured it on synthetic photos, so the type-level numbers of F29 no longer describe the app exactly (to check with D9).
Decision (Nov 19): DeepFashion2 real-pair training is skipped for now. It's the only lever left for exact-item R@1, but
it's costly (an access form, a large download, hours of training). I'll revisit it only if more than 1 week is left
after G.

## Finding 33: Docker with the app catalogue (Dockerfile, docker-compose.yml, serving/latency.py)
The image now ships both catalogues and serves the app catalogue by default. A build step checks that the saved app
index matches (the weights line plus the 33,141 image paths), so it gets reused instead of re-embedded on the CPU.
Check: I saved a reference top 5 from the Mac's own app API (MPS) for 105 synthetic medium photos
(results/latency_reference_app.csv), and the container gives the same top 5 for 105/105.
CPU in Docker (Colima, 4 vCPU), 100 photos after 5 warm-up: server median 263 ms, p95 279 ms; round trip median
265 ms, p95 282 ms (the evaluation catalogue in F26: 266 / 282 ms). Five times more items cost nothing measurable,
because scoring 33k x 768 dot products is tiny next to the two image-model forward passes.
Also new, but not evaluated: an optional crop box in the UI (two sliders, cropped in memory; by default the original
bytes are sent). LookBench queries are hand-drawn crops, so the setting I evaluated is the one with the garment marked.

## Finding 34: H&M items added to the app catalogue (data_prep/hm_catalog.py, data_prep/app_add_hm.py)
The app catalogue grows from 33,141 to 54,178 items with 21,037 H&M products (2018-20; adult sections; 27 product
types, at most 1,000 per type; mostly flat or plain-background photos). Before running, I mapped each H&M type to an
existing app type (e.g. Hoodie -> Sweatshirts, Boots -> Casual Shoes), so the badge prompts, the out-of-catalogue
threshold and the group bonus stay the same; the page shows H&M's own type name.
Only the new images were embedded and appended. Checks: the old 33,141 vectors are unchanged, 20 re-embedded images
reproduce their saved vectors (cosine > 0.999), and the engine loads the merged index without re-embedding. The old
files are kept as *_fpi. Evaluation numbers aren't affected (they use the evaluation catalogue).
Own photo (a long tiered floral skirt worn by a person, crop box around the skirt): the top 5 are all skirts, maxi and
tiered or floral in the right tones, and rank 1 is a black-ground floral tiered maxi (H&M) that's close in design. The
shortlist size is 1,784: with 54k items, many more are about as similar as the best match. The shortlist was calibrated
on LookBench's 3,278-item gallery, so in the app it's only indicative, as I'd already noted.
Colour names are the shops' own labels (H&M gives one main colour per item), while the swatch is measured, so the page
now says "listed as".

### Finding 34 addendum (Nov 19, 13:10): jeans photos after the H&M merge; photo style beats cut
This is qualitative only: 5 photos, not chosen at random, looked at after the app was frozen, with no setting changed.
None of them is a D9 wardrobe photo (they're web/product photos). App catalogue 54,178 items, group bonus w = 0.05, no
crop box.
- F29 flat lay (my_photos/jeanstrial.jpg, light-wash straight): the top 5 are all adult straight light/mid-wash jeans,
  all H&M flat lays, with similarity 0.81-0.75 (before: kids' denim shorts first at 0.660). Kids' items were also
  hidden in between (F30), so the H&M effect here is the flat-lay adult jeans and the higher similarity.
- F30 worn wide-leg (my_photos/images.jpg): the top 5 are IDENTICAL to before (the same 5 men's straight/bootcut jeans
  photographed on a person, 0.67/0.67/0.66/0.62/0.59); only the shortlist grew (66 -> 182). And yet the catalogue now
  has the cut: of 999 H&M Trousers, 250 descriptions say denim, 37 say denim + "wide", 14 + "loose", 8 + "flare",
  5 + "relaxed".
- Three more flat-lay web photos: wide-leg light wash with stains -> 5/5 H&M wide-leg light denim (0.66-0.68). Faded
  black straight -> grey/black denim, 4 H&M + 1 FPI, with #3 an H&M "Leggings/Tights" (still bottoms). Dark greenish
  vintage wash, wide straight -> 5/5 H&M jeans (0.72-0.75), roughly the right cut, but the vintage tone is missed (3/5
  "colour slightly different", all a brighter blue).
My reading: with a flat-lay photo, the H&M flat-lay items with the right cut (wide-leg included) come first. With a
worn photo, the old worn-on-model jeans come first, even though wide-leg items now exist. For jeans, the model matches
HOW the garment was photographed before WHAT the cut is. Correction to F30: catalogue coverage is necessary but not
sufficient, and photo style decides (the same mechanism as F29, in the other direction).
Scope (added 13:27, after the sweater rehearsal below): a worn sweater matched flat H&M sweaters with no trouble, so
photo style isn't a general rule. It's probably specific to jeans (the old catalogue has hundreds of men's jeans
photographed on a model).
Not measured: where the 37 wide-leg denim items rank for the worn photo (that would need a small script).

### Demo rehearsal (Nov 19, 13:20): my own brown crew-neck sweater (not in the catalogue)
CORRECTION (13:40): this sweater is MINE (I first wrote it down as a friend's), so it's a D9 wardrobe garment that got
searched in the app before D9. My decision, made now and before any D9 photo: it stays in D9 like every other garment
(the protocol drops nothing), and it gets flagged in the D9 result. The app wasn't changed after seeing it (it's been
frozen since the H&M merge), so seeing it can't have tuned the method. Dropping it now, after seeing its results, would
be the biased choice.
Descriptive only; the app is frozen and nothing changed. Same sweater, three photos:

| photo | rank 1 | sim | shortlist | colour badges (top 5) |
|---|---|---|---|---|
| worn, no crop (room, face) | H&M brown knit "Greenish Khaki" | 0.71 | 1,386 | 4 very close, 1 different (beige) |
| worn, crop box on the sweater | same item | 0.74 | 1,081 | 5 very close |
| flat on a bed (other lighting) | same item | 0.74 | 1,299 | 5 slightly different |

All 15 results are sweaters (group badge "tops"), mostly H&M flat photos. The crop box raised the top similarity and
shrank the shortlist, and rank 1 was the same item in all three photos. In the flat photo the estimated colour moved
from #4b3e36 to #7d6f67 (lighter and greyer) because of the lighting, so the colour badges changed while the ranking
barely did, just as the README's lighting caveat says. Demo recipe I chose: a worn photo with the crop box around the
garment.

## Pre-registration: "colour can't be judged" check for the app (serving/colour_check.py)
Problem: on black-and-white or very dark photos, the colour estimate (median CIELAB of the image centre) is
meaningless, but the app still shows confident colour badges. The check uses only the QUERY photo; catalogue colours
and the colour bands of Finding 24 stay as they are.
Two numbers per photo, over the whole photo (longer side 224 px): colourfulness = the 95th percentile of CIELAB chroma,
and brightness = the 95th percentile of L*. A photo is flagged "black-and-white" if colourfulness < T_c and "too dark"
if brightness < T_L. The rule, fixed now: T_c and T_L are the 1st percentile of each number over the DISTINCT LookBench
query crops in the DEV half (identical crops counted once), so by construction at most ~1% of normal real photos get
flagged. No other value is tried.
Check, done once, on the distinct TEST-half crops and on 300 random catalogue images (seed 1): the flag rate on the
original photos (false alarms; the catalogue split into items listed Black/White/Grey and the rest), and on grayscale,
brightness x0.5 (inside the training range 0.45-1.55), x0.3 and x0.15 (outside) versions; plus how often the colour
estimate moves from the original by more than the "different" band (Delta-E > 37.0).
Ship rule: ship if false alarms on the test originals are <= 2% and grayscale detection is >= 95% on the test crops.
The darkness flag is reported (there's no ground truth for "too dark" beyond the estimate moving).
Sanity check: the script first recomputes the colour bands of Finding 24 and stops if they don't match serving/badges.json.


## Finding 35: "colour can't be judged" check (serving/colour_check.py -> results/colour_check.csv)
Check OK: the colour bands come out as 16.45 / 37.00, matching serving/badges.json. Thresholds from the pre-registered
rule (1st percentile over 362 distinct dev crops): colourfulness < 3.64, brightness < 54.41.
Test (338 distinct test crops), flag rate:

| version | black-and-white | dark | either |
|---|---|---|---|
| original | 0.012 | 0.015 | 0.027 |
| grayscale | 1.000 | 0.015 | 1.000 |
| brightness x0.5 | 0.021 | 1.000 | 1.000 |
| x0.3 | 0.092 | 1.000 | 1.000 |
| x0.15 | 0.467 | 1.000 | 1.000 |

Catalogue (300 random items, seed 1), black-and-white flag on the originals: 0.006 for items listed in a colour (172)
and 0.227 for items listed Black / White / Grey (128). The grayscale versions were flagged 1.000 in both.
The pre-registered ship rule FAILED: 2.7% false alarms on the test originals, against a limit of 2%. The cause is that
two flags, each set to fire on ~1% of normal photos, together fire on ~2.7%. The rule left no room for variation
between dev and test. That's a design error in the rule, not bad luck.
The dark flag fires on 100% of x0.5 photos (inside the training range 0.45-1.55), even though the colour estimate
never moved past the "different" band there (0%). It only moved at x0.3 (39%) and x0.15 (48%). So brightness alone
doesn't tell you whether colour can be judged.
The black-and-white flag by itself catches 100% of grayscale photos, with 1.2% false alarms on real test crops and
0.6% on colourful catalogue items, but 22.7% on plain-background photos of black / white / grey items. Those photos
have no colour anywhere, so there's no way to tell them apart from a black-and-white photo.
Also, grayscale never moved the estimate past Delta-E 37 (0%), so "moved past the different band" is a poor yardstick
for black-and-white photos. The harm there is a meaningless grey swatch next to confident badges.
Decision (made AFTER seeing the test numbers, so the test half is used up for this check; it's a display feature, not
an accuracy claim): ship the black-and-white flag only, since it meets both ship conditions on its own, and drop the
dark flag. In the app, below colourfulness 3.64 there's no swatch and no colour badges, just the message "This photo
has almost no colour (a black-and-white photo, or a black, white or grey garment), so the colour can't be judged".
Darkness moves to task 3 (colour-badge errors on darkened real photos).
What this check doesn't fix: on whole photos the estimate can pick up the background (a wallet on a beige backdrop
gave #9c7f68, which is the backdrop, not the dark-brown wallet). The page now suggests marking the garment.

## Pre-registration: app catalogue clean-up (data_prep/app_cleanup.py)
Problems I found in the app: H&M fabric close-ups reach the top 5; some Kaggle rows show a different product from their
label (item 13378, "Timberland Unisex Infant Pink Casual Shoes", is actually a black Fossil wallet); and kids' items
listed as Men/Women/Unisex aren't hidden (~444 by product name). This is for the app catalogue only. The same
rules are COUNTED on the evaluation catalogue (the first 6,197 rows) but never applied there.
Rules, fixed now, with no value tried twice (pretrained FashionSigLIP, zero-shot):
  closeup     : a 2-way choice between "a photo of a close-up of fabric" and the item's own type prompt; hide if P >= 0.5.
  wrong_image : the zero-shot group (app prompts, 7 groups, as in the group badge) differs from the listed group with
                group probability >= 0.9 (the badge's own threshold); hide.
  kids_name   : the Kaggle product name matches kid(s), kid's, kidswear, infant(s), baby, babies, toddler, boy(s),
                boy's, girl(s), girl's (whole words, any case); hide. H&M items come from adult sections.
Check, by eye: for each rule, a contact sheet of 24 random items it drops and 24 random items it keeps (seed 0; for
kids, only drops that aren't already hidden by gender). A rule ships only if I judge >= 20/24 of its dropped items to
be correct drops (judged by me, not independently). I'll report how many items each rule drops, per source.
Caveat: wrong_image uses the same zero-shot model as the group badge and the group bonus, so hiding its disagreements
makes the app look more self-consistent than it is. Later app measurements of the group badge will be slightly
flattered by it.
Sanity checks: the pretrained embeddings of the first 6,197 app rows equal the saved fashion_siglip.npy, and the batch
zero-shot group equals TypePredictor("app").analyse on 5 random items.

## Finding 36: app catalogue clean-up (data_prep/app_cleanup.py -> serving/app_hidden.csv, results/app_cleanup.csv)
Checks OK: the pretrained embeddings of the first 6,197 app rows equal the saved fashion_siglip.npy, and the batch
zero-shot groups equal TypePredictor("app").analyse on 5 random items.
Items each rule would hide (evaluation rows / Kaggle extra / H&M / whole app): closeup 1 / 24 / 154 / 179;
wrong_image 30 / 32 / 413 / 475; kids_name 395 / 411 / 0 / 806 (542 not already hidden by gender).
Contact sheets (report/figures/cleanup_*_dropped.png, 24 random drops each; Claude did the counting and I accepted the
counts, as in Finding 20, so it's not an independent annotation). Correct drops:
- kids_name: 24/24, so it SHIPPED. I saw no kids' item among the 24 kept. The app now hides 809 items (267 by gender,
  542 more by name), and "not sure" and the shortlist still use only visible items.
- closeup: 11/24 real fabric close-ups (15/24 if I count 4 items whose image shows another product), so NOT shipped.
  The rest are whole garments with odd H&M labels (scarves listed as "Hat/beanie"). One close-up slipped through among
  the 24 kept (a partial bag shot).
- wrong_image: about 0/24, so NOT shipped. It flags H&M's own label noise on normal garments, not broken items: parkas
  and hooded jackets listed as "Hoodie" (mapped to Sweatshirts; 163 of the 413 H&M hits), jumpsuits and playsuits
  listed as "Trousers"/"Shorts", shirts listed as "Jacket". It didn't catch the wallet (13378), but the wallet is hidden
  anyway by kids_name (its name says "Infant").
What this means: (1) fabric close-ups and wrong-image rows stay a stated limitation of the app catalogue. I didn't try
a second rule, because re-tuning after seeing the sheets would be fishing. (2) H&M's own type labels are noisy, so the
app's group badge and group bonus inherit that noise on H&M items. (3) 395 of the 6,197 evaluation items (6.4%) are
kids' items by name. That's harmless for the evaluation (nothing is hidden there), but I note it as label noise.
Limitation: kids' items are hidden, not detected in the photo. A photo of a kids' garment gets the most similar ADULT
items, with no warning.

## Pre-registration: whole photo vs crop on LookBench (task 2a, evaluation/whole_photo.py)
Question: every real-photo number so far uses LookBench's hand-drawn crop, but by default the app sends the WHOLE photo
(the crop box is optional). How much worse is the app's default setting?
Runs (no tuning, nothing chosen): evaluation/run_lookbench.py --query_image raw_image for finetuned_random and
fashion_siglip, the same 3,278 gallery and dedup protocol, plain similarity, no group bonus.
Reported on all 981 queries and on the TEST half: R@1, R@5 and group P@5 for crop and whole photo. Paired cluster
bootstrap over the split clusters for (1) finetuned_random whole - crop, (2) fashion_siglip whole - crop, and (3) on
whole photos, finetuned_random - fashion_siglip; Holm over these 3 per metric and side.
Also: the R@1 ceiling on whole photos (one photo is the query for several items but only gets one ranking), and R@1 for
photos with one query vs photos shared by several queries.
Sanity check: the crop numbers reproduce the saved R@1 (lookbench_summary.csv) and group P@5 (group_p5.csv).
My expectation, written now: the whole photo is worse, and the size of the drop is the result.

## Finding 37: whole photo vs crop on LookBench (task 2a, evaluation/whole_photo.py -> results/whole_photo.csv)
Checks OK: crop R@1 and group P@5 reproduce the saved values for both models (0.520 / 0.869 and 0.486 / 0.869).
The 981 queries come from 621 distinct whole photos, and 589 queries share their photo with other queries. The R@1
ceiling on whole photos (one ranking per photo) is 0.902, so shared photos are NOT what limits the score (many gallery
pictures are outfit photos that show several of the photo's items).

| model, view | all 981: R@1 / R@5 / group P@5 | test 481: R@1 / R@5 / group P@5 |
|---|---|---|
| finetuned_random crop | 0.520 / 0.694 / 0.869 | 0.511 / 0.701 / 0.871 |
| finetuned_random whole | 0.406 / 0.549 / 0.668 | 0.407 / 0.559 / 0.684 |
| fashion_siglip crop | 0.486 / 0.657 / 0.869 | 0.489 / 0.665 / 0.874 |
| fashion_siglip whole | 0.376 / 0.498 / 0.649 | 0.378 / 0.509 / 0.664 |

Paired cluster bootstrap, Holm over 3 tests per metric and side (test half):
- finetuned_random whole - crop: R@1 -0.104 [-0.145, -0.063], group P@5 -0.187 [-0.232, -0.142], p_Holm < 0.001
- fashion_siglip whole - crop: R@1 -0.110 [-0.156, -0.069], group P@5 -0.210 [-0.256, -0.163], p_Holm < 0.001
- whole photos, finetuned_random - fashion_siglip: R@1 +0.029 [+0.006, +0.054] p_Holm 0.020; group P@5 +0.020
  [+0.005, +0.035] p_Holm 0.013
(All 981 tells the same story: -0.114 / -0.200 and +0.030 / +0.019, all p_Holm <= 0.001.)
What this means:
(1) The app's default (whole photo) is much worse than the setting I evaluated: about 11 points of exact-item R@1 and
about 20 points of group P@5. The headline group P@5 (0.905 with the group bonus) is a CROP number. The app's default
is closer to 0.67-0.68 (that's without the bonus, which I didn't measure on whole photos).
(2) Marking the garment matters more than any model change I've tried: crop vs whole is worth +10 to +11 points of
R@1, while fine-tuning vs pretrained is worth +3.
(3) Fine-tuning still helps on whole photos, and there it also gives a small significant group P@5 gain (+2.0
points), which it didn't on crops (Finding 21).
(4) Photos shared by several queries are harder in BOTH settings (whole 0.28 vs 0.60, crop 0.38 vs 0.72). These are
outfit photos with several garments, so it's not only the ambiguity of using the whole photo.
Not measured here: the 54k app catalogue (task 2b) and the group bonus on whole photos.

## Pre-registration: LookBench against the official corpus (task 2b, evaluation/lookbench_full.py)
I checked on Hugging Face (srpone/look-bench, version_info.json): the official corpus of each subset is its own gallery
plus the 58,275 shared "noise" images. For real_streetlook that's 3,278 + 58,275 = 61,553 entries, the number in the
paper's Table 1. Noise images have no category or attributes (category "noise data"), so on this corpus only
exact-item metrics are defined (R@1, R@5), and group P@5 stays a 3,278-gallery metric. (The paper lists 1,000
real_streetlook queries; the release has 981, which is the set we use.)
Runs, nothing tuned: finetuned_random and fashion_siglip, crop and whole photo. The noise is embedded once per model,
the gallery embeddings are the saved ones, and the dedup protocol is used (identical image files merged). Reported:
R@1 and R@5 (95% CIs), the share of noise images in the top 5, and a paired cluster bootstrap finetuned_random -
fashion_siglip (split clusters) on all 981 and on the test half.
Sanity check: ranking only the 3,278 gallery entries reproduces the saved R@1 of every query.
Not in this step: re-calibrating "not sure" and the shortlist at this scale (that's next), ensemble_mv (three times
the embedding cost), and near-duplicates between noise and gallery (only exact copies are merged).
To decide together after the results: which corpus the later experiments use.

## Finding 38: LookBench against the official 61,553-image corpus (task 2b, evaluation/lookbench_full.py)
Checks OK: 0 noise images are byte-identical to a gallery picture, and for both models and both views, ranking only
the 3,278 gallery entries reproduces the saved rank-1 result for 981/981 queries.
R@1 / R@5 on the full corpus (all 981; 95% CI), with the 3,278-gallery R@1 and the share of noise in the top 5:

| model, view | R@1 [95% CI] | R@5 | gallery R@1 | noise in top 5 |
|---|---|---|---|---|
| finetuned_random crop | 0.501 [0.471, 0.531] | 0.677 | 0.520 | 16% |
| finetuned_random whole | 0.382 [0.353, 0.412] | 0.521 | 0.406 | 23% |
| fashion_siglip crop | 0.472 [0.441, 0.504] | 0.643 | 0.486 | 14% |
| fashion_siglip whole | 0.353 [0.323, 0.383] | 0.474 | 0.376 | 21% |

finetuned_random - fashion_siglip, paired cluster bootstrap (split clusters):
- crop, all 981: R@1 +0.029 [+0.011, +0.046] p=0.001; R@5 +0.034 p<0.001
- crop, test 481: R@1 +0.023 [+0.000, +0.047] p=0.070; R@5 +0.025 [-0.004, +0.056] p=0.113
- whole, all 981: R@1 +0.030 [+0.010, +0.050] p=0.004; R@5 +0.047 p<0.001
- whole, test 481: R@1 +0.025 [+0.002, +0.050] p=0.044; R@5 +0.052 [+0.029, +0.079] p<0.001
What this means: (1) A corpus 19 times bigger costs only about 2 points of R@1 (crop 0.520 -> 0.501, whole photo
0.406 -> 0.382). The 58k distractors rarely beat the true item, even though they fill 14-23% of the top-5 slots.
(2) The fine-tuning gain is about +2.5 to +3 points of R@1 at this scale too. On the test half alone it's borderline
for crops (p = 0.07) and significant for whole photos (p = 0.044), and with whole photos the R@5 gain is bigger (+5
points). These p-values aren't Holm-corrected. (3) It's still not comparable to the paper's FashionSigLIP 55.2% "fine
R@1" (a different correctness rule, Finding 5).
This corpus only has exact-item metrics (noise images have no category), so group P@5 stays a 3,278-gallery metric.

## Decision (Nov 19, after Finding 38): protocol for all later experiments
From here on, the main protocol for exact-item metrics (R@1, R@5) is the OFFICIAL corpus (3,278 gallery + 58,275
noise = 61,553 entries) with the dedup protocol. Group P@5 stays on the 3,278 gallery, since noise has no category. I
keep the old 3,278-gallery exact-item numbers as a clearly labelled secondary protocol. Later re-rankers work on the
top 20 of the full corpus. Same dev/test split, same paired cluster bootstrap, Holm across methods.

## Pre-registration: shortlist re-calibration at the official-corpus scale (serving/calibrate_full.py)
The shortlist q (relative split conformal, alpha 0.1) is re-calibrated with the best similarity taken over the
61,553-image corpus instead of the 3,278 gallery, separately for crops and for whole photos (all 981 photos,
finetuned_random). The rule, fixed now: the app uses the LARGER of the two q values, so the 90% holds on LookBench for
both kinds of photo the app can get. Reported: q for each view, coverage over 200 random 50/50 splits, the median
shortlist size on the corpus, the coverage of the OLD q at the new scale, and the coverage of each view's q on the
other view. "Not sure" is NOT re-calibrated, because its loss needs garment groups and the noise images don't have
any; I only report its answer rate at the current lambda on the gallery vs the full corpus. Sanity checks: gallery-only
crops give back the saved q, and the best full-corpus similarity equals the saved rank-1 scores of lookbench_full.py.

## Finding 39: shortlist re-calibrated at the official-corpus scale (serving/calibrate_full.py -> results/calibration_full.csv)
Checks OK: gallery-only crops give back the saved q (0.228381), and the best full-corpus similarity equals the saved
rank-1 score of lookbench_full.py for every photo.

| view | q (61,553 corpus) | coverage (200 50/50 splits) | median shortlist | coverage of the OLD q |
|---|---|---|---|---|
| crop | 0.2297 | 0.902 | 116 entries | 0.897 |
| whole photo | 0.2363 | 0.902 | 878 entries | 0.888 |

The crop q covers 0.889 on whole photos, and the whole-photo q covers 0.905 on crops. By the pre-registered rule, the
app now uses q = 0.2363 (it was 0.2284; the old value is kept in serving/calibration.json as q_gallery_3278).
"Not sure" (lambda 0.671, unchanged) answers 92% of crops on the gallery and 93% on the full corpus, but only 49% of
whole photos (on both corpora).
What this means: (1) A corpus 19 times bigger barely moves the shortlist constant (0.228 -> 0.230 for crops), and the
old value still covers about 89-90%, which fits Finding 38 (distractors rarely beat the true item). (2) The honest 90%
shortlist is long: about 116 items for a marked garment and about 880 for a whole photo, out of 61,553. The page only
shows the number when it's at most 50, so in practice it says "many". (3) With a whole photo, "not sure" fires on
about half the photos. That fits Finding 37 (whole photos are much harder): the warning reacts to the harder setting
even though it was calibrated on crops.

## Pre-registration: robustness to bad photos on REAL LookBench crops (task 3, evaluation/robustness.py)
Question: how much do imperfect photos hurt, inside vs outside the training augmentation? So far all our robustness
numbers are synthetic (made with the same augmentation as training).
Filters, fixed now, applied to the 981 real crops (sizes in px at the 224 input, as in phone_aug.py):
  inside the training augmentation: Gaussian blur radius 1.5 and 3; brightness x0.5
  outside: Gaussian blur radius 6; horizontal motion blur 5, 9, 15 px; brightness x0.3 and x0.15;
           half saturation and grayscale; warm colour cast (R, G, B multiplied by 1/0.85/0.65 and 1/0.7/0.45).
Methods: finetuned_random, fashion_siglip, and the colour histogram as a reference baseline (task 4 adds the credible
ones).
Metrics: R@1 and R@5 on the official corpus (main protocol), group P@5 on the 3,278 gallery; the "not sure" rate of the
app rule (finetuned_random); and the share of photos whose TRUE item gets the "different" colour badge, before and
after the black-and-white fix (Finding 35). All 981 queries (nothing is tuned).
Tests: the drop from clean with a 95% cluster-bootstrap CI for each method, and finetuned_random - fashion_siglip per
filter (R@1 and group P@5) with a paired cluster bootstrap over the split clusters, Holm over the 14 conditions per
metric. My expectation, written now: small drops inside the augmentation, bigger ones outside, and fine-tuning should
help more inside than outside.
Sanity check: with no filter, both models reproduce the saved per-query full-corpus R@1.

## Finding 40: robustness to bad photos on real LookBench crops (task 3, evaluation/robustness.py)
Checks OK: with no filter, both models reproduce the saved per-query full-corpus R@1. All 981 crops; R@1 on the 61,553
corpus (drop from clean, 95% cluster CI) / group P@5 on the gallery; * = outside the training augmentation.

| condition | finetuned_random R@1 (drop) | group P@5 | fashion_siglip R@1 (drop) | group P@5 | colour histogram R@1 |
|---|---|---|---|---|---|
| clean | 0.501 | 0.869 | 0.472 | 0.869 | 0.077 (group P@5 0.327) |
| gauss 1.5 | 0.490 (-1.0, n.s.) | 0.838 | 0.389 (-8.3) | 0.790 | 0.072 |
| gauss 3 | 0.355 (-14.6) | 0.718 | 0.147 (-32.5) | 0.429 | 0.058 |
| gauss 6 * | 0.118 (-38.2) | 0.415 | 0.022 (-45.0) | 0.191 | 0.042 |
| motion 5 * | 0.501 (0.0) | 0.848 | 0.444 (-2.8) | 0.843 | 0.075 |
| motion 9 * | 0.454 (-4.7) | 0.828 | 0.368 (-10.4) | 0.795 | 0.068 |
| motion 15 * | 0.368 (-13.3) | 0.751 | 0.244 (-22.8) | 0.659 | 0.066 |
| dark x0.5 | 0.501 (0.0) | 0.865 | 0.461 (-1.1) | 0.864 | 0.000 |
| dark x0.3 * | 0.490 (-1.0) | 0.860 | 0.451 (-2.1) | 0.850 | 0.000 |
| dark x0.15 * | 0.391 (-10.9) | 0.829 | 0.358 (-11.4) | 0.790 | 0.000 |
| half colour * | 0.505 (+0.4) | 0.866 | 0.472 (0.0) | 0.864 | 0.024 |
| grayscale * | 0.379 (-12.1) | 0.846 | 0.388 (-8.4) | 0.847 | 0.001 |
| warm cast mild * | 0.485 (-1.5) | 0.861 | 0.456 (-1.6) | 0.861 | 0.000 |
| warm cast strong * | 0.452 (-4.9) | 0.862 | 0.432 (-4.0) | 0.855 | 0.000 |

finetuned_random - fashion_siglip, R@1 (paired cluster bootstrap, Holm over the 14 conditions):
- clean +2.9 (p_Holm 0.006)
- gauss 1.5 / 3 / 6: +10.1 / +20.8 / +9.6; motion 5 / 9 / 15: +5.6 / +8.6 / +12.4 (all p_Holm < 0.001)
- dark x0.5 / x0.3 / x0.15: +4.0 / +4.0 / +3.4 (p_Holm < 0.001 / 0.006 / 0.052)
- half colour +3.3 (< 0.001); grayscale -0.9 (0.43); warm cast mild / strong +3.0 / +1.9 (0.052 / 0.27)
For group P@5, the big gains only show up under blur (gauss 3 +29.0, gauss 6 +22.4, motion 15 +9.2 points), and
there are none for colour changes.
"Not sure" (finetuned_random) fires on 7% of clean photos, 15% at gauss 3, 19% at gauss 6, 14% at motion 15 and 22% at
dark x0.15. The colour badge calls the TRUE item "different" for 4% of clean photos, 12% / 36% / 47% at dark x0.5 /
x0.3 / x0.15, and 4% for grayscale. After the black-and-white fix: grayscale 0%, dark x0.3 33%, dark x0.15 27%.
What this means:
(1) My expectation was only partly right. Inside the augmentation, mild blur and half brightness cost nothing, but blur
radius 3 (the edge of the training range) already costs 15 points.
(2) The main value of fine-tuning is ROBUSTNESS TO BLUR. On clean real photos it adds +3 points; on blurred ones it adds
+6 to +21 points, including motion blur, which it never saw in training. For colour changes it never saw (grayscale,
colour casts), it adds nothing beyond its clean-photo gain. So the synthetic augmentation does transfer to real
photos, for the kinds of damage it simulates.
(3) Very blurry photos break every model (gauss 6: R@1 0.12), and "not sure" catches few of them (it fires on 19%
while R@1 is 12%). As a safety net it's weak.
(4) Dark photos are the colour badge's real problem. Even after the fix, 27-33% of very dark photos give the true item
a wrong "different" badge (the darkness flag was dropped in Finding 35).
(5) The colour histogram finds the exact item 7.7% of the time on real crops (vs 1% on synthetic photos), but it fails
completely under any lighting or colour change (R@1 0.000), as you'd expect from a pure colour method.
The "not sure" numbers only mean something for finetuned_random, since lambda is calibrated on its similarity scale.

## Pre-registration: keyword search baseline, "how shops work today" (task 4a, evaluation/keyword_baseline.py)
This replaces the strawman baselines (the colour histogram, and the type menu in random order). A shopper types what
they see, and the shop ranks items by text match: BM25 (k1 = 1.2, b = 0.75, fixed, no tuning) over each gallery item's
LookBench text (category + main attribute + other attributes; read from Hugging Face and checked row by row against
our saved files). The shopper is IDEALISED: the query text is LookBench's own annotation of the photo, so it always
uses the shop's words. Two variants: keywords_basic (category + main attribute) and keywords_detailed (plus the other
attributes). Ties are broken at random, averaged over 50 orders.
Protocol: the 3,278 gallery only (dedup), because noise images have no text and would be invisible to keyword search on
the full corpus. The photo model is compared on the same gallery (saved finetuned_random results). Metrics: R@1, R@5,
group P@5, on all 981 and on the test half; paired cluster bootstrap photo - keywords for R@1 and group P@5, Holm over
the 2 variants. I also report how often a query's annotation is identical to its true item's annotation (that is, how
idealised the shopper is).
My expectation, written now: keywords get the KIND of garment right (high group P@5) but rarely the exact item.

## Finding 41: keyword search baseline (task 4a, evaluation/keyword_baseline.py -> results/keyword_baseline.csv)
Checks OK: the Hugging Face text lines up row by row with our saved LookBench files, and the saved photo-model gallery
results are reproduced (R@1 0.520, group P@5 0.869). 3,278-gallery protocol (dedup), random tie-breaking averaged over
50 orders. 35.8% of queries have an annotation that's word-for-word identical to their true item's.

| method | all 981: R@1 / R@5 / group P@5 | test 481: R@1 / R@5 / group P@5 |
|---|---|---|
| keywords_basic (category + main) | 0.309 / 0.626 / 0.981 | 0.312 / 0.634 / 0.983 |
| keywords_detailed (+ other attr.) | 0.490 / 0.753 / 0.986 | 0.486 / 0.760 / 0.986 |
| photo (finetuned_random) | 0.520 / 0.694 / 0.869 | 0.511 / 0.701 / 0.871 |

Photo - keywords (paired cluster bootstrap, Holm over the 2 variants), test half:
- basic: R@1 +0.199 [+0.121, +0.284] p_Holm < 0.001; group P@5 -0.112 [-0.144, -0.085] p_Holm < 0.001
- detailed: R@1 +0.025 [-0.042, +0.096] p_Holm 0.46 (n.s.); group P@5 -0.115 [-0.145, -0.089] p_Holm < 0.001
This result goes against us, so I'll state it plainly. (1) My expectation was WRONG for the detailed variant. An
idealised shopper who types the category and every attribute in the shop's own words finds the exact item first about
as often as our photo model (0.49 vs 0.51 on the test half, no significant difference), and finds the right kind of
garment far more often (0.99 vs 0.87). (2) The photo model clearly beats the shorter, more realistic query (category +
main attribute), by +20 points of R@1. (3) Typing the category makes group P@5 almost trivial (0.98). That confirms
group P@5 is a sanity floor and not a headline, since a text box gets it nearly for free.
Why the idealised shopper is so strong here, and why that's optimistic: the query words are LookBench's own annotation
of the photo (identical to the target's annotation for 36% of queries). Real shoppers don't know words like "applique"
or "suede", and that's exactly the problem the app exists to solve (the user can't describe the garment). The gallery
is also small (2,537 distinct pictures). In a real shop, many more items would share the same few attribute words, so
there would be more ties. I couldn't measure that, because the noise images have no text.
Implication: the honest pitch isn't "photos beat text". A photo gets you what a perfect description would, without the
shopper having to write one. And photo + a typed or menu-chosen category is the natural combination (Finding 14: type
menu + our ranking, R@1 0.554 on the gallery, not tested on the frozen split).

## Pre-registration: classical (non-deep-learning) image baseline (task 4b, evaluation/classical_baseline.py)
This replaces the plain colour histogram as the non-ML image baseline. Each image (resized to 128 x 256) is described
by a masked HSV colour histogram (8x4x4; near-white pixels with S < 26 and V > 230 left out), a masked LBP texture
histogram (8 neighbours, radius 1, uniform, 10 bins) and HOG shape features (64 x 128 grayscale, 9 orientations, 16 px
cells, 2x2 blocks). Each part is normalised, and the score is a weighted sum of the three cosines.
Weights: a grid of every setting in steps of 0.1 that sums to 1 (66 settings), chosen by the highest DEV R@1 on the
official corpus (main protocol), then run ONCE on the TEST half. Variant "classical + menu": the shopper picks the right
garment group from a menu (idealised), and only gallery items in that group get ranked (gallery only, since noise has
no labels). References, not tested: colour only (masked) with the same features, and the same menu in front of our
photo model. Tests on the test half: photo - classical (full corpus) and photo (gallery, no menu) - classical + menu,
R@1, paired cluster bootstrap, Holm over these 2.
Sanity check: the saved finetuned_random query embeddings reproduce the saved gallery R@1 (0.520).
My expectation, written now: far below the photo model on exact items, with the menu helping it a lot.

## Finding 42: classical image baseline (task 4b, evaluation/classical_baseline.py -> results/classical_*.csv)
Check OK: the saved finetuned_random query embeddings reproduce the saved gallery R@1 (0.5199).
Dev choice (66 weight settings, official corpus): the best dev R@1 is 0.112, for colour 0.3 / texture 0.2 / shape 0.5,
tied with 0.3 / 0.1 / 0.6. I took the first one in the sorted table; the two differ by 0.1 in two weights. Shape
matters most, colour second and texture least. TEST half (481), run once:

| method | corpus | R@1 [95% CI] | R@5 | group P@5 (gallery) |
|---|---|---|---|---|
| photo (finetuned_random) | full 61,553 | 0.499 | 0.688 | (0.871, Finding 37) |
| classical | full 61,553 | 0.123 [0.096, 0.154] | 0.162 | 0.387 |
| colour only (masked) | full 61,553 | 0.081 [0.056, 0.106] | 0.123 | 0.315 |
| classical + menu (idealised) | gallery 3,278 | 0.237 [0.200, 0.277] | 0.360 | 1.000 |
| photo + menu (reference) | gallery 3,278 | 0.536 [0.491, 0.580] | 0.713 | 1.000 |

Photo - classical (full corpus): R@1 +0.376 [+0.323, +0.435]. Photo on the gallery without a menu (0.511) - classical
+ menu: +0.274 [+0.220, +0.329]. Both p_Holm < 0.001.
What this means: this time the expectation was right. A fair classical system (colour + texture + shape, tuned on dev)
finds the exact item first 12% of the time on the official corpus, 4 times less often than the photo model. Even with
an idealised menu in front it only reaches 24%, half of what the photo model gets without a menu. The menu adds +2.5
points to the photo model (0.511 -> 0.536 on the same test half), consistent with Finding 14.
A note on the code: the LBP step got a float copy of the grayscale image (a warning was printed). skimage converts
every input to float internally anyway, so the numbers are the same as with integer input.
Baseline ladder on the real test half (exact item first): colour histogram 0.08, classical 0.12 (full corpus);
classical + menu 0.24, short keywords 0.31, detailed idealised keywords 0.49 (gallery); photo 0.50 (full corpus),
0.51 (gallery); photo + menu 0.54 (gallery).

## Pre-registration: learned re-ranker trained on real photos (task 5, evaluation/rerank_features.py + rerank_model.py)
Problem (Finding 31): most misses are the right kind of garment with the wrong design, and a single embedding decides
on its own what matters. Stage 1: finetuned_random's top 20 distinct pictures on the official 61,553-image corpus (main
protocol; whether the true item is in the top 20 bounds what any re-ranker can do). Stage 2: a model scores each
(photo, candidate) pair from 9 features, fixed now: fine-tuned similarity, gap to rank 1, rank, pretrained
FashionSigLIP similarity, DINOv2 similarity (CLS, 224 px), DINOv2 patch matching (for each of the photo's 256 patches,
the best cosine with the candidate's patches, averaged), masked HSV colour-histogram cosine, CIEDE2000 colour
difference, and P(candidate's zero-shot group | photo). The label is whether the candidate is the true item.
Models, with fixed settings: logistic regression (standardised, C = 1) and HistGradientBoostingClassifier (200 trees,
lr 0.05, 15 leaves, seed 0), trained pointwise. The choice is made on DEV only, by 5-fold cross-validation by split
cluster inside dev: the higher cross-validated dev R@1 wins (on a tie, logistic regression). The chosen model is
trained on all of dev and run ONCE on TEST, whatever its dev result, with a paired cluster bootstrap against stage 1
(R@1, R@5).
Descriptive, after the test run (nothing gets chosen with these): a learning curve with 50 / 100 / 250 / 500 dev photos
(20 random subsets) -> test R@1; permutation importance on test; CPU latency of stage 2.
Not in this step (decided now; may follow later): extra training data from real_studio_flat (which needs a leakage
check first) and a linear adapter ablation. Group P@5 isn't defined on the full corpus, so it isn't reported here.
Sanity checks: the recomputed fine-tuned similarity equals the saved stage-1 score of every pair, and the stage-1 order
reproduces the saved dev and test R@1.
My expectation, written now: a small gain or none (the fixed-weight colour and DINOv2 fusion didn't help, Finding 32).

## Finding 43: learned re-ranker on real photos (task 5, evaluation/rerank_*.py -> results/rerank_*.csv)
Checks OK: the recomputed fine-tuned similarity equals the saved stage-1 score (max difference 2.4e-07), and the
stage-1 order reproduces the saved dev (0.502) and test (0.499) R@1 query by query. (The features script printed
"saved 0.5010"; that was the rounded 0.501. The exact saved value is 491/981 = 0.5005, as computed.)
The true item is in the top 20 (the ceiling for any re-ranker) for 0.771 of photos overall and 0.764 on dev.
Dev, 5-fold CV by cluster: stage 1 0.502, logistic regression 0.506, boosted trees 0.508, so boosted trees were chosen.
TEST, run once (481 photos, official corpus):

| | R@1 | R@5 |
|---|---|---|
| stage 1 (finetuned_random) | 0.499 | 0.688 |
| re-ranker (boosted trees) | 0.520 | 0.709 |

Difference: R@1 +0.021 [-0.004, +0.050] p = 0.12; R@5 +0.021 [-0.004, +0.048] p = 0.12 (paired cluster bootstrap).
Learning curve (descriptive; test R@1, mean of 20 random training subsets): 50 dev photos 0.486 (below stage 1),
100: 0.504, 250: 0.513, 500: 0.520. It's still rising at 500.
Permutation importance on test (R@1 lost when the feature is shuffled): fine-tuned similarity 0.151, gap to rank 1
0.072, pretrained similarity 0.025, rank 0.015, group agreement 0.012, colour difference 0.011, patch matching 0.009,
colour histogram 0.005, DINOv2 similarity 0.003.
CPU latency of stage 2: 76 ms per photo (4 threads), but only IF the candidates' DINOv2 patch tokens are stored with
the catalogue. For the 54k app catalogue that's about 54k x 256 x 768 numbers, ~21 GB in half precision, which isn't
practical as it stands.
What this means: NEGATIVE / INCONCLUSIVE at the pre-registered test. The re-ranker gains +2.1 points of R@1 on
held-out real photos, in the expected direction but not significant (the CI includes 0). The learning curve is the
more informative result. With 50 labelled photos the re-ranker does worse than doing nothing, and it improves steadily
up to the 500 available, so the method looks data-limited rather than useless. The features meant to catch "right
garment, wrong design" (patch matching, DINOv2, colour) contribute little; the model mostly re-weighs the fine-tuned
and pretrained similarities and their gaps. The expectation (a small gain or none) was right.

## Pre-registration: re-ranker with extra real training photos from real_studio_flat (evaluation/rerank_studio.py)
Reason: the learning curve in Finding 43 was still rising at 500 labelled photos. Extra data: LookBench
real_studio_flat (1,011 studio / flat-lay photos, 3,951-item gallery), used for TRAINING only.
Leakage rule, fixed now: a studio photo is dropped if its own image, or any gallery picture of its true item, is
byte-identical to or a near-duplicate of (pretrained FashionSigLIP cosine >= 0.97, the rule from Finding 18) a
real_streetlook TEST query crop or a gallery picture of a TEST item. The counts get reported.
Studio pairs: stage 1 = finetuned_random's top 20 on real_studio_flat's official corpus (its gallery + the 58,275
noise images), with the same 9 features as Finding 43. Model: Finding 43's choice (boosted trees, same settings).
sklearn's automatic early stopping, which would switch on above 10,000 rows, is switched off so nothing else changes.
It's trained on dev + studio pairs. Reported: dev 5-fold CV by cluster (studio always in training) next to Finding
43's 0.508; then ONCE on test, a paired cluster bootstrap against stage 1 with Holm over the two re-rankers taken to
test (Finding 43's and this one), plus this one minus Finding 43's, descriptively.
Sanity check: the dev-only model reproduces Finding 43's test R@1 (0.520).
My expectation, written now: a small extra gain at most, since studio photos are a different kind of photo from
street crops.

## Finding 44: re-ranker with extra studio training photos (evaluation/rerank_studio.py -> results/rerank_studio*.csv)
Check OK: the dev-only model reproduces Finding 43's test R@1 (0.5198).
Leakage check: 0 of 1,011 studio photos and 0 of 3,951 studio gallery pictures are identical to or near-duplicates of
(cosine >= 0.97) a streetlook test crop or test-item picture, so all 1,011 studio photos are kept.
Stage 1 is much easier on studio photos: the true item is in the top 20 for 92.7% of them (street dev: 76.4%).

| | dev CV R@1 | test R@1 | test R@5 | vs stage 1 (test) |
|---|---|---|---|---|
| stage 1 (finetuned_random) | 0.502 | 0.499 | 0.688 | |
| re-ranker, dev only (Finding 43) | 0.508 | 0.520 | | +0.021 [-0.004, +0.050] p 0.120, p_Holm 0.240 |
| re-ranker, dev + studio | 0.484 | 0.516 | 0.705 | +0.017 [-0.007, +0.046] p 0.226, p_Holm 0.240 |

dev + studio minus dev only: -0.004 [-0.033, +0.027] p 0.82.
What this means: NEGATIVE. Adding 1,011 studio photos doesn't help. On dev it makes the re-ranker worse than doing
nothing (0.484 vs 0.502), and on test it changes nothing (-0.4 points, n.s.). After Holm, neither re-ranker is
significant (p_Holm 0.24). The learning curve in Finding 43 rose with more STREET photos, and studio and flat-lay
photos are a different kind of photo (and an easier task), so they can't stand in for them. Conclusion for task 5: a
learned re-ranker on real photos gives at most about +2 points of R@1 here, which isn't significant with 500 street
photos. What it would need is more labelled street photos of the shop's own items (Finding 43's curve).

## Pre-registration: latency budget and the optional type model (task 6, serving/latency_cpu.py)
Budget, fixed now: at most 500 ms server time per search (median) on 4 CPU threads. The reasoning: about 1 s is the
limit for an answer to feel immediate, and the phone upload plus the page take roughly the other half.
Question: Finding 26 says about half the time goes to the second model (pretrained FashionSigLIP for the type badge,
the group bonus and the out-of-catalogue warning), but I stated that without measuring it. A new API switch,
DAYFIT_TYPE_MODEL=off, skips that model (no type badge, plain similarity order, no warning).
Measurement: the API's own search function, in-process on the Mac CPU (DAYFIT_DEVICE=cpu, 4 threads), 100 synthetic
medium photos after 5 warm-up, for (evaluation catalogue, on), (app catalogue, on) and (app, off). Reported: median and
95th percentile server time, and the ratio off / on. This isn't Docker: the Mac's CPU is faster than the container's,
so the absolute numbers are optimistic (Docker gave 266 ms on 4 vCPUs, Finding 26).
What switching it off costs has already been measured: group P@5 0.905 -> 0.871 on the test half (Finding 32), plus
losing the type badge and the out-of-catalogue warning.
Sanity check: (evaluation, on) reproduces the saved top 5 for at least 98% of photos.

## Finding 45: latency budget and the optional type model (serving/latency_cpu.py -> results/latency_cpu.csv)
Check OK: on the CPU, the evaluation catalogue gives the saved top 5 for 105/105 photos.
Server time per search, Mac CPU, 4 threads, 100 photos after 5 warm-up (median / 95th percentile):

| setting | median / p95 |
|---|---|
| evaluation catalogue (6,197 items), type model on | 124 / 125 ms |
| app catalogue (54,178 items), type model on | 129 / 131 ms |
| app catalogue (54,178 items), type model off | 73 / 74 ms |

What this means: (1) Every setting is well inside the 500 ms budget. The Docker container (4 vCPUs, slower) measured
266 ms with the type model on (Finding 26), which is also inside it. (2) The second model takes 56 ms of the 129
(44%), so Finding 26's "about half" was roughly right (off / on = 0.56). (3) Nine times more catalogue items cost 5 ms.
Decision: the type model stays ON by default. The app is inside the budget with it, and switching it off would lose
the type badge, the out-of-catalogue warning and 3.4 points of group P@5 (Finding 32) just to save ~56 ms.
DAYFIT_TYPE_MODEL=off stays available for slower machines.

## README check (task 6, evaluation/readme_check.py)
The script recomputes every README number that comes from a results file and checks that it appears verbatim in the
README (39 checks). It found two mismatches, and I fixed both: (1) the synthetic fine-tuning gain was given as +17.7
points, which belongs to the FIRST (hard-batch) model of Finding 4; for the recommended random-batch model it's +15.6
(0.949 vs 0.793, medium photos). (2) "confidence intervals are 4-5 points wide" was wrong: they're about +-4 to 5
points, so twice that wide. After the fixes there are 0 mismatches. Numbers that only exist in this log (Docker
266 ms, MPS 36 ms, the wide-leg jeans check, H&M label noise, the 500 / 481 split) are listed by the script so I can
check them by eye.

## D9 amendment 2 (Nov 19, 21:25, BEFORE any wardrobe photo is taken; only the sweater of the 13:20 rehearsal has been searched)
Reason: the app changed today after the "frozen since the H&M merge" note, but before any D9 photo existed:
(a) kids' items are now also hidden by product name (Finding 36), which changes which items can appear;
(b) the shortlist constant was re-calibrated (Finding 39), which only changes the shortlist size;
(c) the "colour can't be judged" flag (Finding 35) and the type-model switch (Finding 45, on by default) don't change
the ranking.
Decision: D9 tests the app at whatever commit it's run on (recorded with the result), i.e. with all of these changes.
Nothing else in the protocol changes. It's still the whole photo with no crop box, as pre-registered (Finding 37 now
shows that this is the harder setting).
My expectation, written now, based on Finding 37 (whole street photos, gallery, no bonus: group P@5 0.67-0.68): my own
photos, with the garment filling the frame, should be easier than street photos, so I expect the primary group P@5 to
land between 0.70 and 0.90. Whatever the value turns out to be, it gets reported.
## Finding 46: D9, my own wardrobe (evaluation/d9_own_photos.py -> results/d9_*.csv), run ONCE on Nov 23
Run at commit 051c560 (plus the labels), protocol of Nov 18 with amendments of Nov 19 00:55 and 21:25, labels
checked with --check before the run. 83 photos (one per garment, whole photo, no crop box, no editing): tops 51
(39 t-shirts, 9 sweaters, 2 hoodies, 1 shirt), bottoms 14, outerwear 6, footwear 4, accessories 7, other 1
(a wallet). No dresses or bags (I own none). All 83 photos loaded. The sanity check passed first (every
evaluation method reproduced its saved top 5; app catalogue 54,178 items, group bonus w = 0.05).

Primary results (95% bootstrap CI over photos):
- (1) app as shipped, group P@5: **0.985 [0.968, 0.998]**, n = 82 (all but the wallet). Per group: tops 0.980
  (51), bottoms 0.986 (14), outerwear 1.0 (6), footwear 1.0 (4), accessories 1.0 (7). Only 4 photos have any
  wrong-group result (two t-shirts, a pair of trousers, a sweater).
- (2) type P@5, finetuned_random, evaluation catalogue: **0.745 [0.668, 0.822]**, n = 65.
Descriptive, same 65 photos: fashion_siglip 0.711, ensemble_mv 0.689, color_hist 0.092 (fine-tuned minus
pretrained: +3.4 points on average, n~65, not tested). By type (finetuned_random): Jeans 1.0 (3), Shorts 1.0 (5),
Jackets 0.92 (5), Tshirts 0.75 (39), Sweatshirts 0.70 (2), Casual Shoes 0.55 (4), Shirts 0.40 (1), Trousers 0.32
(5). Trousers are mostly matched with Track Pants and casual shoes with Sports Shoes: the right group, the wrong
type. Two t-shirts (IMG_4236, IMG_4237) get heels and shoes as their top types on the evaluation catalogue.

App behaviour (descriptive):
- Group badge shown for 76 of 83 photos (92%) and right for all 76.
- Out-of-catalogue warning: fired once, on the wallet (the only item outside the catalogue), and on none of the
  82 garments the shop sells. (The summary file lists this as a count, 1, not a rate.)
- **"Not sure" fired on 39 of 83 photos (47%) and carries no information here**: group P@5 is 0.984 when it fires
  (38 garments) and 0.986 when it does not (44). It was calibrated on LookBench's gallery (Finding 23), whose
  similarity levels differ from the app catalogue's, so on the app it is far too cautious.
- Shortlist: median size 1,552 (range 67-8,806), so the page never shows it (it shows sizes <= 50). Its 90%
  guarantee can't be checked here (my clothes aren't in the catalogue), and at this size it isn't useful.

Against my written expectation (amendment 2: group P@5 between 0.70 and 0.90): the result is ABOVE the range.
Likely reason: my photos have one garment filling the frame on a plain surface or hanger, which is much closer to
a shop's product photo than a LookBench street photo with a person, background and other clothes. So D9 says the
app works well on clean photos of one garment; it does not contradict the low whole-street-photo numbers of
Finding 37.

Limits: one person's wardrobe, one session, one phone; 61% of the photos are tops (39 t-shirts); no dresses,
bags or skirts; the type labels and the photos are mine. Per-type numbers rest on 1-5 photos except t-shirts.
Nothing was changed after seeing these results.

Correction to Finding 46 (Nov 23, AFTER the run): IMG_4237.HEIC is not a garment. It was taken by mistake and
labelled as a t-shirt by mistake; I noticed while choosing examples for the figures, i.e. after seeing that it was
one of the lowest-scoring photos. The protocol says no photo is dropped after seeing results, so the primary
numbers above stay as run (82 / 65 photos). Without that photo (81 / 64 photos): app group P@5 0.990
[0.978, 1.0]; type P@5 finetuned_random 0.756 [0.675, 0.825], fashion_siglip 0.722, ensemble_mv 0.700,
color_hist 0.094; t-shirts 0.768 (38). "Not sure" then fires on 39 of 82 photos, with group P@5 0.984 when it
fires vs 0.995 when it does not, so the conclusion (it barely separates good from bad results) is unchanged.
Both versions are reported; the conclusions don't depend on which one is used. The example gallery
(report/make_final_figures.py) skips this photo, since it does not show a garment.
The photo is of my finger, which makes it an accidental test of a non-garment photo: the app returned blazers,
shirts, tops and a scarf, did NOT say "not sure", and did not give the out-of-catalogue warning (only the type
badge was held back, confidence < 0.9). The warning only knows clothing the shop doesn't sell, so it can't
recognise "not clothing at all"; that's a limitation of the app.
