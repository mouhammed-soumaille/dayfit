# DAYFIT: find a garment from a photo

You see a jacket on someone in the street, or a pair of jeans in a photo, and you can't describe it well
enough to search for it. DAYFIT lets you search with the photo instead: upload it, and you get the five
most similar items from a shop's catalogue. The app also tells you when it isn't sure, and labels each
result with what it measured (kind of garment, colour) so you can judge the matches yourself.

Course project for EECE 490 (Machine Learning), American University of Beirut, Fall 2026. I did the work
alone; `report/results_log.md` is my lab notebook.

**Who it is for.** The user is an online shopper; the system would run as a "search by photo" feature on
one shop's site, so it can only suggest what that shop sells. **How it is done today:** on a shop's own
site, with text search and menu filters (garment type, colour), which needs the shopper to name what they
see; outside the shop, with general photo-search tools such as Google Lens, which search the whole web
rather than one shop's stock. I compare against keyword search, menu filters and a classical
(non-deep-learning) image search; I did not compare against general photo-search tools.

![The app on a photo of a sweater, whole photo](report/figures/demo_sweater_worn.png)
![The same photo with the crop box around the sweater](report/figures/demo_sweater_crop.png)
*My own sweater, which is not in the catalogue. Top: the whole photo. Bottom: the same photo with the
garment marked (only the red box is searched). Both find similar brown knit sweaters and put the same one
first; marking the garment makes the matches tighter.*

## Results on real photos

All numbers below are on **real street photos** (LookBench `real_streetlook`) that were never used to choose
anything: the 481 photos of the held-out test half (a 500 / 481 split fixed in advance, photos of the same
street picture or item kept on one side). Each method was tuned only on the other half and run once on
these photos. The main question is the hard one the app exists for: **is the exact item first?**

**Against the benchmark's official corpus of 61,553 images** (the 3,278 shop items + 58,275 distractors):

| Method | Exact item first (R@1) | Exact item in top 5 (R@5) |
|---|---|---|
| Colour histogram (background masked) | 0.081 | 0.123 |
| Classical image search: colour + texture (LBP) + shape (HOG), weights tuned on the other half | 0.123 | 0.162 |
| Pretrained FashionSigLIP, no training | 0.476 | 0.663 |
| **Fine-tuned FashionSigLIP (the app's model)** | **0.499** | **0.688** |
| Fine-tuned + learned re-ranker trained on 500 real photos | 0.520 (+2.1, not significant, p = 0.12) | 0.709 |

**Against the 3,278 shop items only** (needed for anything that uses item text or type labels, which the
distractors do not have):

| Method | Exact item first (R@1) | Right kind of garment in top 5 (group P@5) |
|---|---|---|
| Classical image search + an ideal type menu | 0.237 | 1.0 (by construction) |
| Keyword search, short ("bag backpack"), BM25 | 0.312 | 0.983 |
| Keyword search, every attribute in the shop's own words (idealised shopper) | 0.486 | 0.986 |
| **Fine-tuned FashionSigLIP (photo only)** | **0.511** | **0.871** |
| Fine-tuned + an ideal type menu | 0.536 | 1.0 (by construction) |
| Fine-tuned + group bonus (the app's ranking) | 0.509 | 0.905 |

What these show:

- **A photo beats every realistic alternative that does not need the shopper to describe the garment.** The
  photo model finds the exact item first 4 times as often as a classical image search (0.50 vs 0.12) and
  clearly more often than a short typed search (0.51 vs 0.31). Only an idealised shopper who types every
  attribute in the shop's own words ties it (0.49 vs 0.51, no significant difference); real shoppers cannot
  do that, which is the problem the app solves.
- **"Right kind of garment in the top 5" is a sanity floor, not a headline.** A typed category gets 0.98 and
  a menu gets 1.0 by construction; the group bonus raises the photo model from 0.87 to 0.905 by adding a
  zero-shot type classifier's probability, so it optimises this metric fairly directly.
- **The fine-tuning's real value is robustness to blur.** On clean real photos it adds about +2 to +3 points
  over the pretrained model. On blurred real photos it adds +6 to +21 points, including motion blur, which it
  never saw in training (table below). For colour changes it never saw (black-and-white, coloured light) it
  adds nothing.
- **A learned re-ranker on real photos does not help significantly yet.** Trained on 500 labelled street
  photos with 9 features (similarities of three models, small-patch matching, colour, garment-type agreement),
  it gains +2.1 points, not significant. Its learning curve is still rising at 500 photos (50 photos: worse
  than no re-ranker); adding 1,011 studio photos did not help. Most misses are the right kind of garment with
  the wrong design, and the re-ranker mostly re-weighs the two models' similarities.
- **A 19 times bigger corpus costs only about 2 points** (0.520 -> 0.501 on all 981 photos): the distractors
  rarely beat the true item.

### The app's real setting: whole photos and bad photos

LookBench photos are hand-drawn crops of the garment. **The app sends the whole photo unless you mark the
garment**, and that is much harder (official corpus, test half):

| Photo sent | Exact item first (R@1) | Exact item in top 5 |
|---|---|---|
| Garment marked (crop) | 0.499 | 0.688 |
| Whole photo (person, background, other clothes) | 0.391 | 0.530 |

Marking the garment matters more than any model change I tried, so the page now recommends it.

Bad photos (all 981 real crops with fixed filters; * = outside the training augmentation):

| Filter | Fine-tuned R@1 | Pretrained R@1 | Fine-tuned minus pretrained |
|---|---|---|---|
| none | 0.501 | 0.472 | +2.9 |
| blur, radius 1.5 / 3 / 6* px | 0.490 / 0.355 / 0.118 | 0.389 / 0.147 / 0.022 | +10.1 / +20.8 / +9.6 |
| motion blur 5* / 9* / 15* px | 0.501 / 0.454 / 0.368 | 0.444 / 0.368 / 0.244 | +5.6 / +8.6 / +12.4 |
| brightness x0.5 / x0.3* / x0.15* | 0.501 / 0.490 / 0.391 | 0.461 / 0.451 / 0.358 | +4.0 / +4.0 / +3.4 |
| grayscale* | 0.379 | 0.388 | -0.9 (n.s.) |
| warm colour cast, mild* / strong* | 0.485 / 0.452 | 0.456 / 0.432 | +3.0 / +1.9 (n.s.) |

![Real crops with filters: R@1 per filter for the fine-tuned model, the pretrained model and a colour histogram](report/figures/robustness.png)

## What the app shows

For each photo:

- **The 5 closest catalogue items.** The ranking comes from FashionSigLIP fine-tuned on the catalogue, plus a
  small bonus for items of the kind of garment the photo seems to show
  (`similarity + 0.05 * P(item's group | photo)`, tested on held-out real photos). Kids' items are hidden,
  and the same product photographed twice is shown only once.
- **A type badge** (`same type` / `different type`): which group the photo belongs to (tops, outerwear,
  bottoms, dresses, footwear, bags, accessories), guessed by the pretrained FashionSigLIP. It only appears
  when the guess is at least 90% confident; on real photos it then appears for 65% of photos and is right
  95% of the time.
- **A colour badge** and the photo's estimated colour, next to the shop's own colour name ("listed as").
  The estimate comes from the photo, so lighting moves it. When the photo has almost no colour (a
  black-and-white photo, or a black, white or grey garment) no colour badges are shown.
- **"Not sure"** when even the best match is weak, and a **shortlist size** (how many items are about as
  similar as the best one, a 90% conformal shortlist calibrated on real photos at the official-corpus scale).
  The page shows the size only when it is 50 or fewer; for a marked garment the honest 90% list is about 116
  items long, for a whole photo about 880.
- **"This looks like a skirt; our catalogue has no skirts"** when the photo shows something the shop does
  not sell at all.
- Two optional filters, *same type only* and *hide clearly different colours*. They hide results; they
  never change the order.
- A **crop box** ("Mark the garment", recommended), done in memory.

The badges describe differences I measured; they are not the model's reasons. The order comes from
overall visual similarity.

## Two catalogues

| `DAYFIT_CATALOG` | Catalogue | Used for |
|---|---|---|
| unset (default) | `data/catalog.csv`: 6,197 items, 14 types | the synthetic-photo results and the tests |
| `app` | `data/app_catalog.csv`: **54,178 items**, 68 types | the demo |

The evaluation catalogue is Kaggle's Fashion Product Images. For the demo I wanted something closer to a real
shop, so the app catalogue adds 26,944 more adult items from the same dataset and 21,037 newer H&M products
(2018-20). Each H&M product type was mapped to one of the existing types before anything was run, so the
badges and the ranking bonus did not have to be recalibrated; the page still shows H&M's own name for the
product. The app hides 809 kids' items (by the shop's gender field and by product name, Finding 36).
`curl -s localhost:8000/health` tells you which catalogue is running.

## Privacy

Uploaded photos are **never stored**. The API reads the photo straight into memory (FastAPI's usual
`UploadFile` would write anything over 1 MB to a temporary file) and logs nothing about it.
`tests/test_api.py` checks that a search writes no file, and in Docker the API runs on a read-only file
system.

## Run it

### On your machine (Mac with Apple Silicon, or anything with Python 3.14)

```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
DAYFIT_CATALOG=app uvicorn serving.api:app --port 8000
```

Without `DAYFIT_CATALOG=app` you get the evaluation catalogue. If the page behaves like an older version,
an old server is probably still running on port 8000: `lsof -ti :8000 | xargs kill -9`.

Then, in a second terminal:

```
source .venv/bin/activate
streamlit run serving/ui.py
```

and open http://localhost:8501. You need the catalogues and their images, the fine-tuned checkpoint
(`models/finetuned_random.pt`) and the saved indexes (`models/index/finetuned_random*`). These are
generated or downloaded, not stored in git (see "Reproduce").

### With Docker (CPU only)

```
docker compose up --build
```

The API is on http://localhost:8000/health and the page on http://localhost:8501. The image contains
both catalogues and serves the demo one (`DAYFIT_CATALOG=evaluation docker compose up` for the other).
The build stops early if the saved demo index does not match the catalogue, rather than spending a long
time re-embedding 54k images on the CPU. On a Mac without Docker Desktop, Colima works
(`brew install colima docker-compose`, then `colima start --cpu 4 --memory 8`). The image is built from
local files, so it currently needs the reproduction steps first.

### API

```
curl -X POST --data-binary @photo.jpg http://localhost:8000/search
```

returns JSON with the `results` (rank, item, score, type and colour badges, colour), the `query`
(predicted group, estimated colour or `colour_judged: false`, out-of-catalogue warning), `not_sure`,
`shortlist_size`, `latency_ms` and a `note` on the limitations.

**Response time.** Budget: at most 500 ms of server time per search on a 4-core CPU (about 1 s is the limit
for an answer to feel immediate, and the phone upload and page take roughly the other half). Measured: 266 ms
median in Docker on 4 vCPUs (Findings 26, 33); on the Mac's CPU with 4 threads 129 ms for the 54k-item app
catalogue, of which 56 ms is the second (type) model, and 36 ms on the Mac GPU (Finding 45). For slower machines
`DAYFIT_TYPE_MODEL=off` skips the type model (no type badge, no group bonus, no out-of-catalogue warning).

## Synthetic phone photos (secondary result)

Before real photos were available I built synthetic "phone photos" of the evaluation catalogue (random
background, rotation, crop, lighting, blur, JPEG). The fine-tuned model was **trained on the same kind of
synthetic photo**, so these numbers mostly measure how well it learned that augmentation: its gain over the
pretrained model is +15.6 points here (0.949 vs 0.793 on medium photos) and about +2 to +3 on clean real
photos (Findings 7, 38).

| Test (294 items, medium / strong photos) | Value |
|---|---|
| Exact item first (R@1) | 0.949 / 0.813 |
| Right article type among the other 5 (P@5) | 0.867 / 0.822 |

![Exact-item Recall@1 for each step of the pipeline on synthetic phone photos](report/figures/fig1_ladder.png)

Other first-phase findings that still hold: same-type batches during fine-tuning made the model worse at
telling types apart and random batches fixed it (Findings 7-8); the conformal shortlist only keeps its 90%
promise on real photos if it is calibrated on real photos (Finding 10).

## Limitations

- The app can only return what the catalogue sells, so for many photos the best answer is "similar", not
  "the same". The exact item comes first about half the time on real crops, 39% on whole photos.
- The real-photo test is one benchmark (LookBench street photos, 981 photos, 621 independent groups); the
  95% confidence intervals are about +-4 to 5 points. My own wardrobe test (D9) was planned with fixed rules but not run.
- Very blurry photos break every model (strong blur: R@1 0.12), and "not sure" catches few of them (it fires
  on 19% of such photos). Very dark photos often get a wrong "colour different" badge (27-33% for the true
  item even after the black-and-white fix).
- The keyword-search comparison is idealised (the shopper's words are the benchmark's own annotation) and runs
  on the 3,278 shop items only; in a bigger shop many items would share the same words.
- The learned re-ranker would need the candidates' small-patch features stored with the catalogue (about
  21 GB for 54k items), so it is not in the app.
- **Catalogue coverage and bias.** The base catalogue is Fashion Product Images (Indian e-commerce, 2011-12)
  and leans towards men's and older styles (none of its jeans are wide-leg, Finding 30). Women's and current
  styles depend mostly on the H&M items. H&M's own type labels are noisy (parkas filed as hoodies, jumpsuits
  as trousers, Finding 36), and some catalogue images are fabric close-ups or show the wrong product; I could
  not remove those reliably.
- Kids' items are hidden, not detected: a photo of a kids' garment gets the most similar adult items.
- The colour estimate looks at the middle of the photo, so with a whole photo it can pick up the background.
- **Licence.** Fashion Product Images, LookBench and the H&M data are for non-commercial research only.
  This is a research demo and cannot be sold with these images.

## Repository layout

```
data_prep/    build the catalogues (evaluation, app, H&M), synthetic phone photos, app clean-up
retrieval/    colour histogram, CLIP / FashionSigLIP retrievers, ensembles
training/     fine-tuning (contrastive, same-type or random batches)
evaluation/   metrics, bootstrap tests, LookBench (official corpus, whole photos, bad photos), baselines,
              re-ranker, conformal methods, analyses, own-photo test (D9)
serving/      search engine, calibration, badges, colour check, API, Streamlit page, latency
tests/        unit tests (metrics, statistics, conformal) and API tests
results/      saved result tables (CSV)
report/       results log and figures
```

## Reproduce

What is in git: code, saved result tables (`results/*.csv`), figures and the LookBench dev/test split.
What is **not** in git (too big, or licensed): the datasets, their images, the fine-tuned checkpoint and
the saved indexes. Everything below regenerates them.

1. **Data.**
   - Kaggle "Fashion Product Images (Small)": unpack `styles.csv` and `images/` into `data/raw/`.
   - Kaggle "Fashion Product Images Dataset" (full, high-resolution): save the zip as
     `data/raw_full/fashion-product-images-dataset.zip`.
   - LookBench: downloaded automatically from Hugging Face (`srpone/look-bench`); needs the evaluation
     packages: `pip install -r requirements-dev.txt`.
   - H&M (demo catalogue only): accept the Kaggle competition rules, then save the zip as
     `data/hm/h-and-m-personalized-fashion-recommendations.zip`.
2. **Model, catalogues and app** (in this order):

```
python -m data_prep.build_catalog
python -m data_prep.hires_catalog --zip data/raw_full/fashion-product-images-dataset.zip
python -m data_prep.make_queries
python -m training.finetune --hard_batches 0 --out models/finetuned_random.pt --log models/finetune_random_log.csv
python -m evaluation.run_eval --method finetuned_random
python -m evaluation.run_lookbench --subset real_streetlook --method finetuned_random
python -m serving.calibrate
python -m serving.badges
python -m serving.outside_check
python -m data_prep.app_catalog
python -m serving.app_badges_check
python -m serving.type_rerank
python -m data_prep.hm_catalog --from_zip data/hm/h-and-m-personalized-fashion-recommendations.zip
python -m data_prep.app_add_hm
python -m serving.colour_check
python -m data_prep.app_cleanup
```

3. **Real-photo experiments** (each script checks itself against a saved number and stops if it does not
   match; the test half is used once per method, see the pre-registrations in `report/results_log.md`):

```
python -m evaluation.lookbench_split
python -m evaluation.export_lookbench_features
python -m evaluation.fusion_lookbench --side dev
python -m evaluation.fusion_lookbench --side test --a 0   --w 0                 --save base
python -m evaluation.fusion_lookbench --side test --a 0   --w 0    --beta 0.02  --save colour
python -m evaluation.fusion_lookbench --side test --a 0.1 --w 0                 --save dinov2
python -m evaluation.fusion_lookbench --side test --a 0   --w 0.05              --save groupbonus
python -m evaluation.fusion_lookbench --side test --a 0.1 --w 0.05 --beta 0.02  --save combined
python -m evaluation.fusion_test_stats
python -m evaluation.run_lookbench --subset real_streetlook --method fashion_siglip
python -m evaluation.run_lookbench --subset real_streetlook --method finetuned_random --query_image raw_image
python -m evaluation.run_lookbench --subset real_streetlook --method fashion_siglip --query_image raw_image
python -m evaluation.whole_photo
python -m evaluation.lookbench_full
python -m serving.calibrate_full
python -m evaluation.robustness
python -m evaluation.keyword_baseline
python -m evaluation.classical_baseline
python -m evaluation.rerank_features
python -m evaluation.rerank_model
python -m evaluation.rerank_studio
python -m pytest
```

The H&M items are added after the badges, the out-of-catalogue threshold and the bonus were fixed; those
are not recomputed on the merged catalogue.

Data: Kaggle "Fashion Product Images" (catalogue), LookBench (`srpone/look-bench` on Hugging Face, real
photos and the 58,275 distractor images) and Kaggle "H&M Personalized Fashion Recommendations" (demo
catalogue only). Pretrained models: Marqo FashionSigLIP, FashionCLIP and DINOv2.

## Documents

- `report/results_log.md`: the full lab notebook, with a one-page index at the top, every pre-registration
  and every result, including corrections and negative results. Later findings can overrule earlier ones;
  this README is the current summary.
