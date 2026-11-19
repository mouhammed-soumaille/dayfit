# DAYFIT: find a garment from a photo

DAYFIT is a visual clothing search. A shopper uploads a photo of a garment they saw and cannot
describe; DAYFIT returns the 5 most similar items of a catalogue, says when it is not sure, and
explains each result with measured badges (type, colour). Course project for EECE 490 (Machine
Learning), American University of Beirut, Fall 2026.

## What the app shows

For one photo:

- **Top 5 catalogue items**, ranked by visual similarity (a FashionSigLIP model fine-tuned on our
  catalogue with random batches; see "Results"). With the app catalogue, the order also adds a small
  bonus for the kind of garment the photo looks like (`similarity + 0.05 * P(item's group | photo)`),
  the version tested on held-out real photos; kids' items are hidden and repeated photos of the same
  product are skipped.
- **Type badge** (`same type` / `different type`): the photo's garment group (tops, outerwear,
  bottoms, dresses, footwear; plus bags and accessories with the app catalogue) predicted zero-shot by the *pretrained* FashionSigLIP. Shown only when
  the prediction is at least 90% confident (then correct for 98-99% of our synthetic test photos).
- **Colour badge** and **estimated colour** (hex code + swatch): the median colour of the image
  centre in CIELAB, compared with CIEDE2000. It is an estimate from the photo; lighting shifts it.
- **"Not sure"** when the best match is weak, and a **shortlist size** ("about N items are as similar
  as the best match"), both calibrated on real street photos (LookBench). They come from conformal
  methods, but the guarantees do not formally hold in the app, because the app searches a different
  catalogue than the one they were calibrated on.
- **"This looks like a skirt; our catalogue has no skirts"** when the photo shows a kind of garment the
  catalogue does not sell (zero-shot check; detects 92% of such real photos, 4% false alarms).
- Two optional filters (`Same type only`, `Hide clearly different colours`) that **hide** results;
  they never change the ranking.
- An optional **crop box** ("Mark the garment"): two sliders mark the garment so that the background and
  other clothes do not count. The crop is made in memory; with the default box the photo is sent unchanged.

The badges **describe measured differences**; they are not the model's reasons. The ranking comes
from overall visual similarity.

## Two catalogues

| `DAYFIT_CATALOG` | Catalogue | Used for |
|---|---|---|
| unset (default) | `data/catalog.csv`: 6,197 items, 14 types | **every reported number** and the tests |
| `app` | `data/app_catalog.csv`: 33,141 adult items, 68 types (skirts, bags, hats, ...) | the demo |

The app catalogue is built from the full Fashion Product Images zip with `python -m data_prep.app_catalog`
and has its own index (`models/index/finetuned_random_app.npy`), badge vocabulary (72 prompts, 7 groups)
and out-of-catalogue threshold (`serving/outside_app.json`). Check which one is running with
`curl -s localhost:8000/health` (`"catalogue":"app","catalogue_items":33141`).

## Privacy

Uploaded photos are **never stored**. The API reads the photo into memory (not FastAPI's `UploadFile`,
which writes uploads over 1 MB to a temporary file), and nothing about the photo is logged.
`tests/test_api.py` checks that no file is written during a search, and in Docker the API container
runs with a **read-only file system**.

## Run it

### Locally (Mac with Apple Silicon, or any machine with Python 3.14)

```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
uvicorn serving.api:app --port 8000
```

For the demo catalogue use `DAYFIT_CATALOG=app uvicorn serving.api:app --port 8000`. If the page shows
old behaviour, an old server is probably still on port 8000: `lsof -ti :8000 | xargs kill -9`.

In a second terminal:

```
source .venv/bin/activate
streamlit run serving/ui.py
```

Then open http://localhost:8501. The app needs the catalogue (`data/catalog.csv`,
`data/images_hires/`), the fine-tuned checkpoint (`models/finetuned_random.pt`) and its saved index
(`models/index/finetuned_random*`); these are generated or downloaded files and are not in git (see
"Reproduce").

### With Docker (CPU)

```
docker compose up --build
```

API on http://localhost:8000/health, page on http://localhost:8501. The image is CPU-only; the first
build downloads the pretrained weights into the image. It contains both catalogues and serves the app
catalogue by default (`DAYFIT_CATALOG=evaluation docker compose up` for the evaluation one); the build
stops early if the saved app index does not match, instead of re-embedding 33k images on the CPU. On a Mac without Docker Desktop, Colima works
(`brew install colima docker-compose`, then `colima start --cpu 4 --memory 8`).

### API

```
curl -X POST --data-binary @photo.jpg http://localhost:8000/search
```

returns JSON with `results` (rank, item, score, type and colour badges, colour hex), `query`
(predicted group, estimated colour, out-of-catalogue warning), `not_sure`, `shortlist_size`,
`latency_ms` and a `note` with the limitations.

## Results (recommended model: fine-tuned FashionSigLIP, random batches)

| Test | Metric | Value |
|---|---|---|
| Synthetic phone photos of the catalogue (294 items), medium / strong | exact item first (Recall@1) | 0.949 / 0.813 |
| Same, medium / strong | right article type among the other 5 (P@5) | 0.867 / 0.822 |
| Real street photos (LookBench, 981 queries, duplicate-aware) | Recall@1 | 0.520 |
| Same | right kind of garment in the top 5 (exact names / groups) | 0.725 / 0.869 |
| Real photos, held-out test half (481 queries), with the group bonus | right kind of garment in the top 5 (groups) | **0.905** (+3.4 points, 95% CI +2.4 to +4.5) |
| Docker on CPU (4 vCPU), median / p95, evaluation / app catalogue | response time | 266 / 282 ms; 265 / 282 ms |
| Mac GPU (MPS), median | response time | 36 ms |

Main findings, each with its statistics in `report/results_log.md`:

- **Target reached on held-out real photos**: the right kind of garment is in the top 5 for 90.5% of
  photos with a small zero-shot group bonus (chosen on a development half, tested once; random ranking:
  21.4%). The exact item comes first for only about 51%: most misses are the right kind of garment with
  the wrong design. DINOv2 fusion and colour re-ranking did not help (negative results, kept).
- Fine-tuning helps on real photos: +3.4 points Recall@1 over the pretrained model (p = 0.001,
  clustered bootstrap), at the same speed; it ties the pretrained two-model ensemble at half the latency.
- Fine-tuning with same-type ("hard") batches lowered type precision; random batches fix it.
- Conformal shortlists keep their 90% guarantee on real photos only if calibrated on real photos.
- An ideal type-and-colour menu filter finds the exact item first in 3-5% of cases; the model in 52-95%.
- LookBench has duplicate pictures, overlapping category names and identical query crops; all are
  handled (duplicate-aware protocol, group-level P@5, clustered bootstrap), and the report says which
  conclusions they change.

## Limitations

- **Catalogue coverage and photo style.** The evaluation catalogue has 14 types; the app catalogue adds
  skirts, bags and hats, but its 2011-12 products have few modern cuts (no wide-leg jeans) and adult
  jeans are only photographed on a person, so a flat-lay photo can match kids' items. Similarity search
  can only rank what the shop sells.
- Most testing uses synthetic phone photos; real-photo results come from LookBench, which was also
  used to choose the model (only the later group bonus has a clean held-out test). The author's own
  photos are the final untouched test.
- The 90% target is coarse (7 garment groups); finding the exact item on real photos is about 51%.
- **Licence.** Fashion Product Images and LookBench are for non-commercial research: this is a research
  demo and cannot be sold with these images.
- The confidence signal behind "not sure" (top similarity) is weak: it lowers the error rate mostly
  by answering less often, not by recognising the bad cases.
- The colour estimate uses the image centre and can pick up a second garment or a shadow.

## Repository layout

```
data_prep/    build the catalogue, high-resolution images, synthetic phone photos
retrieval/    colour histogram, CLIP / FashionSigLIP retrievers, ensembles
training/     fine-tuning (contrastive, hard or random batches)
evaluation/   metrics, bootstrap tests, LookBench, conformal methods, analyses
serving/      search engine, calibration, badges, API, Streamlit page, latency
tests/        unit tests (metrics, statistics, conformal) and API tests
results/      saved result tables (CSV)
report/       results log and figures
```

## Reproduce

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
python -m evaluation.lookbench_split
python -m evaluation.export_lookbench_features
python -m evaluation.fusion_lookbench --side dev
python -m evaluation.fusion_test_stats
python -m pytest
```

Data: Kaggle "Fashion Product Images" (catalogue) and LookBench (`srpone/look-bench` on Hugging Face,
real street photos). Pretrained models: Marqo FashionSigLIP and FashionCLIP; DINOv2 (tested, not used). Tuning on real photos
uses a frozen development/test split of LookBench (`evaluation/lookbench_split.py`); `evaluation/fusion_test_stats.py`
needs the test runs saved first (`fusion_lookbench.py --side test ... --save NAME`, see `report/results_log.md`,
Finding 32).
