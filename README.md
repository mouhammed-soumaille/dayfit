# DAYFIT: find a garment from a photo

DAYFIT is a visual clothing search. A shopper uploads a photo of a garment they saw and cannot
describe; DAYFIT returns the 5 most similar items of a catalogue, says when it is not sure, and
explains each result with measured badges (type, colour). Course project for EECE 490 (Machine
Learning), American University of Beirut, Fall 2026.

## What the app shows

For one photo:

- **Top 5 catalogue items**, ranked by visual similarity (a FashionSigLIP model fine-tuned on our
  catalogue with random batches; see "Results").
- **Type badge** (`same type` / `different type`): the photo's garment group (tops, outerwear,
  bottoms, dresses, footwear) predicted zero-shot by the *pretrained* FashionSigLIP. Shown only when
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

The badges **describe measured differences**; they are not the model's reasons. The ranking comes
from overall visual similarity.

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
build downloads the pretrained weights into the image. On a Mac without Docker Desktop, Colima works
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
| Docker on CPU (4 vCPU), median / p95 | response time | 266 / 282 ms |
| Mac GPU (MPS), median | response time | 36 ms |

Main findings, each with its statistics in `report/results_log.md`:

- Fine-tuning helps on real photos: +3.4 points Recall@1 over the pretrained model (p = 0.001,
  clustered bootstrap), at the same speed; it ties the pretrained two-model ensemble at half the latency.
- Fine-tuning with same-type ("hard") batches lowered type precision; random batches fix it.
- Conformal shortlists keep their 90% guarantee on real photos only if calibrated on real photos.
- An ideal type-and-colour menu filter finds the exact item first in 3-5% of cases; the model in 52-95%.
- LookBench has duplicate pictures, overlapping category names and identical query crops; all are
  handled (duplicate-aware protocol, group-level P@5, clustered bootstrap), and the report says which
  conclusions they change.

## Limitations

- **Catalogue coverage.** The catalogue has 14 types and no skirts, bags, hats or jumpsuits, and few
  modern cuts (for example wide-leg jeans). For such photos the app can only give loose suggestions;
  it now says so.
- Most testing uses synthetic phone photos; real-photo results come from LookBench, which was also
  used to choose the model. A small set of the author's own photos is the final untouched test.
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
python -m pytest
```

Data: Kaggle "Fashion Product Images" (catalogue) and LookBench (`srpone/look-bench` on Hugging Face,
real street photos). Pretrained models: Marqo FashionSigLIP and FashionCLIP.
