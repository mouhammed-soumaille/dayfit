"""Latency budget check on the CPU, with and without the second (type) model (Finding 45; pre-registered).

Budget (stated before measuring): a search must take at most 500 ms of server time, median, on 4 CPU threads.
Reason: a shopper waits about 1 second before an answer stops feeling immediate; with a phone upload and the
page taking the other half, the server gets half of that second.

Runs the API's own search function in-process (no HTTP, no upload), on the CPU, 4 threads, over synthetic
medium photos (WARMUP not counted). The settings come from environment variables, as for the API:
    DAYFIT_DEVICE=cpu                      forces the CPU (otherwise the Mac would use its GPU)
    DAYFIT_CATALOG=app | evaluation        which catalogue
    DAYFIT_TYPE_MODEL=on | off             with or without the second model (type badge, bonus, warning)
Sanity check (evaluation catalogue with the type model on): the top 5 must equal the saved evaluation
results for at least 98% of photos (CPU and GPU can swap two almost equal items); stops otherwise.

Run from the repo root, once per setting (each run appends one row to results/latency_cpu.csv):
    DAYFIT_DEVICE=cpu DAYFIT_CATALOG=evaluation DAYFIT_TYPE_MODEL=on python -m serving.latency_cpu
    DAYFIT_DEVICE=cpu DAYFIT_CATALOG=app DAYFIT_TYPE_MODEL=on python -m serving.latency_cpu
    DAYFIT_DEVICE=cpu DAYFIT_CATALOG=app DAYFIT_TYPE_MODEL=off python -m serving.latency_cpu
"""
import os

import torch

torch.set_num_threads(4)                       # like the 4-core Docker container

import numpy as np
import pandas as pd
from PIL import Image

from serving import api

BUDGET_MS = 500
N, WARMUP = 100, 5
OUT = "results/latency_cpu.csv"


def main():
    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    photos = queries[queries["condition"] == "phone_medium"].head(N + WARMUP)
    saved = pd.read_csv("results/finetuned_random_phone_medium.csv", dtype={"item_id": str})
    times, same = [], []
    for n, q in enumerate(photos.itertuples()):
        answer = api.search_photo(Image.open(q.image_path).convert("RGB"))
        got = [r["item_id"] for r in answer["results"]]
        expected = list(saved[saved["query_id"] == q.query_id].sort_values("rank")["item_id"])
        same.append(got == expected)
        if n >= WARMUP:
            times.append(answer["latency_ms"])
        print(f"  {n + 1}/{len(photos)}", end="\r")
    print()
    catalogue, type_model = api.engine.catalogue, "on" if api.types is not None else "off"
    if catalogue == "evaluation" and type_model == "on":
        print(f"  check: top 5 equals the saved evaluation results for {sum(same)}/{len(same)} photos")
        if np.mean(same) < 0.98:
            raise SystemExit("PROBLEM: the CPU search does not reproduce the evaluated model; stopping.")
    t = np.array(times)
    row = dict(catalogue=catalogue, items=len(api.engine.item), type_model=type_model,
               device=os.environ.get("DAYFIT_DEVICE", "default"), threads=torch.get_num_threads(), n=len(t),
               median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), budget_ms=BUDGET_MS,
               within_budget=bool(np.median(t) <= BUDGET_MS))
    print(row)
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(OUT, index=False)
    print(f"appended to {OUT}")


if __name__ == "__main__":
    main()
