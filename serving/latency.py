"""E7: response time of the deployed app (the approved constraint), measured against the API in Docker (CPU).

Sends synthetic phone photos (medium) to a running API, one at a time, as a shopper's browser
would, and records two times per request:
  server_ms : time inside the API for one search (embedding + similarity + type badge + colours)
  total_ms  : full round trip seen by the client (upload + decode + search + JSON)
The first WARMUP requests are not counted (the first calls are slower while things load).

Sanity check: for every photo, the API's top 5 must equal the saved evaluation results, so we
know the container serves exactly the evaluated model (CPU vs MPS can change scores very slightly,
so we compare the items, not the scores).

Run from the repo root on the Mac (.venv), with the containers running (docker compose up):
    python -m serving.latency
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

WARMUP = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    photos = queries[queries["condition"] == "phone_medium"].head(args.n + WARMUP)
    saved = pd.read_csv("results/finetuned_random_phone_medium.csv", dtype={"item_id": str})
    print(requests.get(f"{args.api}/health", timeout=10).json())

    server, total, same = [], [], []
    for n, q in enumerate(photos.itertuples()):
        with open(q.image_path, "rb") as f:
            body = f.read()
        start = time.perf_counter()
        r = requests.post(f"{args.api}/search", data=body, timeout=120)
        elapsed = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        answer = r.json()
        expected = list(saved[saved["query_id"] == q.query_id].sort_values("rank")["item_id"])
        same.append([x["item_id"] for x in answer["results"]] == expected)
        if n >= WARMUP:
            server.append(answer["latency_ms"])
            total.append(elapsed)
        print(f"  {n + 1}/{len(photos)}", end="\r")

    print(f"\n  check: top 5 equals the saved evaluation results for {sum(same)}/{len(same)} photos")
    for name, values in [("server (one search)", server), ("total round trip", total)]:
        v = np.array(values)
        print(f"  {name:20s} median {np.median(v):6.0f} ms   p95 {np.percentile(v, 95):6.0f} ms   "
              f"max {v.max():6.0f} ms   (n={len(v)})")


if __name__ == "__main__":
    main()
