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

App catalogue (DAYFIT_CATALOG=app): there are no saved evaluation results for it, so first save a
reference top 5 from the Mac's own API (app catalogue, MPS), then compare the container against it:
    python -m serving.latency --save_reference results/latency_reference_app.csv   (local app API running)
    python -m serving.latency --reference results/latency_reference_app.csv        (Docker running)
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
    parser.add_argument("--reference", help="CSV of expected top 5 (query_id, rank, item_id)")
    parser.add_argument("--save_reference", help="save the API's top 5 here instead of checking")
    args = parser.parse_args()

    queries = pd.read_csv("data/queries.csv", dtype={"source_item_id": str})
    photos = queries[queries["condition"] == "phone_medium"].head(args.n + WARMUP)
    health = requests.get(f"{args.api}/health", timeout=10).json()
    print(health)
    if args.reference:
        saved = pd.read_csv(args.reference, dtype={"item_id": str})
    else:
        saved = pd.read_csv("results/finetuned_random_phone_medium.csv", dtype={"item_id": str})
        if health.get("catalogue") != "evaluation" and not args.save_reference:
            raise SystemExit("app catalogue: use --save_reference (local API) or --reference (Docker)")
    found_rows = []

    server, total, same = [], [], []
    for n, q in enumerate(photos.itertuples()):
        with open(q.image_path, "rb") as f:
            body = f.read()
        start = time.perf_counter()
        r = requests.post(f"{args.api}/search", data=body, timeout=120)
        elapsed = (time.perf_counter() - start) * 1000
        r.raise_for_status()
        answer = r.json()
        got = [x["item_id"] for x in answer["results"]]
        found_rows += [dict(query_id=q.query_id, rank=k + 1, item_id=i) for k, i in enumerate(got)]
        expected = list(saved[saved["query_id"] == q.query_id].sort_values("rank")["item_id"])
        same.append(got == expected)
        if n >= WARMUP:
            server.append(answer["latency_ms"])
            total.append(elapsed)
        print(f"  {n + 1}/{len(photos)}", end="\r")

    if args.save_reference:
        pd.DataFrame(found_rows).to_csv(args.save_reference, index=False)
        print(f"\n  saved the reference top 5 to {args.save_reference} (no check in this mode)")
    else:
        print(f"\n  check: top 5 equals the reference for {sum(same)}/{len(same)} photos")
    for name, values in [("server (one search)", server), ("total round trip", total)]:
        v = np.array(values)
        print(f"  {name:20s} median {np.median(v):6.0f} ms   p95 {np.percentile(v, 95):6.0f} ms   "
              f"max {v.max():6.0f} ms   (n={len(v)})")


if __name__ == "__main__":
    main()
