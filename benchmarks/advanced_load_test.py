import time
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

API_URL = "http://127.0.0.1:8000/predict_raw"
X_TEST_PATH = "X_test.parquet"
OUT = os.path.join("results", "advanced_api_load_test.json")
os.makedirs("results", exist_ok=True)

def main():
    # load sample rows
    X = pd.read_parquet(X_TEST_PATH)
    sample = X.sample(n=2000, random_state=42)
    rows = sample.values.tolist()

    # params
    concurrency = 32
    total_requests = 2000
    batch_size = 1  # each request sends one row
    timeout = 15

    results = {"concurrency": concurrency, "total_requests": total_requests, "requests": []}

    def post_one(row):
        payload = [row]
        t0 = time.perf_counter()
        try:
            r = requests.post(API_URL, json=payload, timeout=timeout)
            status = r.status_code
        except Exception:
            status = None
        t1 = time.perf_counter()
        return status, t1 - t0

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(post_one, rows[i % len(rows)]) for i in range(total_requests)]
        for f in as_completed(futures):
            try:
                status, latency = f.result()
            except Exception:
                status, latency = None, None
            results["requests"].append({"status": status, "latency": latency})

    # summarize
    good = [r["latency"] for r in results["requests"] if r["latency"] is not None]
    success = sum(1 for r in results["requests"] if r["status"] == 200)
    throughput = success / sum(good) if good and sum(good) > 0 else None
    import numpy as np
    summary = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "success_count": success,
        "throughput_reqs_per_sec": throughput,
        "avg_latency_sec": float(np.mean(good)) if good else None,
        "p95_latency_sec": float(np.percentile(good, 95)) if good else None
    }

    out = {"summary": summary, "requests_sample": results["requests"][:200]}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print('Advanced load test complete, saved to', OUT)
    print(summary)


if __name__ == "__main__":
    main()
