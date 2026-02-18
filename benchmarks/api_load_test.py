import time
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

API_URL = "http://127.0.0.1:8000/predict_raw"
X_TEST_PATH = "X_test.parquet"
OUT = os.path.join("results", "api_load_test.json")
os.makedirs("results", exist_ok=True)

# load a small sample to send as payload rows
X = pd.read_parquet(X_TEST_PATH)
SAMPLE = X.sample(n=1000, random_state=42)
rows = SAMPLE.values.tolist()

concurrencies = [1, 8, 32, 64]
requests_per_level = 500  # total requests per concurrency level

results = []

def post_once(payload):
    t0 = time.perf_counter()
    r = requests.post(API_URL, json=payload, timeout=10)
    t1 = time.perf_counter()
    return r.status_code, t1 - t0

for c in concurrencies:
    latencies = []
    statuses = []
    total = requests_per_level
    with ThreadPoolExecutor(max_workers=c) as ex:
        futures = []
        for i in range(total):
            # send a single row each request, rotate sample
            row = rows[i % len(rows)]
            futures.append(ex.submit(post_once, [row]))
        for f in as_completed(futures):
            try:
                status, lat = f.result()
            except Exception as e:
                statuses.append(0)
                latencies.append(None)
            else:
                statuses.append(status)
                latencies.append(lat)
    good = [l for l in latencies if l is not None]
    success = sum(1 for s in statuses if s == 200)
    total_requests = len(statuses)
    throughput = success / sum(good) if sum(good) > 0 else None
    results.append({
        "concurrency": c,
        "total_requests": total_requests,
        "success_count": success,
        "throughput_reqs_per_sec": throughput,
        "avg_latency_sec": float(sum(good) / len(good)) if good else None,
        "p95_latency_sec": float(pd.Series(good).quantile(0.95)) if good else None
    })
    print(results[-1])

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"API load test results saved to {OUT}")
