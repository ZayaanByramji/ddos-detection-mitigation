import time
import numpy as np
import pandas as pd
import joblib
import os
import argparse

MODEL_PATH = "models/final_random_forest_model.joblib"
SCALER_PATH = "models/scaler.joblib"
X_TEST_PATH = "X_test.parquet"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# CLI
parser = argparse.ArgumentParser(description="Benchmark model inference")
parser.add_argument("--samples", type=int, default=2000, help="number of samples to benchmark (default: 2000)")
args = parser.parse_args()

# Use a sample of test set for benchmarking
X = pd.read_parquet(X_TEST_PATH)
sample_n = min(args.samples, len(X))
sample = X.sample(n=sample_n, random_state=42)
Xs = scaler.transform(sample)

try:
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
except Exception:
    pass
batch_sizes = [1, 8, 32, 128, 512]
results = []

import joblib as _joblib

for batch in batch_sizes:
    iterations = int(np.ceil(len(Xs) / batch))
    timings = []
    for i in range(iterations):
        start = i * batch
        end = min((i + 1) * batch, len(Xs))
        batch_data = Xs[start:end]
        t0 = time.perf_counter()
        # use predict (faster) and force single-threaded backend to avoid overhead
        with _joblib.parallel_backend("loky", n_jobs=1):
            _ = model.predict(batch_data)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    total_time = sum(timings)
    total_samples = len(Xs)
    throughput = total_samples / total_time if total_time > 0 else float('inf')
    avg_latency = np.mean(timings)
    p95 = np.percentile(timings, 95)
    results.append({
        "batch_size": batch,
        "total_samples": total_samples,
        "total_time_sec": total_time,
        "throughput_samples_per_sec": throughput,
        "avg_latency_sec": avg_latency,
        "p95_latency_sec": p95
    })

OUT = os.path.join("results", "benchmark_inference.json")
import json
os.makedirs("results", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"Benchmark results saved to {OUT}")
for r in results:
    print(r)
