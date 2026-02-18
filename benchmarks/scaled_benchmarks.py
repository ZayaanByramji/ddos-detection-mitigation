import time
import numpy as np
import pandas as pd
import joblib
import os
import json
import argparse

MODEL_PATH = "models/final_random_forest_model.joblib"
SCALER_PATH = "models/scaler.joblib"
X_TEST_PATH = "X_test.parquet"

os.makedirs("results", exist_ok=True)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# CLI
parser = argparse.ArgumentParser(description="Scaled benchmark runner")
parser.add_argument("--samples", type=int, default=20000, help="number of samples to use")
args = parser.parse_args()

# configurations to test
configs = [
    {"method": "predict", "n_jobs": 1},
    {"method": "predict", "n_jobs": -1},
    {"method": "predict_proba", "n_jobs": 1},
]

# skip batch_size=1 to avoid excessive overhead in single-sample loops
batch_sizes = [8, 32, 128, 512]

# sample
X = pd.read_parquet(X_TEST_PATH)
N = min(args.samples, len(X))
sample = X.sample(n=N, random_state=42)
Xs = scaler.transform(sample)

results = []

for cfg in configs:
    method = cfg["method"]
    nj = cfg["n_jobs"]
    # try to set model.n_jobs where applicable
    try:
        if hasattr(model, "n_jobs"):
            model.n_jobs = nj
    except Exception:
        pass

    for batch in batch_sizes:
        iterations = int(np.ceil(len(Xs) / batch))
        timings = []
        for i in range(iterations):
            start = i * batch
            end = min((i + 1) * batch, len(Xs))
            batch_data = Xs[start:end]
            t0 = time.perf_counter()
            try:
                # map -1 to CPU count for backend
                backend_jobs = os.cpu_count() if nj == -1 else max(1, int(nj))
                import joblib as _joblib
                with _joblib.parallel_backend("loky", n_jobs=backend_jobs):
                    if method == "predict":
                        _ = model.predict(batch_data)
                    else:
                        _ = model.predict_proba(batch_data)
            except Exception as e:
                # record failure and break
                timings.append(None)
                print(f"Error during {method} with n_jobs={nj}, batch={batch}: {e}")
                break
            t1 = time.perf_counter()
            timings.append(t1 - t0)
        # filter out failed runs
        good = [t for t in timings if t is not None]
        total_time = sum(good) if good else None
        total_samples = len(Xs) if good else 0
        throughput = (total_samples / total_time) if total_time and total_time > 0 else None
        avg_latency = float(np.mean(good)) if good else None
        p95 = float(np.percentile(good, 95)) if good else None
        results.append({
            "method": method,
            "n_jobs": nj,
            "batch_size": batch,
            "total_samples": total_samples,
            "total_time_sec": total_time,
            "throughput_samples_per_sec": throughput,
            "avg_latency_sec": avg_latency,
            "p95_latency_sec": p95
        })
        print(results[-1])

OUT = os.path.join("results", "scaled_benchmarks.json")
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

print(f"Scaled benchmark results saved to {OUT}")
