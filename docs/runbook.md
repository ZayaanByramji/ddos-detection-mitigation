# Runbook — DDoS Detection & Mitigation Prototype

- Start the API:
  - `python -m uvicorn api:app --host 0.0.0.0 --port 8000`

- Health check:
  - `GET /health`

- Predict endpoints:
  - `POST /predict` expects JSON with `data: [ {feature: value, ...}, ... ]`
  - `POST /predict_raw` expects list-of-lists matching training feature order

- Mitigation simulation:
  - `python mitigation_engine.py --profile demo`

- Benchmarks:
  - `python benchmarks/benchmark_inference.py --samples 20000`
  - `python benchmarks/scaled_benchmarks.py --samples 20000`
  - `python benchmarks/api_load_test.py`
  - `python benchmarks/advanced_load_test.py`

- Monitoring:
  - `python monitoring.py` -> writes `results/monitoring_summary.json`

- Notes:
  - Save trained models into `models/` and scaler as `models/scaler.joblib`.
  - Threshold is persisted at `models/threshold.json`.
