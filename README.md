# Detection and Mitigation of DDoS Attacks (Machine Learning)

This repository contains a machine-learning based DDoS detection and mitigation demo. It includes preprocessing, model training, evaluation, streaming-style mitigation simulation, explainability (SHAP), and a small API prototype for deployment.

Contents
- `preprocessing.py` - prepare datasets and save `X_train.parquet`, `X_test.parquet`, `y_train.parquet`, `y_test.parquet`
- `model_training.py` - train Random Forest, Decision Tree, SVM and save artifacts in `models/`
- `threshold_selection.py` - compute ROC/PR curves and select an operating threshold
- `evaluation_reproducibility.py` - run cross-validation and save `results/eval_cv.json`
- `mitigation_engine.py` - streaming demo with rate-limiting, TTL blocks, and safety profiles
- `explainability.py` - feature importances and SHAP summaries
- `run_pipeline.py`, `evaluate_models.py`, `visualizations.py` - helpers for evaluation and plotting
- `run_all.py` - convenience script to run the full pipeline end-to-end
- `api.py` - FastAPI model serving prototype
- `Dockerfile` - buildable container for the API
- `requirements.txt` - Python dependencies

Quick start (local)
1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```


2. Run the full pipeline (this will re-run preprocessing, training, evaluation, and generate results):

```bash
python run_all.py
```

3. Inspect results in `results/` and logs in `logs/`.

## Running the API (development)
Start the FastAPI service locally (port 8000):

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` - basic liveness
- `GET /metadata` - model metadata
- `POST /predict` - accepts a JSON array of feature objects or a dict with `data` key. Example payload:

```json
[
  {"Avg Bwd Segment Size": 123, "Bwd Packet Length Mean": 44.3, ...},
  {...}
]
```

Response contains `predictions` (0/1) and `probabilities` (float).

## Docker
Build and run the API container:

```bash
docker build -t ddos-mitigation:latest .
docker run -p 8000:8000 ddos-mitigation:latest
```

## Notes
- The chosen operating probability threshold is saved in `models/threshold.json` and is used by `mitigation_engine.py` if the profile is not overridden.
- SHAP explanations require the `shap` package and can be expensive on large datasets; `explainability.py` samples data to keep compute reasonable.
- Default safety profiles are available via `--profile` in `mitigation_engine.py` (demo/dev/prod).

## Next steps / suggestions
- Add unit and integration tests (recommended: small synthetic dataset for CI).
- Harden the API for production: validation, authentication, rate-limits, metrics.
- Add monitoring and alerting integrations.

