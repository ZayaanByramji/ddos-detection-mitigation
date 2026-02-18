from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List, Dict, Any

app = FastAPI(title="DDoS Detection API")

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "final_random_forest_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
TRAIN_FEATURES_PATH = "X_train.parquet"

# Load artifacts at startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = pd.read_parquet(TRAIN_FEATURES_PATH).columns.tolist()
except Exception as e:
    model = None
    scaler = None
    feature_columns = None
    load_error = str(e)
else:
    load_error = None

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "feature_count": len(feature_columns) if feature_columns is not None else None,
        "load_error": load_error
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    try:
        df = pd.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    # Ensure columns match training features
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "missing_features", "missing": missing})

    # Order columns
    df = df[feature_columns]

    # Scale and predict
    X_scaled = scaler.transform(df)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1].tolist()
    else:
        probs = model.predict(X_scaled).tolist()
    preds = [int(p >= 0.5) if isinstance(p, float) else int(p) for p in probs]

    return {"predictions": preds, "probabilities": probs}

@app.post("/predict_raw")
def predict_raw(data: List[List[float]]):
    """Accept raw numeric rows (must match training feature order)."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    try:
        df = pd.DataFrame(data, columns=feature_columns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    X_scaled = scaler.transform(df)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1].tolist()
    else:
        probs = model.predict(X_scaled).tolist()
    preds = [int(p >= 0.5) if isinstance(p, float) else int(p) for p in probs]
    return {"predictions": preds, "probabilities": probs}
