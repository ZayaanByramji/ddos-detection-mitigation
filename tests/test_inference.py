import joblib
import pandas as pd
import os
import pytest


def test_model_and_scaler_load():
    assert os.path.exists("models/final_random_forest_model.joblib")
    assert os.path.exists("models/scaler.joblib")


def test_inference_output_shape_and_types():
    if not os.path.exists("X_test.parquet"):
        pytest.skip("X_test.parquet not available in CI")
    model = joblib.load("models/final_random_forest_model.joblib")
    scaler = joblib.load("models/scaler.joblib")

    X_test = pd.read_parquet("X_test.parquet")
    sample = X_test.head(10)

    Xs = scaler.transform(sample)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xs)[:, 1]
    else:
        probs = model.predict(Xs)

    preds = (probs >= 0.5).astype(int) if probs.dtype.kind == 'f' else probs

    assert len(preds) == sample.shape[0]
    for p in preds:
        assert int(p) in (0, 1)
