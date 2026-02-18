import json
import os
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Config
N_SPLITS = 5
RANDOM_STATE = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
X = pd.read_parquet("X_train.parquet")
y = pd.read_parquet("y_train.parquet").values.ravel()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

folds = []

fold_idx = 1
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Predict
    preds = model.predict(X_val_scaled)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val_scaled)[:, 1]
        try:
            roc = roc_auc_score(y_val, probs)
        except Exception:
            roc = None
    else:
        roc = None

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)

    folds.append({
        "fold": fold_idx,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": None if roc is None else float(roc)
    })

    fold_idx += 1

# Aggregate
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
aggregate = {}
for m in metrics:
    vals = [f[m] for f in folds if f[m] is not None]
    if vals:
        aggregate[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    else:
        aggregate[m] = {"mean": None, "std": None}

out = {
    "datetime": datetime.now().isoformat(),
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "folds": folds,
    "aggregate": aggregate
}

# Save results
with open(os.path.join(RESULTS_DIR, "eval_cv.json"), "w") as f:
    json.dump(out, f, indent=2)

# Append a brief summary to training output
summary_lines = [
    f"CV run: {out['datetime']}",
    f"n_splits: {N_SPLITS}, random_state: {RANDOM_STATE}",
    f"Aggregate accuracy: {aggregate['accuracy']['mean']:.6f} (std {aggregate['accuracy']['std']:.6f})",
    f"Aggregate precision: {aggregate['precision']['mean']:.6f} (std {aggregate['precision']['std']:.6f})",
    f"Aggregate recall: {aggregate['recall']['mean']:.6f} (std {aggregate['recall']['std']:.6f})",
    f"Aggregate f1: {aggregate['f1']['mean']:.6f} (std {aggregate['f1']['std']:.6f})",
]
with open("training output.txt", "a") as f:
    f.write("\n" + "\n".join(summary_lines) + "\n")

print("Cross-validation completed. Results written to results/eval_cv.json and appended to training output.txt")
