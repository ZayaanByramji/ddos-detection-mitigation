import json
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

# Paths
MODEL_PATH = "models/final_random_forest_model.joblib"
SCALER_PATH = "models/scaler.joblib"
X_TEST_PATH = "X_test.parquet"
Y_TEST_PATH = "y_test.parquet"
RESULTS_DIR = "results"
MODEL_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data and model
X_test = pd.read_parquet(X_TEST_PATH)
y_test = pd.read_parquet(Y_TEST_PATH).values.ravel()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X_test_scaled = scaler.transform(X_test)

# Get predicted probabilities
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_test_scaled)[:, 1]
else:
    # fallback to predictions as 0/1
    probs = model.predict(X_test_scaled)

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=200)
plt.close()

# Precision-Recall
precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
avg_precision = average_precision_score(y_test, probs)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"PR (AP = {avg_precision:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"), dpi=200)
plt.close()

# Choose threshold by maximizing F1 on test set
# Compute f1 for each threshold in pr_thresholds
# Note: pr_thresholds has len = len(precision)-1
f1_scores = []
thresholds = np.append(pr_thresholds, 1.0)  # pad to match lengths
for thr in thresholds:
    preds = (probs >= thr).astype(int)
    f1_scores.append(f1_score(y_test, preds))

f1_scores = np.array(f1_scores)
best_idx = f1_scores.argmax()
best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])

# Also compute metrics at default 0.5
default_preds = (probs >= 0.5).astype(int)
def_acc = (default_preds == y_test).mean()
def_prec = precision_score(y_test, default_preds)
def_rec = recall_score(y_test, default_preds)
def_f1 = f1_score(y_test, default_preds)

# Save summary
summary = {
    "roc_auc": float(roc_auc),
    "avg_precision": float(avg_precision),
    "best_threshold": best_threshold,
    "best_f1": best_f1,
    "default_threshold": 0.5,
    "default_metrics": {"precision": def_prec, "recall": def_rec, "f1": def_f1}
}

with open(os.path.join(RESULTS_DIR, "threshold_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Persist chosen threshold for mitigation engine
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
    json.dump({"probability_threshold": best_threshold}, f)

print("Threshold selection complete:")
print(f" - ROC AUC: {roc_auc:.6f}")
print(f" - Avg Precision: {avg_precision:.6f}")
print(f" - Best threshold (max F1): {best_threshold:.6f} (F1={best_f1:.6f})")
print(f" - Default (0.5) F1: {def_f1:.6f}")
print(f"Saved plots to {RESULTS_DIR} and threshold to models/threshold.json")
