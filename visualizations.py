import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Setup
os.makedirs("results", exist_ok=True)

# Load data
X_test = pd.read_parquet("X_test.parquet")
y_test = pd.read_parquet("y_test.parquet").values.ravel()

y_train = pd.read_parquet("y_train.parquet").values.ravel()

# Load scaler
scaler = joblib.load("models/scaler.joblib")
X_test_scaled = scaler.transform(X_test)

# Load models
rf_model = joblib.load("models/final_random_forest_model.joblib")
dt_model = joblib.load("models/decision_tree.joblib")
svm_model = joblib.load("models/svm.joblib")

# Model comparison bar chart
models = {
    "Random Forest": rf_model,
    "Decision Tree": dt_model,
    "SVM": svm_model
}

metrics = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-score": []
}

for model in models.values():
    preds = model.predict(X_test_scaled)
    metrics["Accuracy"].append(accuracy_score(y_test, preds))
    metrics["Precision"].append(precision_score(y_test, preds))
    metrics["Recall"].append(recall_score(y_test, preds))
    metrics["F1-score"].append(f1_score(y_test, preds))

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, metrics["Accuracy"], width, label="Accuracy")
plt.bar(x - 0.5*width, metrics["Precision"], width, label="Precision")
plt.bar(x + 0.5*width, metrics["Recall"], width, label="Recall")
plt.bar(x + 1.5*width, metrics["F1-score"], width, label="F1-score")

plt.xticks(x, models.keys())
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.close()

# Class distribution (dataset)
labels, counts = np.unique(
    np.concatenate((y_train, y_test)),
    return_counts=True
)

plt.figure(figsize=(6, 5))
plt.bar(["Benign", "DDoS"], counts)
plt.title("Class Distribution in Dataset")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("results/class_distribution.png")
plt.close()


# Detection vs benign (test set)
rf_preds = rf_model.predict(X_test_scaled)
labels, counts = np.unique(rf_preds, return_counts=True)

plt.figure(figsize=(6, 5))
plt.bar(["Benign", "DDoS"], counts)
plt.title("Traffic Classification Outcome (Test Set)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("results/detection_vs_benign.png")
plt.close()


# Mitigation impact graph
# Values based on mitigation engine output
total_ddos = int((rf_preds == 1).sum())
blocked_ips = 255  # from your mitigation summary

plt.figure(figsize=(6, 5))
plt.bar(["Detected DDoS Samples", "Blocked IPs"], [total_ddos, blocked_ips])
plt.title("Mitigation Impact Summary")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/mitigation_impact.png")
plt.close()

print("All visualizations generated successfully.")
