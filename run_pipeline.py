import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load data
print("\n[1] Loading preprocessed test data...")

X_test = pd.read_parquet("X_test.parquet")
y_test = pd.read_parquet("y_test.parquet").values.ravel()

print(f"Test samples loaded: {X_test.shape[0]}")

# Load scaler and models
print("\n[2] Loading trained models...")

scaler = joblib.load("models/scaler.joblib")

rf_model  = joblib.load("models/final_random_forest_model.joblib")
dt_model  = joblib.load("models/decision_tree.joblib")
svm_model = joblib.load("models/svm.joblib")

print("Models loaded successfully")

# Scale features
X_test_scaled = scaler.transform(X_test)

# Model evaluation helper
def evaluate_model(name, model, X, y):
    preds = model.predict(X)

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec  = recall_score(y, preds)
    f1   = f1_score(y, preds)
    cm   = confusion_matrix(y, preds)

    print(f"\n===== {name} EVALUATION =====")
    print(f"Accuracy : {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall   : {rec:.6f}")
    print(f"F1-score : {f1:.6f}")
    print("Confusion Matrix:")
    print(cm)

    return preds

# Evaluate all models
print("\n[3] Evaluating models...")

rf_preds  = evaluate_model("RANDOM FOREST", rf_model, X_test_scaled, y_test)
dt_preds  = evaluate_model("DECISION TREE", dt_model, X_test_scaled, y_test)
svm_preds = evaluate_model("SVM", svm_model, X_test_scaled, y_test)

# Mitigation (using best model: Random Forest)
print("\n[4] Running mitigation engine...")

# Simulated source IPs (for offline demo)
X_test_with_ip = X_test.copy()
X_test_with_ip["Source_IP"] = [
    f"192.168.1.{i % 255}" for i in range(len(X_test_with_ip))
]

X_test_with_ip["Prediction"] = rf_preds

ddos_traffic = X_test_with_ip[X_test_with_ip["Prediction"] == 1]
blocked_ips = ddos_traffic["Source_IP"].value_counts().head(255)

# Log mitigation
os.makedirs("logs", exist_ok=True)
with open("logs/mitigation_actions.log", "w") as f:
    for ip in blocked_ips.index:
        f.write(f"Blocked IP: {ip}\n")


# Final summary
print("\n===== MITIGATION SUMMARY =====")
print(f"Total traffic samples : {len(X_test)}")
print(f"DDoS detected         : {len(ddos_traffic)}")
print(f"IPs blocked           : {len(blocked_ips)}")

print("\nBlocked IPs (sample):")
for ip in blocked_ips.index[:10]:
    print(ip)

print("\nPipeline execution completed successfully.")