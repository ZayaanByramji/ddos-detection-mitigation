import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load preprocessed data (produced by preprocessing)
X_train = pd.read_parquet("X_train.parquet")
X_test  = pd.read_parquet("X_test.parquet")
y_train = pd.read_parquet("y_train.parquet").values.ravel()
y_test  = pd.read_parquet("y_test.parquet").values.ravel()

print("Data loaded successfully")
print("Training samples:", X_train.shape)
print("Testing samples :", X_test.shape)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1-score :", f1_score(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# Decision Tree (baseline)
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy :", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall   :", recall_score(y_test, dt_pred))
print("F1-score :", f1_score(y_test, dt_pred))


# Support Vector Machine (may be slow on large datasets)
svm_model = None
try:
    if X_train_scaled.shape[0] > 50000:
        # use a linear SVM implementation for large datasets (faster)
        svm_model = LinearSVC(max_iter=2000, dual=False)
    else:
        svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")

    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)

    print("\n===== SVM RESULTS =====")
    print("Accuracy :", accuracy_score(y_test, svm_pred))
    print("Precision:", precision_score(y_test, svm_pred))
    print("Recall   :", recall_score(y_test, svm_pred))
    print("F1-score :", f1_score(y_test, svm_pred))
except Exception as e:
    print(f"SVM training skipped due to error or timeout: {e}")
    svm_model = None

print("\nModel training and evaluation completed successfully.")





# Saving models and scaler
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save models
joblib.dump(rf_model, "models/final_random_forest_model.joblib")
joblib.dump(dt_model, "models/decision_tree.joblib")
joblib.dump(svm_model, "models/svm.joblib")

# Save scaler
joblib.dump(scaler, "models/scaler.joblib")

print("\nModels and scaler saved successfully.")
print("Saved files:")
print(" - models/final_random_forest_model.joblib")
print(" - models/decision_tree.joblib")
print(" - models/svm.joblib")
print(" - models/scaler.joblib")