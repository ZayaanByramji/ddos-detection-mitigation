import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


#Load test data
X_test = pd.read_parquet("X_test.parquet")
y_test = pd.read_parquet("y_test.parquet").values.ravel()

print("Test data loaded")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#Load feature scaler and scale test features
scaler = joblib.load("models/scaler.joblib")
X_test_scaled = scaler.transform(X_test)

#Load saved models
models = {
    "Random Forest": joblib.load("models/final_random_forest_model.joblib"),
    "Decision Tree": joblib.load("models/decision_tree.joblib"),
    "SVM": joblib.load("models/svm.joblib")
}

# Create results directory
os.makedirs("results", exist_ok=True)

#Evaluate each model and save a confusion matrix plot
for name, model in models.items():
    print(f"\nEvaluating model: {name}")

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc}")
    print(f"Precision: {prec}")
    print(f"Recall   : {rec}")
    print(f"F1-score : {f1}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "DDoS"]
    )

    disp.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    # Save plot
    filename = f"results/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Confusion matrix saved as: {filename}")

print("\nModel evaluation completed successfully.")
