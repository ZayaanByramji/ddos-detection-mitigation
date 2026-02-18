import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data and model
X_test = pd.read_parquet("X_test.parquet")
model = joblib.load(os.path.join(MODELS_DIR, "final_random_forest_model.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

# Feature importances
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_names = X_test.columns.tolist()
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    fi_df.sort_values("importance", ascending=False, inplace=True)
    fi_df.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"), index=False)

    # Bar plot of top 30
    topk = fi_df.head(30)
    plt.figure(figsize=(8, 10))
    plt.barh(topk["feature"][::-1], topk["importance"][::-1])
    plt.title("Top 30 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importances.png"), dpi=200)
    plt.close()
    print("Feature importances saved to results/")
else:
    print("Model lacks feature_importances_ attribute")

# Attempt SHAP explanations
try:
    import shap
    # Use a small sample for SHAP to keep compute low
    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    X_scaled = scaler.transform(sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    print("DEBUG: shap_values type:", type(shap_values))
    try:
        # If list, print element shapes
        if isinstance(shap_values, list):
            print("DEBUG: shap_values list lengths and shapes:")
            for i, arr in enumerate(shap_values):
                try:
                    print(i, "->", getattr(arr, 'shape', type(arr)))
                except Exception:
                    print(i, "-> (unprintable)")
        else:
            print("DEBUG: shap_values shape:", getattr(shap_values, 'shape', type(shap_values)))
    except Exception as e:
        print("DEBUG: failed to inspect shap_values:", e)

    # shap_values can be a list (one per class) for classifiers; pick class-1 if present
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            sv = shap_values[1]
        else:
            sv = shap_values[0]
    else:
        sv = shap_values

    print("DEBUG: selected sv type/shape:", type(sv), getattr(sv, 'shape', 'no-shape'))
    print("DEBUG: sample shape:", sample.shape)

    # If shap returned a 3D array (samples, features, classes), pick class-1 axis
    if hasattr(sv, 'ndim') and sv.ndim == 3:
        try:
            sv = sv[..., 1]
            print("DEBUG: reduced sv to shape:", sv.shape)
        except Exception:
            pass

    # summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"), dpi=200)
    plt.close()

    # save mean absolute shap per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs_shap})
    shap_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    shap_df.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)

    print("SHAP summary saved to results/")
except Exception as e:
    print("SHAP explanation skipped or failed:", e)
    print("Install the `shap` package to enable SHAP explanations.")
