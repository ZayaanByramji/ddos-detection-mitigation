import subprocess
import sys

steps = [
    ("Preprocessing", [sys.executable, "preprocessing.py"]),
    ("Model training", [sys.executable, "model_training.py"]),
    ("Threshold selection", [sys.executable, "threshold_selection.py"]),
    ("Cross-validation", [sys.executable, "evaluation_reproducibility.py"]),
    ("Run pipeline (evaluation + mitigation log)", [sys.executable, "run_pipeline.py"]),
    ("Mitigation engine (prod profile)", [sys.executable, "mitigation_engine.py", "--profile", "prod"]),
    ("Explainability", [sys.executable, "explainability.py"]),
    ("Visualizations", [sys.executable, "visualizations.py"]),
    ("Evaluate models (confusion matrices)", [sys.executable, "evaluate_models.py"]),
]

for name, cmd in steps:
    print(f"\n=== Step: {name} ===")
    try:
        r = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Step '{name}' failed with exit code {e.returncode}. Stopping.")
        sys.exit(e.returncode)

print("\nAll steps completed successfully.")
