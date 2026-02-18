import os
import sys
import subprocess
import pandas as pd


def test_mitigation_runs_and_logs(tmp_path):
    # create a small sample input from X_test
    X_test = pd.read_parquet("X_test.parquet")
    small = X_test.head(200)
    input_path = tmp_path / "small_X.parquet"
    small.to_parquet(str(input_path))

    # run mitigation_engine.py with small input
    cmd = [sys.executable, "mitigation_engine.py", "--input", str(input_path),
           "--chunk-size", "50", "--prob-threshold", "0.5", "--hits-required", "1",
           "--window", "10", "--block-ttl", "10", "--rate-limit", "100"]

    r = subprocess.run(cmd, cwd=os.getcwd())
    assert r.returncode == 0

    # Expect logs to be created
    assert os.path.exists("logs/mitigation_actions.log")
    assert os.path.exists("logs/ddos_alerts.log")

    # Check logs contain at least the ACTION keyword (may be empty depending on detections)
    with open("logs/mitigation_actions.log", "r") as f:
        data = f.read()
    assert "ACTION" in data or data == "", "Mitigation actions log missing expected content"
