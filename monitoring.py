import os
import json
from collections import Counter

LOG_DIR = "logs"
RESULTS_DIR = "results"
OUT_PATH = os.path.join(RESULTS_DIR, "monitoring_summary.json")

alerts_file = os.path.join(LOG_DIR, "ddos_alerts.log")
actions_file = os.path.join(LOG_DIR, "mitigation_actions.log")

summary = {
    "alerts_count": 0,
    "actions_count": 0,
    "unique_blocked_ips": 0,
    "blocked_ip_counts": {},
}

# Read alerts
if os.path.exists(alerts_file):
    with open(alerts_file, "r") as f:
        alerts = [l.strip() for l in f if l.strip()]
    summary["alerts_count"] = len(alerts)
else:
    alerts = []

# Read actions and extract IPs
ips = []
if os.path.exists(actions_file):
    with open(actions_file, "r") as f:
        actions = [l.strip() for l in f if l.strip()]
    summary["actions_count"] = len(actions)
    for line in actions:
        # naive parse: look for 'IP ' and take following token
        parts = line.split()
        if "IP" in parts:
            try:
                i = parts.index("IP")
                ip = parts[i+1]
                # strip punctuation
                ip = ip.strip().strip(".,;")
                ips.append(ip)
            except Exception:
                continue

counts = Counter(ips)
summary["unique_blocked_ips"] = len(counts)
summary["blocked_ip_counts"] = counts.most_common(200)

# Try to include high-level detection numbers from results if available
threshold_file = os.path.join("models", "threshold.json")
if os.path.exists(threshold_file):
    try:
        with open(threshold_file, "r") as f:
            th = json.load(f)
        summary["configured_threshold"] = th.get("probability_threshold")
    except Exception:
        pass

# Save summary
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("Monitoring summary written to", OUT_PATH)
