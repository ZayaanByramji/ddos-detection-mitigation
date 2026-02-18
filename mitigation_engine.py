import argparse
import joblib
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque

# CLI/config
parser = argparse.ArgumentParser(description="Mitigation engine (streaming demo)")
parser.add_argument("--input", default="X_test.parquet", help="Parquet file with features")
parser.add_argument("--chunk-size", type=int, default=5000, help="Processing chunk size")
parser.add_argument("--prob-threshold", type=float, default=0.90, help="Probability threshold")
parser.add_argument("--hits-required", type=int, default=5, help="Hits required to block")
parser.add_argument("--window", type=int, default=60, help="Sliding window (seconds)")
parser.add_argument("--block-ttl", type=int, default=300, help="Block TTL (seconds)")
parser.add_argument("--rate-limit", type=int, default=100, help="Allowed requests per window before rate-limit")
parser.add_argument("--whitelist", default="", help="Comma-separated IPs to whitelist")
parser.add_argument("--profile", choices=["demo", "dev", "prod"], default=None, help="Safety profile presets")
args = parser.parse_args()

# Load trained model & scaler
MODEL_PATH = "models/final_random_forest_model.joblib"
SCALER_PATH = "models/scaler.joblib"

# Directory for runtime logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print("Failed to load model/scaler:", e)
    raise

print("Model and scaler loaded successfully")

# Load incoming traffic data (simulated)
incoming_data = pd.read_parquet(args.input)
print("Incoming traffic loaded")
print("Samples:", incoming_data.shape[0])

# Simulated IP addresses (one per flow)
simulated_ips = [f"192.168.1.{i % 255}" for i in range(len(incoming_data))]

# Mitigation configuration (from args)
CHUNK_SIZE = args.chunk_size
PROBABILITY_THRESHOLD = args.prob_threshold
HITS_REQUIRED = args.hits_required
WINDOW_SECONDS = args.window
BLOCK_TTL_SECONDS = args.block_ttl
RATE_LIMIT = args.rate_limit

# Load saved threshold if available and user didn't override
THRESHOLD_FILE = os.path.join("models", "threshold.json")
if os.path.exists(THRESHOLD_FILE):
    try:
        import json
        with open(THRESHOLD_FILE, "r") as f:
            th = json.load(f).get("probability_threshold")
            # only apply if user left the default
            if abs(PROBABILITY_THRESHOLD - parser.get_default("prob_threshold")) < 1e-9:
                PROBABILITY_THRESHOLD = float(th)
                print(f"Loaded saved threshold: {PROBABILITY_THRESHOLD}")
    except Exception:
        pass

# Apply profile presets if requested
PROFILE_PRESETS = {
    "demo": {"prob": 0.95, "hits": 10, "window": 120, "block_ttl": 60, "rate_limit": 1000},
    "dev":  {"prob": 0.85, "hits": 7,  "window": 60,  "block_ttl": 300, "rate_limit": 500},
    "prod": {"prob": 0.57, "hits": 5,  "window": 60,  "block_ttl": 300, "rate_limit": 200}
}

if args.profile:
    preset = PROFILE_PRESETS.get(args.profile)
    if preset:
        # Only override values if user didn't specify non-defaults
        if abs(args.prob_threshold - parser.get_default("prob_threshold")) < 1e-9:
            PROBABILITY_THRESHOLD = preset["prob"]
        if args.hits_required == parser.get_default("hits_required"):
            HITS_REQUIRED = preset["hits"]
        if args.window == parser.get_default("window"):
            WINDOW_SECONDS = preset["window"]
        if args.block_ttl == parser.get_default("block_ttl"):
            BLOCK_TTL_SECONDS = preset["block_ttl"]
        if args.rate_limit == parser.get_default("rate_limit"):
            RATE_LIMIT = preset["rate_limit"]
        print(f"Applied profile '{args.profile}' with settings: {preset}")

WHITELIST = set([ip.strip() for ip in args.whitelist.split(",") if ip.strip()])

ALERT_LOG = os.path.join(LOG_DIR, "ddos_alerts.log")
ACTION_LOG = os.path.join(LOG_DIR, "mitigation_actions.log")

# State
ip_hits = defaultdict(deque)        # ip -> deque of hit timestamps
blocked_ips = {}                    # ip -> unblock_datetime
rate_limited_ips = {}               # ip -> unblock_datetime
request_counters = defaultdict(deque)  # ip -> deque of request timestamps

total_detected = 0

def log_alert(msg):
    with open(ALERT_LOG, "a") as f:
        f.write(msg + "\n")

def log_action(msg):
    with open(ACTION_LOG, "a") as f:
        f.write(msg + "\n")

def now_ts():
    return datetime.now()

# Stream through data in chunks
num_samples = len(incoming_data)
for start in range(0, num_samples, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, num_samples)
    chunk = incoming_data.iloc[start:end]
    ips = simulated_ips[start:end]

    # Scale and predict probabilities
    try:
        X_chunk_scaled = scaler.transform(chunk)
    except Exception as e:
        print("Scaler transform failed:", e)
        break

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_chunk_scaled)[:, 1]
    else:
        probs = model.predict(X_chunk_scaled)

    for i, p in enumerate(probs):
        ip = ips[i]
        ts = now_ts()

        # Skip whitelist
        if ip in WHITELIST:
            continue

        # Unblock expired blocks
        if ip in blocked_ips and ts >= blocked_ips[ip]:
            del blocked_ips[ip]
            log_action(f"[{ts}] ACTION: IP {ip} unblocked after TTL expiry")

        if ip in rate_limited_ips and ts >= rate_limited_ips[ip]:
            del rate_limited_ips[ip]
            log_action(f"[{ts}] ACTION: IP {ip} rate-limit expired")

        # If currently blocked or rate-limited, skip enforcement but count requests
        if ip in blocked_ips:
            continue

        # Maintain request counter for rate limiting
        reqs = request_counters[ip]
        reqs.append(ts)
        window_start = ts - timedelta(seconds=WINDOW_SECONDS)
        while reqs and reqs[0] < window_start:
            reqs.popleft()

        if len(reqs) > RATE_LIMIT and ip not in rate_limited_ips:
            # Apply rate-limit
            unblock_time = ts + timedelta(seconds=BLOCK_TTL_SECONDS)
            rate_limited_ips[ip] = unblock_time
            log_action(f"[{ts}] ACTION: IP {ip} rate-limited until {unblock_time} (reqs={len(reqs)})")
            # don't process further detections for this IP while rate-limited
            continue

        # Detection based on probability threshold
        if p >= PROBABILITY_THRESHOLD:
            total_detected += 1
            dq = ip_hits[ip]
            dq.append(ts)
            # trim hits outside window
            while dq and dq[0] < window_start:
                dq.popleft()

            if len(dq) >= HITS_REQUIRED:
                unblock_time = ts + timedelta(seconds=BLOCK_TTL_SECONDS)
                blocked_ips[ip] = unblock_time
                alert_msg = f"[{ts}] ALERT: DDoS detected from IP {ip} (hits={len(dq)})"
                action_msg = f"[{ts}] ACTION: IP {ip} blocked until {unblock_time}"
                log_alert(alert_msg)
                log_action(action_msg)

    # Small sleep to simulate real-time stream (kept tiny for demo)
    time.sleep(0.01)

# Summary
print("\n===== MITIGATION SUMMARY =====")
print("Total traffic samples :", num_samples)
print("DDoS detections (probability threshold):", total_detected)
print("Currently blocked IPs           :", len(blocked_ips))
print("Currently rate-limited IPs       :", len(rate_limited_ips))

if blocked_ips:
    print("\nBlocked IPs (sample):")
    for ip in list(blocked_ips)[:10]:
        print(ip)

if rate_limited_ips:
    print("\nRate-limited IPs (sample):")
    for ip in list(rate_limited_ips)[:10]:
        print(ip)

print("\nMitigation completed successfully.")
