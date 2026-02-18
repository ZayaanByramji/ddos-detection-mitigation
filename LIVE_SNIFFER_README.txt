Live Network DDoS Detector
==========================

This script sniffs live network traffic and uses the trained ML model for real-time attack detection.

**Requirements:**
- Admin privileges (Windows) or sudo (Linux/Mac)
- Scapy: `pip install scapy`

**Usage:**

Basic (auto-detect network interface):
  python live_network_sniffer.py

Specify interface:
  Windows: python live_network_sniffer.py "Ethernet"
  Linux:   sudo python live_network_sniffer.py eth0
  Mac:     sudo python live_network_sniffer.py en0

Limit packet count (default: 0 = unlimited):
  python live_network_sniffer.py "Ethernet" 5000

**How it works:**

1. Sniffs IP packets from the network
2. Groups packets by source IP into "flows"
3. Every 100 packets per flow: extracts 77 features
4. Runs the model to classify as safe/attack
5. Flags and displays suspicious IPs in real-time
6. Auto-cleanup of flows older than 60 seconds

**Output Example:**
  [14:32:15] 192.168.1.100     → ✓ SAFE (92% confidence)
  [14:32:16] 203.0.113.42      → ✗ ATTACK (87% confidence)
              ⚠ Blocking 203.0.113.42

**Notes:**
- Skips localhost (127.x) and private ranges (192.168.x)
- Detection threshold: 0.57 (tuned from ROC analysis)
- Features extracted: packet counts, sizes, protocol ratios, flags, port entropy
- Real-time streaming classification (not batch)
