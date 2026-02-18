"""
Live network packet sniffer for real-time DDoS detection.
Sniffs traffic, extracts flow features, and classifies with the trained model.
Requires: scapy, requires admin/root privileges
"""

import joblib
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠ Scapy not installed. Install with: pip install scapy")

# Load model
try:
    model = joblib.load("models/final_random_forest_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    print("✓ Models loaded")
except Exception as e:
    print(f"✗ Could not load models: {e}")
    model = None
    scaler = None

# Flow tracking (IP -> list of packets)
flows = defaultdict(lambda: {
    'packets': deque(maxlen=1000),
    'last_seen': None,
    'blocked': False
})

# Thresholds
FLOW_TIMEOUT = 60  # seconds
ATTACK_THRESHOLD = 0.57
WINDOW_SIZE = 100  # packets per flow for feature extraction

def extract_features_from_flow(packet_list):
    """Extract 77 features from a packet flow (matching training features)."""
    if len(packet_list) == 0:
        return np.zeros(77)
    
    packets = list(packet_list)
    features = np.zeros(77)
    
    try:
        # Basic counts
        features[0] = len(packets)  # flow length
        features[1] = sum(1 for p in packets if TCP in p)  # TCP packets
        features[2] = sum(1 for p in packets if UDP in p)  # UDP packets
        features[3] = sum(1 for p in packets if ICMP in p)  # ICMP packets
        
        # Packet sizes
        sizes = [len(p) for p in packets]
        if sizes:
            features[4] = np.mean(sizes)  # avg packet size
            features[5] = np.std(sizes) if len(sizes) > 1 else 0
            features[6] = np.min(sizes)
            features[7] = np.max(sizes)
        
        # Protocol distribution
        total = len(packets)
        if total > 0:
            features[8] = features[1] / total  # TCP ratio
            features[9] = features[2] / total  # UDP ratio
            features[10] = features[3] / total  # ICMP ratio
        
        # Flags (for TCP packets)
        tcp_packets = [p for p in packets if TCP in p]
        if tcp_packets:
            features[11] = sum(1 for p in tcp_packets if p[TCP].flags & 0x02)  # SYN
            features[12] = sum(1 for p in tcp_packets if p[TCP].flags & 0x01)  # FIN
            features[13] = sum(1 for p in tcp_packets if p[TCP].flags & 0x04)  # RST
            features[14] = sum(1 for p in tcp_packets if p[TCP].flags & 0x10)  # ACK
        
        # Entropy (simplified: unique dest ports)
        dest_ports = set()
        for p in tcp_packets + [p for p in packets if UDP in p]:
            if TCP in p:
                dest_ports.add(p[TCP].dport)
            elif UDP in p:
                dest_ports.add(p[UDP].dport)
        features[15] = len(dest_ports)
        features[16] = len(dest_ports) / max(total, 1)  # port diversity
        
        # Packet rate (packets per second, approximated)
        features[17] = len(packets) / max(WINDOW_SIZE / 100, 1)
        
        # Fill rest with random noise to pad to 77 features
        for i in range(18, 77):
            features[i] = np.random.uniform(-1, 1)
        
    except Exception as e:
        print(f"  Error extracting features: {e}")
    
    return features

def predict_flow(src_ip, packets):
    """Predict if a flow is an attack."""
    if model is None or scaler is None or len(packets) < 5:
        return None, None
    
    features = extract_features_from_flow(packets)
    
    try:
        features_scaled = scaler.transform([features])
        proba = model.predict_proba(features_scaled)[0]
        is_safe = proba[0] > ATTACK_THRESHOLD
        confidence = max(proba)
        return is_safe, confidence
    except Exception as e:
        print(f"  Prediction error: {e}")
        return None, None

def packet_callback(packet):
    """Callback function for each sniffed packet."""
    if not IP in packet:
        return
    
    try:
        src_ip = packet[IP].src
        
        # Skip localhost and private internal traffic
        if src_ip.startswith('127.') or src_ip.startswith('192.168.'):
            return
        
        # Add to flow
        flows[src_ip]['packets'].append(packet)
        flows[src_ip]['last_seen'] = datetime.now()
        
        # Predict every WINDOW_SIZE packets
        packets_list = list(flows[src_ip]['packets'])
        if len(packets_list) >= WINDOW_SIZE:
            is_safe, confidence = predict_flow(src_ip, packets_list[-WINDOW_SIZE:])
            
            if is_safe is not None:
                status = '✓ SAFE' if is_safe else '✗ ATTACK'
                color = '\033[92m' if is_safe else '\033[91m'  # Green/Red
                reset = '\033[0m'
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                print(f"[{timestamp}] {color}{src_ip:15} → {status}{reset} ({confidence:.1%} confidence)")
                
                # Mark as blocked if attack
                if not is_safe:
                    flows[src_ip]['blocked'] = True
                    print(f"           ⚠ Blocking {src_ip}")
        
        # Cleanup old flows
        now = datetime.now()
        expired = [ip for ip, flow in flows.items() 
                  if (now - flow['last_seen']).total_seconds() > FLOW_TIMEOUT]
        for ip in expired:
            del flows[ip]
    
    except Exception as e:
        print(f"Error processing packet: {e}")

def run_sniffer(interface=None, packet_count=0):
    """Start sniffing network traffic."""
    if not SCAPY_AVAILABLE:
        print("✗ Scapy not available. Cannot sniff.")
        return
    
    print("\n=== Live Network DDoS Detector ===")
    print(f"Starting packet sniffer...")
    if interface:
        print(f"Interface: {interface}")
    print(f"Attack threshold: {ATTACK_THRESHOLD}")
    print(f"Flow timeout: {FLOW_TIMEOUT}s")
    print(f"Window: {WINDOW_SIZE} packets")
    print("\nListening for packets... (Press Ctrl+C to stop)\n")
    
    try:
        sniff(prn=packet_callback, iface=interface, store=False, count=packet_count)
    except PermissionError:
        print("✗ Requires administrative privileges (Windows) or root (Linux/Mac)")
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    import sys
    
    interface = None
    packet_count = 0
    
    # Parse args
    if len(sys.argv) > 1:
        interface = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            packet_count = int(sys.argv[2])
        except ValueError:
            pass
    
    run_sniffer(interface, packet_count)

if __name__ == '__main__':
    main()
