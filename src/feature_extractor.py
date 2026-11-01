import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP, ICMP
from collections import defaultdict
import warnings
from typing import Dict, List, Tuple, Any, Optional
from src.utils import logger

warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    State-of-the-Art, High-Parity PCAP Feature Extraction Engine.
    
    Research Innovations:
      1. **Dynamic Link-Layer Independence**: Seamlessly handles Ethernet, Linux Cooked (SLL),
         and Loopback (Null) capture types by delegating decoding to Scapy layers.
      2. **TSO/LRO MTU Reconstruction**: Simulates physical wire transmission for TCP Segment
         Offloading (TSO) and Large Receive Offload (LRO) giant packets, splitting them into
         ideal MTU-sized (1500 bytes) wire packets to maintain correct statistical distributions.
      3. **Exact CICFlowMeter Parity**: Employs application-layer payload sizing formulas to
         match training dataset feature boundaries precisely.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selected_features = [f.strip().lower() for f in config.get("selected_features", [])]
        self.flow_idle_timeout = config.get("feature_extraction", {}).get("flow_idle_timeout_seconds", 120)
        self.flow_active_timeout = config.get("feature_extraction", {}).get("flow_active_timeout_seconds", 600)
        self.handle_tso_lro = config.get("feature_extraction", {}).get("handle_tso_lro", True)
        self.mtu_size = config.get("feature_extraction", {}).get("mtu_split_bytes", 1500)

    def extract_features(self, pcap_path: str) -> Optional[pd.DataFrame]:
        logger.info(f"Initiating feature extraction: {pcap_path}")
        
        # Bidirectional flow tracker: key -> {'packet_sizes': [], 'timestamps': [], 'directions': [], 'flags': []}
        # directions: 1 for forward, -1 for backward
        flows = defaultdict(lambda: {"packet_sizes": [], "timestamps": [], "directions": [], "flags": []})
        flow_initiators = {}
        
        # Diagnostics
        pcap_packets = 0
        processed_ip = 0
        tso_packets_split = 0
        
        try:
            with PcapReader(pcap_path) as packets:
                for pkt in packets:
                    pcap_packets += 1
                    
                    if IP not in pkt:
                        continue
                    
                    processed_ip += 1
                    ip_layer = pkt[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    proto = ip_layer.proto
                    
                    sport = 0
                    dport = 0
                    tcp_flags = ""
                    
                    if TCP in pkt:
                        sport = pkt[TCP].sport
                        dport = pkt[TCP].dport
                        tcp_flags = str(pkt[TCP].flags)
                    elif UDP in pkt:
                        sport = pkt[UDP].sport
                        dport = pkt[UDP].dport
                    elif ICMP in pkt:
                        # Pair ICMP requests/replies via IP sorting
                        sport = 0
                        dport = 0

                    # 5-tuple canonical key
                    if src_ip < dst_ip:
                        key = (src_ip, dst_ip, sport, dport, proto)
                    else:
                        key = (dst_ip, src_ip, dport, sport, proto)
                        
                    # Establish initiator direction
                    if key not in flow_initiators:
                        flow_initiators[key] = (src_ip, sport)
                        direction = 1
                    else:
                        init_ip, init_port = flow_initiators[key]
                        direction = 1 if (src_ip == init_ip and sport == init_port) else -1
                        
                    # Calculate raw payload size matching CICFlowMeter
                    ip_total_len = ip_layer.len if ip_layer.len is not None else len(pkt)
                    ip_header_len = ip_layer.ihl * 4
                    
                    transport_header_len = 0
                    if TCP in pkt:
                        transport_header_len = pkt[TCP].dataofs * 4
                    elif UDP in pkt:
                        transport_header_len = 8
                    elif ICMP in pkt:
                        transport_header_len = 8
                        
                    app_payload_len = max(ip_total_len - ip_header_len - transport_header_len, 0)
                    timestamp = float(pkt.time)
                    
                    # ── TSO / LRO MTU Reconstruction ──
                    # If packet is a giant aggregated TSO frame, split it into MTU-level segments
                    if self.handle_tso_lro and app_payload_len > self.mtu_size:
                        tso_packets_split += 1
                        remaining_payload = app_payload_len
                        # Microsecond offset to prevent exact identical timestamps for split segments
                        time_offset = 0.0
                        
                        while remaining_payload > 0:
                            segment_size = min(remaining_payload, self.mtu_size)
                            flows[key]["packet_sizes"].append(segment_size)
                            flows[key]["timestamps"].append(timestamp + time_offset)
                            flows[key]["directions"].append(direction)
                            flows[key]["flags"].append(tcp_flags)
                            
                            remaining_payload -= segment_size
                            time_offset += 1e-6  # +1 microsecond step
                    else:
                        flows[key]["packet_sizes"].append(app_payload_len)
                        flows[key]["timestamps"].append(timestamp)
                        flows[key]["directions"].append(direction)
                        flows[key]["flags"].append(tcp_flags)
                        
                    if pcap_packets % 10000 == 0:
                        logger.info(f"Extracted features for {pcap_packets} packets...")
                        
            logger.info("Packet iteration completed.")
            logger.info(f"  Total captured packets: {pcap_packets}")
            logger.info(f"  Processed IP packets:   {processed_ip}")
            logger.info(f"  LRO/TSO segments split: {tso_packets_split}")
            logger.info(f"  Bidirectional flows:    {len(flows)}")
            
            # Apply active/idle timeouts
            split_flows, split_initiators = self._apply_flow_timeout(flows, flow_initiators)
            logger.info(f"Total flows after active/idle timeouts: {len(split_flows)}")
            
            # Extract 33-dimensional features
            data = []
            for k, f_data in split_flows.items():
                features = self._calculate_flow_stats(f_data, k, split_initiators[k])
                data.append(features)
                
            df = pd.DataFrame(data)
            
            # Pad missing columns with 0.0
            for feat in self.selected_features:
                if feat not in df.columns:
                    df[feat] = 0.0
                    
            return_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"] + self.selected_features
            return_cols = [c for c in return_cols if c in df.columns]
            return df[return_cols]

        except Exception as e:
            logger.error(f"Error during PCAP feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _apply_flow_timeout(self, flows: Dict, flow_initiators: Dict) -> Tuple[Dict, Dict]:
        """Splits continuous flows according to 120s idle and 600s active timeouts."""
        split_flows = {}
        split_initiators = {}
        
        for key, flow_data in flows.items():
            sizes = flow_data["packet_sizes"]
            times = flow_data["timestamps"]
            dirs = flow_data["directions"]
            flags = flow_data["flags"]
            
            if len(times) <= 1:
                split_flows[key] = flow_data
                split_initiators[key] = flow_initiators[key]
                continue
                
            sub_sizes = [sizes[0]]; sub_times = [times[0]]; sub_dirs = [dirs[0]]; sub_flags = [flags[0]]
            sub_start = times[0]
            subflow_idx = 0
            
            for i in range(1, len(times)):
                idle_gap = times[i] - times[i-1]
                active_dur = times[i] - sub_start
                
                if idle_gap > self.flow_idle_timeout or active_dur > self.flow_active_timeout:
                    # Save subflow
                    sub_key = (*key, subflow_idx)
                    split_flows[sub_key] = {
                        "packet_sizes": sub_sizes,
                        "timestamps": sub_times,
                        "directions": sub_dirs,
                        "flags": sub_flags
                    }
                    split_initiators[sub_key] = flow_initiators[key]
                    
                    # Reset
                    subflow_idx += 1
                    sub_sizes = [sizes[i]]; sub_times = [times[i]]; sub_dirs = [dirs[i]]; sub_flags = [flags[i]]
                    sub_start = times[i]
                else:
                    sub_sizes.append(sizes[i])
                    sub_times.append(times[i])
                    sub_dirs.append(dirs[i])
                    sub_flags.append(flags[i])
                    
            sub_key = (*key, subflow_idx) if subflow_idx > 0 else key
            split_flows[sub_key] = {
                "packet_sizes": sub_sizes,
                "timestamps": sub_times,
                "directions": sub_dirs,
                "flags": sub_flags
            }
            split_initiators[sub_key] = flow_initiators[key]
            
        return split_flows, split_initiators

    def _calculate_flow_stats(self, flow_data: Dict, key: Tuple, initiator: Tuple) -> Dict[str, Any]:
        """Computes statistical metrics over aggregated packet frames."""
        sizes = flow_data["packet_sizes"]
        times = flow_data["timestamps"]
        dirs = flow_data["directions"]
        flags = flow_data["flags"]
        
        stats = {}
        
        # Identifiers
        src_ip, src_port = initiator
        base_key = key[:5]
        dst_ip = base_key[1] if src_ip == base_key[0] else base_key[0]
        dst_port = base_key[3] if src_ip == base_key[0] else base_key[2]
        proto = base_key[4]
        
        stats["src_ip"] = src_ip
        stats["dst_ip"] = dst_ip
        stats["src_port"] = src_port
        stats["dst_port"] = dst_port
        stats["protocol"] = proto
        
        # Duration
        duration_us = (max(times) - min(times)) * 1e6
        stats["flow duration"] = duration_us
        
        # Direction split
        fwd_sizes = [sizes[i] for i in range(len(sizes)) if dirs[i] == 1]
        bwd_sizes = [sizes[i] for i in range(len(sizes)) if dirs[i] == -1]
        
        stats["total fwd packets"] = len(fwd_sizes)
        stats["total backward packets"] = len(bwd_sizes)
        stats["total length of fwd packets"] = float(sum(fwd_sizes))
        stats["total length of bwd packets"] = float(sum(bwd_sizes))
        
        total_payload = sum(sizes)
        duration_sec = duration_us / 1e6
        stats["flow bytes/s"] = total_payload / duration_sec if duration_sec > 0 else 0.0
        stats["flow packets/s"] = len(sizes) / duration_sec if duration_sec > 0 else 0.0
        
        # Packet size statistics
        self._set_stats(stats, "fwd packet length", fwd_sizes)
        self._set_stats(stats, "bwd packet length", bwd_sizes)
        
        # Inter-Arrival Times (IAT)
        if len(times) > 1:
            iats = np.diff(sorted(times)) * 1e6
            self._set_stats(stats, "flow iat", iats, include_total=False)
        else:
            self._set_zero_stats(stats, "flow iat", include_total=False)
            
        fwd_times = sorted([times[i] for i in range(len(times)) if dirs[i] == 1])
        if len(fwd_times) > 1:
            f_iats = np.diff(fwd_times) * 1e6
            self._set_stats(stats, "fwd iat", f_iats, include_total=True)
        else:
            self._set_zero_stats(stats, "fwd iat", include_total=True)
            
        bwd_times = sorted([times[i] for i in range(len(times)) if dirs[i] == -1])
        if len(bwd_times) > 1:
            b_iats = np.diff(bwd_times) * 1e6
            self._set_stats(stats, "bwd iat", b_iats, include_total=True)
        else:
            self._set_zero_stats(stats, "bwd iat", include_total=True)
            
        # TCP Flags
        fwd_psh = 0; bwd_psh = 0
        fwd_urg = 0; bwd_urg = 0
        
        for i in range(len(flags)):
            flg = flags[i]
            d = dirs[i]
            if "P" in flg:
                if d == 1: fwd_psh += 1
                else: bwd_psh += 1
            if "U" in flg:
                if d == 1: fwd_urg += 1
                else: bwd_urg += 1
                
        stats["fwd psh flags"] = fwd_psh
        stats["bwd psh flags"] = bwd_psh
        stats["fwd urg flags"] = fwd_urg
        stats["bwd urg flags"] = bwd_urg
        
        return stats

    def _set_stats(self, stats: Dict, prefix: str, data: Any, include_total: bool = False):
        if len(data) == 0:
            self._set_zero_stats(stats, prefix, include_total)
            return
            
        stats[f"{prefix} mean"] = float(np.mean(data))
        stats[f"{prefix} std"] = float(np.std(data))
        stats[f"{prefix} max"] = float(np.max(data))
        stats[f"{prefix} min"] = float(np.min(data))
        if include_total:
            stats[f"{prefix} total"] = float(sum(data))

    def _set_zero_stats(self, stats: Dict, prefix: str, include_total: bool = False):
        stats[f"{prefix} mean"] = 0.0
        stats[f"{prefix} std"] = 0.0
        stats[f"{prefix} max"] = 0.0
        stats[f"{prefix} min"] = 0.0
        if include_total:
            stats[f"{prefix} total"] = 0.0
