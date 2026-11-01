from typing import Dict, Any, Optional, List

# MITRE ATT&CK mapping database based on Enterprise v14+ taxonomy
MITRE_ATTACK_MAP = {
    "Heuristic: Distributed Flood": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service: Direct Network Flood",
        "technique_id": "T1498.001",
        "severity": "CRITICAL",
        "description": "Distributed high-volume volumetric flooding (UDP/SYN/ICMP) targeting system bandwidth."
    },
    "Heuristic: Port Scan": {
        "tactic": "Reconnaissance",
        "tactic_id": "TA0043",
        "technique": "Active Scanning: Port Scanning",
        "technique_id": "T1595.001",
        "severity": "HIGH",
        "description": "Systematic destination port scanning detected across multiple ports."
    },
    "Heuristic: Low-and-Slow (Slowloris)": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Endpoint Denial of Service: Application Exhaustion Flood",
        "technique_id": "T1499.003",
        "severity": "HIGH",
        "description": "Low-rate HTTP/TCP connection exhaustion (Slowloris/R.U.D.Y. pattern) starving server socket slots."
    },
    "Heuristic: DNS Amplification": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service: Reflection Amplification",
        "technique_id": "T1498.002",
        "severity": "CRITICAL",
        "description": "DNS reflection attack utilizing open resolvers with highly amplified payload multipliers."
    },
    "Heuristic: SYN-ACK Reflection": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service: Reflection Amplification",
        "technique_id": "T1498.002",
        "severity": "HIGH",
        "description": "TCP reflection flooding leveraging spoofed source IP addresses sent to legitimate targets."
    },
    "Heuristic: Entropy Anomaly (DDoS)": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "HIGH",
        "description": "Abnormal source IP Shannon entropy indicating highly coordinated distributed flood patterns."
    },
    "XGBoost": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "HIGH",
        "description": "Supervised ensemble classifier identified matching known DoS/DDoS signature."
    },
    "Triple Consensus": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "CRITICAL",
        "description": "Consensus anomaly (B-DAP + T-RAD + QTN) mathematically demonstrating a severe, high-confidence zero-day threat."
    },
    "Consensus": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "HIGH",
        "description": "Dual-model unsupervised consensus anomaly flagging statistically deviant flow behaviour."
    },
    "Temporal Signature": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "MEDIUM",
        "description": "T-RAD temporal reconstruction anomaly — timing and frequency profiles deviate from benign network behaviors."
    },
    "Spatial Structure": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service",
        "technique_id": "T1498",
        "severity": "MEDIUM",
        "description": "B-DAP spatial structure anomaly — feature distributions deviate from learned benign topological manifolds."
    }
}


class MITREMapper:
    """Provides structured threat intelligence mapping for SIEM/SOAR integration."""

    def __init__(self):
        self.mapped_count = 0

    def map_reason(self, reason: str) -> Optional[Dict[str, str]]:
        """Maps a classification reason string to the corresponding MITRE ATT&CK taxonomy."""
        for key, mapping in MITRE_ATTACK_MAP.items():
            if key in reason:
                self.mapped_count += 1
                return mapping.copy()
        return None

    def enrich_results(self, reasons: List[str]) -> List[Optional[Dict[str, str]]]:
        return [self.map_reason(r) for r in reasons]
