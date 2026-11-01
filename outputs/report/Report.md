# Typhon IDS v4.0 -- Threat Intelligence Report

---

**Report Generated**: 2026-07-21 20:43:14
**Input File Source**: `Home Test1.pcapng`
**Analysis Framework**: Typhon Continuous-Time Neuromorphic-Tensor Fusion Architecture (CT-DGNN + ST3 + XGBoost)

---

## 1. Executive Summary

**Overall Verdict: 🟡 [WARNING] SUSPICIOUS**

> Neuromorphic-tensor models flagged 520 suspicious flows. Manual audit recommended.


| Traffic Metric | Flow Count | Percentage | Operational Status |
|---|---:|---:|---|
| **Total Flows Analyzed** | **1,939** | **100.00%** | -- |
| Benign (Clean Traffic) | 1,275 | 65.76% | 🟢 NORMAL |
| Confirmed DoS/DDoS Attacks | 144 | 7.43% | 🔴 DANGER |
| Critical Zero-Day Anomalies | 0 | 0.00% | 🔴 DANGER |
| Suspicious Flow Anomalies | 520 | 26.82% | 🟡 AUDIT |
| **Total Security Alerts** | **664** | **34.24%** | -- |

---

## 2. Dynamic Threat Severity Breakdown

| Severity Level | Flow Count | Tactical Attack Context |
|---|---:|---|
| **CRITICAL** (High Confidence) | 144 | Direct supervised threat signatures or triple unsupervised consensus. |
| **WARNING** (Medium Confidence) | 520 | Single-dimensional neuromorphic or spatial tensor reconstruction anomalies. |
| **INFO** (Cleared Overrides) | 115 | Validated broadcast, mDNS, ARP, or ICMP private keepalives. |

---

## 3. MITRE ATT&CK Enterprise Mapping

Threat signatures mapped strictly to the **MITRE ATT&CK Enterprise v14+** taxonomy:

| Mapping Category | Detections | Tactic | Technique | ID |
|---|---:|---|---|---|
| Impact | 1 | Impact | Network Denial of Service | `T1498` |

---

## 4. SOC Alerts Summary -- Confirmed Attacks (Top 30)

| Alert ID | Flow ID | Classification | Detection Reason | MITRE ATT&CK Context |
|---|---|---|---|---|
| 1 | Flow #98 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 2 | Flow #115 | DoS/DDoS | RL-Fusion Consensus (Action #0) -- Known Signature | `T1498` |
| 3 | Flow #171 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 4 | Flow #191 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 5 | Flow #285 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 6 | Flow #364 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 7 | Flow #376 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 8 | Flow #423 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 9 | Flow #452 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 10 | Flow #469 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 11 | Flow #494 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 12 | Flow #500 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 13 | Flow #505 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 14 | Flow #545 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 15 | Flow #575 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 16 | Flow #587 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 17 | Flow #604 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 18 | Flow #611 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 19 | Flow #616 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 20 | Flow #672 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 21 | Flow #688 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 22 | Flow #694 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 23 | Flow #700 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 24 | Flow #715 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 25 | Flow #724 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 26 | Flow #729 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 27 | Flow #730 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 28 | Flow #731 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 29 | Flow #732 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |
| 30 | Flow #733 | DoS/DDoS | C2E2 Confirmed Attack -- Supervised Core | N/A |

---

## 5. SOC Alerts Summary -- Suspicious Anomalies (Top 30)

| Alert ID | Flow ID | Classification | Anomaly Metric | Severity |
|---|---|---|---|---|
| 1 | Flow #0 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 2 | Flow #11 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 3 | Flow #12 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 4 | Flow #66 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 5 | Flow #67 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 6 | Flow #68 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 7 | Flow #69 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 8 | Flow #70 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 9 | Flow #71 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 10 | Flow #72 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 11 | Flow #73 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 12 | Flow #74 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 13 | Flow #82 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 14 | Flow #83 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 15 | Flow #84 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 16 | Flow #85 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 17 | Flow #100 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 18 | Flow #104 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 19 | Flow #114 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 20 | Flow #116 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 21 | Flow #117 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 22 | Flow #118 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 23 | Flow #119 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 24 | Flow #120 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 25 | Flow #121 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 26 | Flow #122 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 27 | Flow #127 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 28 | Flow #132 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 29 | Flow #134 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
| 30 | Flow #135 | Dynamic Graph/Spline Reconstruction Outlier | WARNING | 
