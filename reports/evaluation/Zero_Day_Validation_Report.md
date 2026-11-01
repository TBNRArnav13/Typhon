# Empirical Zero-Day Validation Report
**Date Generated:** 2026-05-03 01:17:04
**Methodology:** Leave-One-Out (LOO) Attack Validation. For each attack type, the supervised model (XGBoost) was trained on a dataset where that specific attack was completely removed. The full hybrid system was then tested solely on the removed attack. Any successful detections represent genuine zero-day anomaly detection by the unsupervised models or heuristics.

## Results Table
| Attack Type | Total Samples | Detected | Zero-Day Detection Rate | Primary Source |
|-------------|---------------|----------|-------------------------|----------------|
| DrDoS_DNS | 99932 | 99679 | **99.75%** | XGBoost (Feature Overlap) |
| DrDoS_LDAP | 99930 | 99927 | **100.0%** | XGBoost (Feature Overlap) |
| DrDoS_MSSQL | 99969 | 99964 | **99.99%** | XGBoost (Feature Overlap) |
| DrDoS_NetBIOS | 99962 | 99954 | **99.99%** | XGBoost (Feature Overlap) |
| DrDoS_NTP | 98858 | 95100 | **96.2%** | XGBoost (Feature Overlap) |
| DrDoS_SNMP | 99973 | 99963 | **99.99%** | XGBoost (Feature Overlap) |
| DrDoS_SSDP | 99960 | 99955 | **99.99%** | XGBoost (Feature Overlap) |
| Syn | 99964 | 12990 | **12.99%** | XGBoost (Feature Overlap) |
| TFTP | 99875 | 99841 | **99.97%** | XGBoost (Feature Overlap) |
| DrDoS_UDP | 99929 | 99892 | **99.96%** | XGBoost (Feature Overlap) |