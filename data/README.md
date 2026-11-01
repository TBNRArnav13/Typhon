# Typhon Dataset Management & Guide

This directory manages the training and benchmark evaluation datasets for the **Typhon Intrusion Detection System**.

---

## 1. Quick Verification with Sample Data

For immediate out-of-the-box verification without downloading tens of gigabytes of raw traffic, Typhon includes a verified sample dataset:
- `data/sample_flows.csv` (200 standardized network flows: 100 Benign, 100 DDoS attacks across multiple reflection/exploitation vectors).

Run detection immediately:
```bash
python main.py --mode detect --file data/sample_flows.csv
```

---

## 2. Full Benchmark Datasets (CIC-DDoS2019)

Typhon was trained and evaluated on the canonical **Canadian Institute for Cybersecurity (CIC) DDoS2019** benchmark dataset:
- **Paper / Source**: Sharafaldin et al., *"Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"*, IEEE International Conference on Information Systems Security and Privacy (ICISSP), 2019.
- **Official Download URL**: [UNB CIC-DDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)

### Recommended Directory Structure for Full Training / Testing

Place the full raw CSV / PCAP datasets in the following structure (or point `config.yaml` to your custom storage path):

```text
Typhon/
└── Dataset/
    ├── Traning/
    │   ├── 1_Million_Benign.csv             # Clean benign baseline traffic
    │   └── Final_70_30_Dataset.csv          # Combined 70% attack / 30% benign training split
    ├── PolishedDos/
    │   ├── Polished_Dos_DNS.csv
    │   ├── Polished_Dos_LDAP.csv
    │   ├── Polished_Dos_MSSQL.csv
    │   ├── Polished_Dos_NetBIOS.csv
    │   ├── Polished_Dos_NTP.csv
    │   ├── Polished_Dos_SNMP.csv
    │   ├── Polished_Dos_SSDP.csv
    │   ├── Polished_Dos_Syn.csv
    │   ├── Polished_Dos_TFTP.csv
    │   └── Polished_Dos_UDP.csv
    └── Testing CIC/
        ├── Standardized_500K_Test_Set.csv   # Unified 500K zero-leakage test set
        ├── Home Test1.pcapng                # Raw PCAP capture for packet-level validation
        └── MixSet_*.csv                     # Protocol-stratified evaluation sets
```

---

## 3. Creating the Standardized 500,000-Flow Evaluation Set

To reproduce the exact zero-leakage evaluation protocol described in the research paper:

```bash
python src/create_standardized_500k_dataset.py
```

This script:
1. Performs stratified sampling across 12 distinct attack vectors and benign traffic.
2. Removes known timestamp/IP leakage columns to ensure pure Deep Flow Inspection (DFI).
3. Standardizes column names and formats to the 33 essential feature schema specified in `config.yaml`.
4. Outputs `Dataset/Testing CIC/Standardized_500K_Test_Set.csv`.

---

## 4. Feature Schema (33 Flow Attributes)

Typhon extracts and processes 33 statistical flow attributes:
- **Temporal & Volume**: `flow duration`, `flow bytes/s`, `flow packets/s`
- **Forward Flow Statistics**: `total fwd packets`, `total length of fwd packets`, `fwd packet length max/min/mean/std`
- **Backward Flow Statistics**: `total backward packets`, `total length of bwd packets`, `bwd packet length max/min/mean/std`
- **Inter-Arrival Times (IAT)**: `flow iat mean/std/max/min`, `fwd iat total/mean/std/max/min`, `bwd iat total/mean/std/max/min`
- **TCP Control Flags**: `fwd psh flags`, `bwd psh flags`, `fwd urg flags`, `bwd urg flags`
