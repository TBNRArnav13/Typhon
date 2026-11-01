<div align="center">

# 🌪️ Typhon IDS
### Real-Time Neuromorphic-Tensor Hybrid Intrusion Detection System for Adaptive DoS/DDoS Mitigation

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-18534F?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)
[![CI Status](https://img.shields.io/badge/build-passing-success?style=for-the-badge&logo=githubactions)](.github/workflows/ci.yml)

<p align="center">
  <b>Typhon</b> is an edge-optimized, hybrid Deep Flow Inspection (DFI) intrusion detection framework designed to detect signature-based volumetric DDoS attacks and evasive, zero-day protocol shifts at microsecond latencies.
</p>

[Product Overview](#-product-overview) •
[Architecture](#-system-architecture) •
[Performance Benchmarks](#-performance-benchmarks) •
[Quick Start](#-quick-start) •
[CLI Usage](#-cli-usage-guide) •
[Directory Layout](#-directory-layout) •
[License](#-license)

</div>

---

## 🛡️ Product Overview

Traditional network intrusion detection systems (NIDS) face a fundamental trade-off:
- **Supervised classifiers** (e.g., Random Forest, XGBoost) offer exceptional speed and accuracy on known signatures, but suffer severe performance collapse (recall dropping to <65%) when facing novel attack vectors or out-of-distribution protocol shifts.
- **Unsupervised anomaly detectors** (e.g., AutoEncoders) can detect zero-day deviations, but struggle with high false positive rates and higher computational overhead.

**Typhon** bridges this gap by deploying an intelligent multi-stage pipeline:
1. **Lightweight Supervised Fast-Path**: Inspects clean, high-confidence traffic in **$56.4\ \mu\text{s}$**.
2. **Deep Neuromorphic & Tensor Analysis**: Dynamically routes suspicious, ambiguous, or drifted traffic through:
   - **Continuous-Time Dynamic Graph Neural Network (CT-DGNN)** using Liquid Time-Constant (LTC) ODE cells to track host interaction states across irregular packet arrivals.
   - **Spatio-Temporal Tensor-Transformer (ST3)** unifying Kolmogorov-Arnold Networks (KAN), Rotary Position Embeddings (RoPE), and Matrix Product State (MPS) tensor contractions for anomaly discovery.
3. **Neuro-Symbolic Reinforcement Learning (DQN) Fusion**: Continuously fuses signals using a 6-dimensional state vector, maximizing threat recall while applying symbolic safety overrides.
4. **Confidence-Calibrated Early Exit (C2E2)**: Delivers an effective weighted per-flow latency of **$288.89\ \mu\text{s}$** and throughput of **3,461.5 flows/second** on standard edge hardware.

---

## ⚡ System Architecture

```text
                                  TYPHON IDS PIPELINE
  
   Raw Packets (.pcap) / Extracted Flows (.csv)
                        │
                        ▼
   ┌────────────────────────────────────────────────────────────┐
   │    Input Handler & Feature Extractor (33 DFI Attributes)   │
   └─────────────────────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
      ┌───────────────────────┐       ┌────────────────────┐
      │ Supervised Fast-Path  │       │ Concept Drift Check│
      │       (XGBoost)       │       │   (JS Divergence)  │
      └──────────┬────────────┘       └─────────┬──────────┘
                 │ p_xgb                        │
      [C2E2 Gating: p > 0.995]                  │
         ├────────────► Early Exit (288 µs)     │
         │                                      │
         ▼ (Uncertain / Drift Detected)         ▼
   ┌────────────────────────────────────────────────────────────┐
   │            Deep Neuromorphic & Tensor Branch               │
   │                                                            │
   │  ┌────────────────────────────┐  ┌──────────────────────┐  │
   │  │   CT-DGNN (LTC ODE Cells)  │  │  ST3 AutoEncoder     │  │
   │  │  Irregular Inter-Arrivals  │  │  • KAN Splines       │  │
   │  │  Host Interaction States   │  │  • RoPE Attention    │  │
   │  │                            │  │  • MPS Contraction   │  │
   │  └──────────────┬─────────────┘  └──────────┬───────────┘  │
   │                 │                           │              │
   │                 │ Node Embeddings           │ EVT Error    │
   └─────────────────┼───────────────────────────┼──────────────┘
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ DQN RL-Fusion Engine      │
                     │ State: [p, e, H, V, L, Δ] │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │ Final Decision & MITRE    │
                     │ (Benign/Malicious/Alert)  │
                     └───────────────────────────┘
```

---

## 📊 Performance Benchmarks

Evaluated over **500,000 network flows** from the canonical CIC-DDoS2019 dataset under zero-leakage conditions (timestamps, MACs, and IP addresses stripped from statistical learning):

### Detection Performance Comparison

| Model / Pipeline | Accuracy | Precision | Recall | F1-Score | False Positives | Zero-Day Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Standalone XGBoost | **99.81%** | **99.82%** | 99.91% | **99.87%** | **608** | 64.79% |
| Classical Kitsune AutoEncoder | 91.24% | 88.45% | 93.10% | 90.71% | 14,208 | 88.40% |
| Standalone ST3 (KAN + MPS) | 96.72% | 95.10% | 98.40% | 96.72% | 5,830 | 97.90% |
| **Typhon (Full Hybrid Framework)** | 99.55% | 99.45% | **99.77%** | 99.61% | 1,919 | **99.83%** |

> **Key Advantage**: On novel zero-day attack shifts, standalone supervised classifiers suffer a **35.12% drop in recall**. Typhon sustains **99.83% recall**, catching stealthy and shifted vectors while preserving sub-millisecond inspection latency.

### Latency & Throughput Profile

| Pipeline Stage | Latency per Flow | Memory Footprint | Edge Deployment Feasibility |
| :--- | :---: | :---: | :---: |
| Supervised Fast-Path (XGBoost) | $56.40\ \mu\text{s}$ | $< 5\text{ MB}$ | High (Gateway / SmartNIC) |
| Feature Extraction & Scaling | $38.20\ \mu\text{s}$ | Negligible | Real-time |
| CT-DGNN Dynamic Graph Update | $840.10\ \mu\text{s}$ | $19.2\text{ KB}$ | Edge GPU / Jetson |
| ST3 Tensor-Transformer Inference | $1,550.00\ \mu\text{s}$ | $433.6\text{ KB}$ | Edge GPU / NPU |
| DQN Action Selection | $12.30\ \mu\text{s}$ | $22.2\text{ KB}$ | CPU / Microcontroller |
| **C2E2 Weighted Effective Latency** | **$288.89\ \mu\text{s}$** | **Total: $\sim 1.3\text{ MB}$** | **3,461.5 Flows/Sec** |

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: 3.9, 3.10, or 3.11
- **OS**: Windows, Linux, or macOS
- **Hardware**: CPU supported; NVIDIA CUDA GPU recommended for high batch throughput.
- **Pretrained Weights**: Pre-packaged in `models_saved/` (~1.3 MB total, plug-and-play).

### 2. One-Click Setup

#### Windows
Run `run.bat`:
```cmd
run.bat
```

#### Linux / macOS
Run `run.sh`:
```bash
chmod +x run.sh
./run.sh
```

The script automatically initializes a virtual environment, installs dependencies from `requirements.txt`, and runs a detection smoke test on `data/sample_flows.csv`.

### 3. Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/typhon.git
cd typhon

# 2. Create virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as local editable package
pip install -e .
```

---

## 💻 CLI Usage Guide

Typhon provides an intuitive CLI via `main.py`:

### 1. Real-Time Detection & Reporting
Analyze network traffic from PCAP captures or flow CSV files:
```bash
python main.py --mode detect --file data/sample_flows.csv
```
**What happens**:
- Flows are parsed and standardized to 33 DFI features.
- Drift status and flow velocity are evaluated.
- Predictions and risk scores are generated.
- A human-readable Markdown summary report (`outputs/run_*/report/Report.md`) and structured JSON alert metrics (`Report.json`) are generated automatically.

### 2. Full Pipeline Training
Train the complete model stack from scratch (Scaler $\to$ ST3 $\to$ EVT $\to$ CT-DGNN $\to$ XGBoost $\to$ DQN RL-Fusion):
```bash
python main.py --mode train --config config.yaml
```

### 3. Comprehensive Evaluation
Evaluate trained models on custom test datasets:
```bash
python main.py --mode evaluate --file "Dataset/Testing CIC/Standardized_500K_Test_Set.csv"
```

### 4. Component Ablation Studies
Run comparative ablation experiments (KAN vs MLP, RoPE vs Sinusoidal, MPS vs Dense):
```bash
python main.py --mode ablation
```

### 5. Retrain the DQN RL-Fusion Agent
Fine-tune or retrain only the fusion decision policy on new traffic characteristics:
```bash
python retrain_rl.py
```

### 6. Run Latency Profiling Suite
Benchmark processing latency and throughput across hardware components:
```bash
python benchmarks/eval_latency_benchmarks.py
```

---

## 📁 Directory Layout

```text
Typhon/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Multi-OS automated CI workflow
├── .gitignore                         # Hardened git ignore rules
├── .gitattributes                     # Normalizes line endings
├── LICENSE                            # Apache 2.0 License
├── CONTRIBUTING.md                    # Contribution guidelines
├── README.md                          # Product documentation
├── pyproject.toml                     # Modern Python build config
├── setup.py                           # Package installer script
├── requirements.txt                   # Production dependencies
├── run.bat                            # Windows one-click runner
├── run.sh                             # Linux/macOS one-click runner
├── config.yaml                        # System configuration & hyperparameters
├── main.py                            # Primary execution entry point
├── retrain_rl.py                      # Standalone DQN Fusion retrainer
│
├── models/                            # Core Neural Architectures
│   ├── __init__.py
│   ├── ct_dgnn.py                     # Continuous-Time Dynamic GNN (LTC ODE Cells)
│   ├── st3.py                         # Spatio-Temporal Tensor-Transformer (KAN, RoPE, MPS)
│   └── fusion.py                      # XGBoost wrapper and DQN RL-Fusion Engine
│
├── models_saved/                      # Lightweight Pretrained Weights (~1.3 MB)
│   ├── ct_dgnn.pt                     # Pre-trained CT-DGNN weights
│   ├── st3.pt                         # Pre-trained ST3 AutoEncoder weights
│   ├── rl_qnet.pt                     # Pre-trained DQN Agent weights
│   └── typhon_objects.pkl             # RobustScaler & EVT threshold parameters
│
├── src/                               # Ingestion & Preprocessing Modules
│   ├── __init__.py
│   ├── input_handler.py               # PCAP and CSV unified ingestion handler
│   ├── feature_extractor.py           # Scapy-based 33-feature flow extractor
│   ├── data_loader.py                 # Out-of-core streaming CSV loader
│   ├── drift_detector.py              # Jensen-Shannon statistical drift engine
│   ├── mitre_mapper.py                # MITRE ATT&CK tactical threat mapper
│   ├── reporting.py                   # Automated Markdown & JSON audit reporting
│   ├── create_standardized_500k_dataset.py # Zero-leakage dataset builder
│   └── utils.py                       # Hardware telemetry and logger utilities
│
├── benchmarks/                        # Performance & Verification Harnesses
│   ├── __init__.py
│   ├── eval_comprehensive_500k.py     # 500k flow validation benchmark
│   ├── eval_latency_benchmarks.py     # Microsecond latency profiling tool
│   ├── eval_st3_zeroday.py            # Zero-day anomaly benchmark
│   ├── run_scenario.py                # Multi-scenario traffic tests
│   ├── train_baselines.py             # Classical baseline models (RF, SVM, XGB)
│   ├── check_labels.py                # Dataset label validator
│   └── results/
│       └── baseline_results.json      # Serialized baseline evaluation results
│
├── data/                              # Flow Datasets & Samples
│   ├── README.md                      # Guide on full dataset setup
│   ├── sample_flows.csv               # 200-flow sample dataset for instant smoke testing
│   ├── raw/                           # Raw input mount (.gitkeep)
│   └── processed/                     # Preprocessed data mount (.gitkeep)
│
├── docs/                              # Technical Documentation & Guides
│   ├── README.md
│   ├── 1_User_Manual.md               # User operational manual
│   ├── 2_Technical_Documentation.md   # Deep technical specification
│   ├── Unified_DoS_DDoS_XGBoost_Kitsune_PRD.txt # Product Requirements Document
│   └── Typhon_Pipeline_Architecture.drawio # Editable Draw.io architecture diagram
│
├── outputs/                           # Benchmark Traces & Reports
│   ├── README.md
│   ├── standardized_500k_results.json # Verified 500k benchmark results
│   ├── latency_benchmark_results.json # Latency benchmarks per component
│   └── fresh_evaluation_results.json  # Concept drift evaluation traces
│
└── reports/                           # Verification & System Audit Reports
    ├── Typhon_Complete_Verification_and_Benchmarking_Report.md
    └── Typhon_Implementation_Plan.md
```

---

## 🎯 MITRE ATT&CK Threat Mapping

Typhon maps detected anomalies directly to the **MITRE ATT&CK for Enterprise** framework:

| Technique ID | Attack Vector | Detected Flow Characteristics | Mitigation Action |
| :--- | :--- | :--- | :--- |
| **T1498.001** | Direct Volumetric Floods | High packet rate, collapsed window sizes | Rate-limiting, IP sinkholing |
| **T1498.002** | Reflection / Amplification | Extreme byte-to-packet ratios (DNS, NTP, SSDP) | Upstream BGP Flowspec filter |
| **T1499.001** | OS State Exhaustion | SYN floods, anomalous forward IAT distributions | SYN cookies, TCP state tracking |
| **T1499.003** | Application Resource Exhaustion | HTTP GET/POST floods, anomalous payload lengths | WAF challenge, connection concurrency caps |

---

## 📄 License

This software is released under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for complete details.
