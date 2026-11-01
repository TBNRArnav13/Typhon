# Typhon: Continuous-Time Neuromorphic-Tensor Fusion Architecture for Zero-Day Network Intrusion Detection

This implementation plan outlines the design of **Typhon**, a next-generation Hybrid Intrusion Detection System built from a clean slate. Typhon represents a substantial advancement in intrusion detection research. It addresses the core limitations of the Chimera architecture and previous hybrid systems by establishing a mathematically unified, continuous-time spatio-temporal framework coupled with a high-performance, zero-copy feature extraction engine.

---

## User Feedback Incorporated (Version 2.0)

> [!IMPORTANT]
> **1. Wireshark vs. CICFlowMeter Capture Discrepancies**
> Wireshark captures packets using the standard libpcap format, but the data varies depending on network interfaces, hardware offloading, and link-layer encapsulation (e.g. standard Ethernet `DLT_EN10MB`, Loopback `DLT_NULL`, or Linux Cooked Capture `DLT_LINUX_SLL`).
> Additionally, hardware offloading like **TCP Segment Offloading (TSO)** and **Large Receive Offload (LRO)** causes Wireshark to capture aggregated packets that exceed the standard MTU of 1500 bytes, whereas `CICFlowMeter` uses the raw IP header `len` field.
> **Typhon Solution**: The Rust-backed parser (`typhon_parser`) will:
> * Dynamically parse multiple link layers (`Ethernet`, `SLL`, `Null/Loopback`).
> * Handle TSO/LRO by strictly inspecting the IP layer header `len` field rather than trusting `pcap` capture length.
> * Standardize packet sizing computations to ensure complete feature parity.

> [!IMPORTANT]
> **2. Robust CUDA-to-CPU Fallback**
> GPU acceleration remains the first priority. PyTorch modules will use a robust connection helper:
> * Attempts to initialize CUDA.
> * If failure occurs, retries **up to 3 times** with CUDA cache flushing (`torch.cuda.empty_cache()`).
> * If all 3 attempts fail, falls back cleanly and warning-free to CPU execution.

> [!IMPORTANT]
> **3. Clean-Slate Deployment**
> Per your instructions, we are deploying **Typhon directly in the workspace root**, removing all legacy Chimera files to start fresh.

> [!IMPORTANT]
> **4. Performance Targets to Beat**
> We are optimizing Typhon to beat the first research paper's metrics:
> * **Accuracy Target**: $> 99.70\%$
> * **Recall (Detection Rate) Target**: $> 99.95\%$
> * **False Positive Rate Target**: $< 2.72\%$
> * **Zero-Day LOO Detection Target**: $> 95.00\%$

---

## Open Questions
* **None**. All feedback has been aligned. We are ready to begin the clean-slate migration.

---

## Proposed Changes

We will restructure the workspace root, removing old Chimera files and placing Typhon's modular architecture in their place.

### Component 1: Zero-Copy Rust Feature Extractor (Feature Parity)
Provides a compiled Rust library that extracts bidirectional flows directly from raw `.pcap` files with microsecond-level accuracy, matching Java-based `CICFlowMeter` columns exactly.

#### [NEW] [Cargo.toml](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/Cargo.toml)
Configures the Rust crate with dependencies on `pcap-parser`, `pnet`, and `pyo3` for Python bindings.

#### [NEW] [src/lib.rs](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/src/lib.rs)
Core Rust logic for bidirectional flow assembly, active/idle time tracking, TCP flag accumulation, and flow payload length calculations matching `CICFlowMeter`.

---

### Component 2: Continuous-Time Neuromorphic Graph Engine
Implements the continuous-time IP graph modeling to preserve structural network context.

#### [NEW] [models/ct_dgnn.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/ct_dgnn.py)
A PyTorch module defining the **Continuous-Time Dynamic Graph Neural Network**. 
* Nodes representing IP addresses maintain state vectors updated via an Euler-discretized LTC ODE:
  $$\frac{dh(t)}{dt} = -\frac{h(t)}{\tau(t)} + f(h(t), x(t))$$
* Message passing is triggered by incoming flow packets, capturing topological relationships in real-time.

---

### Component 3: Quantum-Inspired Spatio-Temporal Tensor-Transformer (QST3)
The core unsupervised zero-day anomaly detector.

#### [NEW] [models/qst3.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/qst3.py)
* **True KAN Layer**: Implements a native B-spline mapping applied strictly to individual edge connections *before* node aggregation.
* **LTC-Transformer**: Combines RoPE position embedding, multi-scale dilational TCN, and LTC dynamics for multi-scale session representation.
* **MPS Tensor Layer**: Matrix Product State contraction of the 33-dimensional feature space to model complex feature correlations.
* **Adaptive EVT Thresholding**: Fits a Generalized Pareto Distribution (GPD) on reconstruction errors above the 90th percentile to establish mathematically sound detection boundaries.

---

### Component 4: Neuro-Symbolic Reinforcement Learning Fusion Engine
Combines supervised and unsupervised decisions.

#### [NEW] [models/fusion.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/fusion.py)
* **XGBoost Classifier**: Highly tuned gradient boosting classifier for known signatures.
* **DQN Fusion Agent**: A reinforcement learning agent that selects the routing action (e.g., trust XGBoost, invoke QST3, or apply protocol overrides) based on a 6-dimensional state space:
  $$\mathcal{S} = [p_{\text{xgb}}, e_{\text{qst3}}, \mathcal{H}_{\text{entropy}}, \mathcal{V}_{\text{flow}}, \mathcal{L}_{\text{system}}, \delta_{\text{drift}}]$$
* **Reward Function**: Formulated to strictly penalize False Negatives (FN) to ensure high recall:
  $$\mathcal{R} = -(100 \cdot \mathbb{I}_{\text{FN}} + 1 \cdot \mathbb{I}_{\text{FP}}) + 10 \cdot \mathbb{I}_{\text{Correct}}$$

---

### Component 5: Executive Pipeline & SIEM Interface
Orchestrates training, evaluation, and reporting.

#### [NEW] [main.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/main.py)
Main execution script supporting:
* `--mode train`: Fits all models.
* `--mode detect`: Evaluates a `.csv` or `.pcap` file using the Rust parser.
* `--mode evaluate`: Labelled evaluation with statistical reliability reports.

#### [NEW] [utils/reporting.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/utils/reporting.py)
Generates SOC-ready executive summaries with **MITRE ATT&CK Enterprise v14** annotations and concept drift alerts.

---

## Verification Plan

### Automated Experiments & Validation
1. **Feature Parity Test**: We will run both `typhon_parser` and Java `CICFlowMeter` on the same test PCAP and assert that the extracted features are within a $10^{-6}$ numerical tolerance.
2. **Leave-One-Out (LOO) Zero-Day Test**: We will hide one full attack category (e.g., DNS Amplification) from the XGBoost training set and verify that the unsupervised QST3/CT-DGNN engine detects it with a zero-day recall rate $> 95\%$.
3. **Speed & Latency Benchmarks**: We will compare parsing and inference throughput (flows/sec) against the baseline Chimera model, demonstrating the systems-level improvements of the Rust-backed feature extractor and optimized C2E2 execution.
