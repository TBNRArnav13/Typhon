# Chimera IDS v4.0: Active Engine vs. Workbench Modules

This document clarifies exactly which of the 30 novel features are actively "bolted into" the live detection engine, and which serve as "plug-and-play" modules for future research.

---

## Part 1: The "Bolted-In" Features (Active Pipeline)

These 14 features are actively wired into the `train_v4` and `detect` pipelines. Every network packet processed by Chimera flows through these mechanisms.

### 1. B-DAP Engine (Benign-Data Alignment Processor)
*The unsupervised spatial model that looks at a single packet's geometry.*

| Feature | Where is it used? | How is it used? | Why is it used? |
|---------|------------------|-----------------|-----------------|
| **AttentionGate** | `models/bdap.py` (Input layer) | Multiplies incoming features by a learned weight before compression. | Forces the network to focus on the specific feature (e.g., timing vs size) that is acting anomalous right now. |
| **VAE Bottleneck** | `models/bdap.py` (Middle layer) | Compresses data into a probability distribution (mean/variance) instead of a fixed point. | Smooths out the latent space, making the model far more robust against slightly mutated zero-day attacks. |
| **KAN Autoencoders** | `models/bdap.py` (Parallel to VAE) | Processes features using learnable spline curves per edge. Fused with VAE via a learned gate. | Splines capture highly non-linear feature relationships that standard neurons (ReLUs) miss entirely. |
| **Manifold Mixup** | `models/bdap.py` (Training only) | Interpolates hidden states between two random benign packets. | Eliminates "blind spots" in the model's understanding of normal traffic. |
| **Differential Privacy** | `models/bdap.py` (Training only) | Injects Laplacian noise into the input gradients. | Mathematically ensures an attacker cannot reverse-engineer the traffic payload from model weights. |
| **EVT Thresholding** | `models/bdap.py` (Output logic) | Fits a Generalized Pareto Distribution (GPD) to the error tails. | Replaces rigid standard deviations with dynamic thresholds built for chaotic, heavy-tailed network traffic. |

### 2. T-RAD Engine (Temporal Reconstruction Anomaly Detector)
*The unsupervised temporal model that looks at packet sequences over time.*

| Feature | Where is it used? | How is it used? | Why is it used? |
|---------|------------------|-----------------|-----------------|
| **Multi-Scale TCN** | `models/transformer_ad.py` (Input stack) | Runs 4 separate dilated convolutions (dilation 1, 2, 4, 8) over the packet sequence. | Captures both micro-millisecond bursts and macro-minute trends simultaneously. |
| **LTC Neural Networks** | `models/transformer_ad.py` (Middle layer) | Passes TCN output through an Ordinary Differential Equation (ODE). | Adapts the network's "clock speed" to the irregular, chaotic timing gaps of real network traffic. |
| **RoPE Encoding** | `models/transformer_ad.py` (Pre-Transformer) | Applies Meta's Llama-3 rotational matrices to packet positions. | Allows the Transformer to process an infinitely long stream of packets without losing track of sequence order. |

### 3. Fusion & System Automation
*The overarching logic that ties the models together.*

| Feature | Where is it used? | How is it used? | Why is it used? |
|---------|------------------|-----------------|-----------------|
| **Quantum Tensor Networks (QTN)** | `models/hybrid.py` (Parallel voter) | Contracts 33 features through a Matrix Product State (MPS) tensor chain. | Captures ultra-high-order correlations (e.g., how feature 1, 14, and 33 interact simultaneously). |
| **RAG-GAN** | `run.py` (End of training) | Finds weak spots, generates 5,000 synthetic attack variants, and retrains XGBoost. | Proactively hardens the supervised model against zero-days *before* they happen. |
| **C2E2 Early Exit** | `models/hybrid.py` (Routing logic) | If XGBoost hits >99.5% confidence, it skips B-DAP/T-RAD entirely. | Saves ~40% inference latency, allowing edge switches to maintain line-rate speed on obvious attacks. |
| **Multi-Tiered Classification** | `models/hybrid.py` (Verdict logic) | Outputs specific tiers: *Critical Anomaly*, *Suspicious (Temporal)*, *Suspicious (Spatial)*. | Gives SOC analysts granular alert triaging instead of a binary Yes/No. |
| **MITRE ATT&CK Mapper** | `src/reporting.py` (Report Gen) | Maps reasons (e.g., "DNS Amplification") to IDs (e.g., "T1498.002"). | Integrates seamlessly with enterprise threat intelligence platforms. |

---

## Part 2: The "Plug-and-Play" Workbench Modules

These 16 features are fully coded in `models/chimera_v3_modules.py` and `chimera_v4_modules.py`. They are NOT currently slowing down the main engine, but exist as modular puzzle pieces that can be swapped in for specific research experiments.

### Category: Alternative Core Architectures
*Can be swapped in to replace B-DAP or T-RAD.*
* **ST-GNN (Spatio-Temporal GNN):** Treats IP addresses as nodes in a graph. *Use case: Detecting highly coordinated, multi-node botnets.*
* **SNN (Spiking Neural Networks):** Converts features into biological spike trains. *Use case: Ultra-low power neuromorphic hardware deployment.*
* **HDC Encoder (Hyperdimensional):** Projects data into 10,000-dimensional orthogonal vectors. *Use case: Running anomaly detection on resource-constrained IoT devices.*
* **Feature Capsules:** Groups features into voting capsules. *Use case: Enhancing interpretability of which specific feature groups failed.*

### Category: Advanced Training & Robustness
*Can be injected into the training loop to harden models.*
* **Flow-SimCLR:** Self-supervised contrastive learning. *Use case: Pre-training on massive unlabelled datasets before fine-tuning.*
* **Defensive Distillation:** Softens probability labels. *Use case: Defending against attackers using adversarial ML to evade detection.*
* **CTGAN Replay:** Standard tabular GAN. *Use case: Legacy synthetic generation (superseded by RAG-GAN).*
* **EWC (Elastic Weight Consolidation):** Prevents catastrophic forgetting. *Use case: Continual learning, allowing the model to learn new traffic without forgetting the old.*

### Category: System Enhancements & Analytics
*Tools that sit outside the neural networks.*
* **Flow-SHAP:** Explainer module. *Use case: Visually mapping which feature caused an anomaly alert.*
* **Concept Drift Detector:** KL-divergence monitor. *Use case: Running as a cron job to alert admins when the network baseline has fundamentally shifted, requiring a retrain.*
* **Federated Enclave:** Secure multi-party computation simulation. *Use case: Training a shared model across 5 different hospitals without sharing raw PCAPs.*
* **RL-Fusion Agent:** Reinforcement learning router. *Use case: Dynamically changing which model to trust based on the time of day or network load.*
* **Sinkhorn Attention:** Sparse attention matrices. *Use case: Expanding T-RAD's context window from 100 packets to 10,000 packets.*
* **QAT (Quantization-Aware Training):** INT8 compression. *Use case: Compressing the final weights to deploy on an FPGA.*
* **Latent Boosting / Temporal Deltas:** Feature engineering tools. *Use case: Feeding unsupervised signals directly into the supervised XGBoost model.*

---

## Part 3: Architecture Flow Diagram

```text
========================================================================
                      THE GARAGE (Workbench)
[ST-GNN] [SNN] [Flow-SimCLR] [EWC] [Federated Enclave] [Flow-SHAP] ...
(16 fully implemented modules available for custom research experiments)
========================================================================

                                  ||
                                  || (Only 14 modules bolted in)
                                  ▼

========================================================================
                     THE ACTIVE ENGINE (v4.0)
========================================================================
                         RAW NETWORK PACKET
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    [XGBoost (Trees)]    [B-DAP (Spatial)]      [T-RAD (Temporal)]
           │                     │                     │
           │             • AttentionGate        • Multi-Scale TCN
           │             • VAE Bottleneck       • LTC Dynamics (ODE)
           │             • KAN Autoencoders     • RoPE Encoding
           │             • DP Noise Injection          │
           │             • Manifold Mixup              │
           │             • EVT Thresholding            │
           │                     │                     │
           └──────────┐          │          ┌──────────┘
                      │          ▼          │
                      │   [QTN (Tensor MPS)]│
                      │          │          │
                      ▼          ▼          ▼
           ┌──────────────────────────────────────────────┐
           │             HYBRID FUSION LOGIC               │
           │                                              │
           │  1. C2E2 Early Exit (Fast-path for XGBoost)  │
           │  2. Triple Consensus (B-DAP + T-RAD + QTN)   │
           │  3. Multi-Tiered Verification:               │
           │     → Critical Anomaly                       │
           │     → Suspicious (Temporal / Spatial)        │
           │     → DoS / DDoS                             │
           └─────────────────────┬────────────────────────┘
                                 │
                     [MITRE ATT&CK Mapping]
                                 │
                      [Report Generation]
```
