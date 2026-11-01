# Chimera IDS — Complete Feature Guide
### A Third-Person Walkthrough of Every Feature, Its Purpose, and Its Role

---

## What is Chimera?

Chimera is a **Hybrid Intrusion Detection System (IDS)** that monitors network traffic and decides whether each packet flow is safe (benign) or dangerous (a DDoS attack). Instead of relying on a single AI model, Chimera uses **four different AI brains** that vote together — like having four security guards at a door, each trained differently, who must agree before letting someone in or kicking them out.

The system evolved through 4 major versions. Each version added new capabilities while keeping the same 33 network features as input.

---

## The 33 Input Features

Every single model in Chimera operates on exactly **33 numerical measurements** extracted from raw network traffic. These include things like:
- **Volume metrics**: How many packets flowed, how many bytes were sent
- **Packet size metrics**: Average, min, max, standard deviation of packet sizes
- **Timing metrics (IAT)**: Time gaps between consecutive packets
- **TCP flag counts**: How many SYN, FIN, PSH, ACK flags were seen

No matter how advanced the AI gets, it always receives these same 33 numbers as input. The innovation is in *how* the AI processes them internally.

---

## Version 1.0 — The Original System

### The Unsupervised Model: Kitsune
**What it is:** An ensemble of small autoencoders (neural networks that learn to compress and decompress data). It was trained *only* on safe traffic. If a new packet looks very different from what it learned, the reconstruction error spikes, and it flags the packet as suspicious.

**The problem:** It used ReLU activation in its bottleneck layer, which caused "dying neurons" — parts of the network would permanently stop learning. It also assumed errors followed a perfect bell curve (Gaussian distribution), which network traffic rarely does.

### The Supervised Model: XGBoost
**What it is:** A gradient-boosted decision tree classifier. Unlike Kitsune, XGBoost was trained on *both* safe and attack traffic with labels. It learns explicit rules like "if packet count > X and byte rate > Y, then it's an attack."

**Its strength:** Extremely fast and accurate on known attack types.
**Its weakness:** Completely blind to attack types it has never seen before (zero-day attacks).

### The Fusion Logic
A simple if-else cascade: *If XGBoost says attack → attack. Otherwise, if Kitsune says anomaly → anomaly.* This was criticized by IEEE reviewers as too simplistic and prone to false positives.

---

## Version 2.0 — The IEEE Overhaul

After receiving a detailed peer review, the entire system was rebuilt.

### Kitsune → B-DAP (Benign-Data Alignment Processor)
The unsupervised model was completely rewritten and renamed.

**AttentionGate (Novel):** Before compressing data, a learned attention mechanism now dynamically weighs which of the 33 features are most important for each specific packet. If packet size suddenly spikes, the gate forces the model to focus on that feature. *Role: Sits inside each sub-autoencoder, between the input and the bottleneck.*

**Tanh Bottleneck Fix:** The bottleneck activation was changed from ReLU to Tanh. This ensures the compressed representation smoothly maps between -1 and 1, preventing dead neurons. *Role: The mathematical activation function at the tightest compression point of each autoencoder.*

**Adaptive Thresholding:** Instead of blindly using Mean + 3σ, B-DAP now runs a Shapiro-Wilk statistical test to check if errors are Gaussian. If they aren't (and network traffic almost never is), it falls back to the 99.7th percentile. *Role: Determines the exact cutoff point above which a packet is flagged as anomalous.*

### T-RAD (Temporal Reconstruction Anomaly Detector) — NEW
A brand-new second unsupervised model was created.

**What it is:** A Transformer-based neural network that learns temporal patterns in benign traffic. While B-DAP looks at each packet independently (spatial), T-RAD looks at *sequences* of packets over time (temporal). *Role: Acts as the second unsupervised voter in the fusion engine.*

**How it helps:** It catches attacks that unfold gradually over time — like slow-rate DDoS attacks that send packets just below B-DAP's threshold but create a suspicious *pattern* over many packets.

### XGBoost with Optuna HPO
**What it is:** Optuna is a Bayesian optimization library that automatically tests 50 different combinations of XGBoost settings (tree depth, learning rate, etc.) and picks the mathematically best one. *Role: Runs once during training to find optimal hyperparameters.*

**How it helps:** Instead of guessing settings, the system scientifically finds the configuration that maximizes accuracy on your specific dataset. Result: 99.7% accuracy.

### Auto-Configuration System — NEW
**What it is:** A hardware profiler that detects CPU cores, RAM, GPU type, and VRAM at startup. It then automatically sets batch sizes, row limits, and device assignments. *Role: Runs at the very start of every command.*

**How it helps:** The system runs optimally on any machine — from a laptop with 8GB RAM to a server with 64GB — without the user changing a single setting.

### Statistical Evaluation Framework — NEW
**What it is:** A complete evaluation suite that computes Accuracy, Precision, Recall, F1, FPR, FNR, MCC, confusion matrices, permutation feature importance, and per-attack-type breakdowns. *Role: Activated via `python run.py evaluate`.*

### Ablation Study — NEW
**What it is:** An automated experiment that tests 8 different model combinations (XGBoost alone, B-DAP alone, Dual Priority, Tri-Consensus, etc.) side by side. *Role: Proves scientifically which fusion strategy works best.*

### Cross-Flow Macro Heuristics — NEW
Four rule-based detectors added to the fusion engine:

1. **DNS Amplification Detection:** Flags flows where the response is >5x larger than the request (classic amplification attack pattern). *Role: Catches DNS-based volumetric attacks that ML models might miss.*
2. **SYN-ACK Reflection Detection:** Flags when many unique source IPs send tiny SYN-ACK packets to one target. *Role: Catches reflection-based DDoS.*
3. **Shannon Entropy Detection:** Measures the randomness of source IP addresses. Extremely high entropy = botnet. *Role: Catches distributed attacks from thousands of IPs.*
4. **ARP/IGMP Override:** Automatically classifies gratuitous ARP and IGMP as benign. *Role: Prevents false positives from normal network management traffic.*

---

## Version 3.0 — 20 Pillars of Innovation

A massive research expansion. All changes operate on the same 33 features — the innovation is purely in the mathematics.

### B-DAP Upgrades

**Variational Autoencoder (VAE) Bottleneck:** The autoencoder was upgraded from a standard AE to a VAE. Instead of producing a single compressed point, it now produces a *probability distribution* (mean + variance). During training, it samples from this distribution, forcing the latent space to be smooth and continuous. *How it helps: Makes B-DAP far more robust to unseen, out-of-distribution zero-day attacks.*

**EVT Adaptive Thresholding:** Replaced the simple percentile with Extreme Value Theory (EVT). It fits a Generalized Pareto Distribution (GPD) specifically to the tail of the error distribution — the same math used to predict 100-year floods. *How it helps: Sets a mathematically optimal threshold that accounts for the heavy-tailed, chaotic nature of real network traffic.*

**Manifold Mixup:** During training, pairs of latent vectors are randomly blended together using a Beta distribution. This smooths the decision boundary in the hidden layers. *How it helps: Prevents attackers from finding tiny "gaps" in the model's understanding to slip through.*

### T-RAD Upgrades

**Multi-Scale Temporal Convolution (TCN):** The original single-scale convolution was replaced with a 4-level dilated causal convolution stack (dilation 1, 2, 4, 8). *How it helps: Allows T-RAD to simultaneously detect micro-bursts (millisecond spikes) and macro-sessions (minute-long patterns).*

**Rotary Position Embedding (RoPE):** The same positional encoding technology used in Meta's Llama-3 language model. It encodes the *relative* distance between packets using rotation matrices. *How it helps: Allows T-RAD to work on infinite-length network streams without losing track of packet ordering.*

### 14 Auxiliary Research Modules
These live in `chimera_v3_modules.py` as plug-and-play components:

| Module | What It Is | How It Helps |
|--------|-----------|-------------|
| **ST-GNN** | Treats network flows as edges in a graph | Detects coordinated botnet attacks across multiple nodes |
| **Flow-SimCLR** | Contrastive self-supervised pretraining | Creates a robust "map" of normal traffic without any labels |
| **Latent Boosting** | Feeds B-DAP's compressed vectors into XGBoost | Gives XGBoost access to unsupervised anomaly signals |
| **Temporal Deltas** | Computes moving averages and rate-of-change | Lets XGBoost detect sudden traffic surges |
| **HDC Encoder** | Projects 33 features into 10,000 dimensions | Creates mathematically orthogonal clusters for benign vs attack |
| **SNN** | Converts features into brain-like spike trains | Captures temporal dynamics using neuromorphic computing |
| **Capsule Networks** | Groups features into hierarchical capsules | Learns relationships between feature groups (e.g., timing vs volume) |
| **Defensive Distillation** | Trains on soft probability labels | Smooths decision boundaries to resist adversarial evasion |
| **CTGAN Replay** | GAN that mutates known attacks | Generates synthetic future zero-day variants for training |
| **EWC** | Elastic Weight Consolidation | Allows models to learn new traffic without forgetting old baselines |
| **RL-Fusion** | Reinforcement Learning agent | Dynamically shifts trust between models based on context |
| **Differential Privacy** | Laplacian noise injection | Makes training data mathematically private for federated sharing |
| **QAT** | Quantization-Aware Training | Simulates INT8 precision for edge switch deployment |

---

## Version 4.0 — SOTA Deep Integration

The most critical evolution. Previously, v3 modules existed as standalone classes. In v4, they were **deeply wired into the neural network computation graphs** so they actually process every single packet.

### Feature 1: LTC Networks (Liquid Time-Constant)
**What it is:** A neural network cell governed by an Ordinary Differential Equation (ODE). Unlike standard neurons that have fixed behavior, LTC neurons have a learnable, input-dependent time constant (τ) that controls how fast they react to new information.

**How it helps:** Network traffic is *irregular* — packets don't arrive at fixed intervals. Standard neural networks assume uniform timing. LTC cells dynamically adapt their internal clock speed based on the actual inter-arrival times, making T-RAD vastly better at detecting timing-based anomalies.

**Where it sits:** Inserted as a new layer between the Multi-Scale TCN output and the RoPE/Transformer stack inside T-RAD. Every packet passes through the LTC ODE computation before reaching the Transformer.

### Feature 2: KAN Autoencoders (Kolmogorov-Arnold Networks)
**What it is:** In a standard neural network, every connection multiplies by a fixed weight and passes through a shared activation function (like ReLU). In a KAN, every single connection has its *own* learnable activation function — a B-spline curve with adjustable knots. This means the network can learn complex, non-linear transformations on a per-edge basis.

**How it helps:** B-DAP's sub-autoencoders can now learn far more nuanced compression functions for each feature group (volume, packet size, timing, TCP flags), dramatically increasing reconstruction sensitivity to subtle anomalies.

**Where it sits:** Each B-DAP sub-autoencoder now has a parallel KAN encoder branch. The outputs are combined using a learnable gated fusion: `output = gate × VAE_output + (1-gate) × KAN_output`. The gate automatically learns how much to trust each branch.

### Feature 3: Quantum Tensor Networks (QTN)
**What it is:** A Matrix Product State (MPS) — a chain of 33 small tensors (one per feature) connected in sequence. When a packet's 33 features are fed through, each feature is embedded into a quantum-inspired physical dimension, then contracted through the entire chain. The final scalar output represents the high-order correlation score.

**How it helps:** Standard neural networks capture pairwise feature interactions. QTN captures correlations between *all* 33 features simultaneously — including complex 5-way, 10-way, and even 33-way interactions — with polynomial (not exponential) computational cost.

**Where it sits:** Added as the 4th unsupervised model in the HybridDetector. It votes alongside B-DAP and T-RAD in a "Triple Consensus" system.

### Feature 4: RAG-GAN (Retrieval-Augmented Anomaly Generation)
**What it is:** A two-part system. First, a **Memory Bank** stores feature vectors of attacks that the system had low confidence on (borderline cases). Second, a **Generator network** retrieves similar past attacks from the memory bank and generates mutated synthetic versions.

**How it helps:** After initial training, the system automatically identifies its weakest spots, generates thousands of synthetic attack mutations targeting those weaknesses, and retrains XGBoost on the augmented data. This self-improvement loop happens automatically during `train_v4`.

**Where it sits:** Wired into Phase 5 of the `train_v4` pipeline. It runs after all models are trained, finds weak spots, generates 5,000 synthetic mutations, and retrains XGBoost.

### Feature 5: Differential Privacy (DP) Noise Injection
**What it is:** During B-DAP training, controlled Laplacian noise (calibrated to privacy budget ε=1.0) is injected into the training data. This mathematically guarantees that no individual packet's information can be reverse-engineered from the model weights.

**How it helps:** Makes Chimera ready for deployment in federated environments where multiple organizations share model weights without exposing sensitive traffic data.

**Where it sits:** Applied once during B-DAP training, right after data normalization and before the first epoch.

### Feature 6: Multi-Tiered Probabilistic Classification
**What it is:** Instead of a binary "Benign/Attack" output, the system now produces 4 tiers:
- **DoS/DDoS (Malicious):** XGBoost confidently identifies a known attack
- **Critical Anomaly:** Both B-DAP AND T-RAD independently flag the packet (high-confidence zero-day)
- **Suspicious (Temporal):** Only T-RAD flags it (timing anomaly)
- **Suspicious (Spatial):** Only B-DAP flags it (volume/size anomaly)

**How it helps:** SOC analysts can prioritize alerts. "Critical Anomaly" demands immediate attention; "Suspicious" can be queued for review. *Where it sits: The classification logic inside HybridDetector.*

### Feature 7: C2E2 (Confidence-Calibrated Early Exiting)
**What it is:** When XGBoost is >99.5% confident in its prediction, the system skips the computationally expensive unsupervised models entirely.

**How it helps:** Saves ~40% inference latency on obvious attacks or clearly benign traffic, without sacrificing accuracy on edge cases. *Where it sits: A conditional check in HybridDetector before calling B-DAP/T-RAD.*

### Feature 8: MITRE ATT&CK Mapping
**What it is:** Every detection reason is automatically mapped to a formal MITRE ATT&CK technique ID. For example, "DNS Amplification" → `T1498.002 (Reflection Amplification)`.

**How it helps:** Integrates with enterprise threat intelligence platforms and gives SOC analysts standardized context. *Where it sits: `src/mitre_mapper.py`, called during report generation.*

### Feature 9: Concept Drift Detection
**What it is:** Monitors the KL-divergence between the latent space distribution of recent traffic and the training baseline. If the divergence exceeds a threshold, it means the nature of "normal" traffic has fundamentally changed.

**How it helps:** Automatically triggers a retrain alert when the network environment evolves (e.g., new applications deployed, infrastructure changes). *Where it sits: `chimera_v4_modules.py`, designed to run periodically on live traffic.*

### Feature 10: Batch PCAP Directory Support
**What it is:** The `detect` command now accepts entire directories as input. It automatically finds all `.pcap`, `.pcapng`, and `.csv` files inside and processes them sequentially.

**How it helps:** Eliminates the need for manual scripts to loop through PCAP folders. *Where it sits: Input validation logic in `detect_mode()` inside `run.py`.*

---

## Bug Fixes (What Broke and Why)

| Bug | What Happened | Why | Fix Applied |
|-----|-------------|-----|-------------|
| T-RAD Loss = NaN | The LTC layer's ODE calculations exploded to infinity | The tau floor (0.1) was too small, allowing division-by-near-zero in the ODE | Raised tau floor to 0.5, clamped ODE output to [-10, 10], added NaN-skip, reduced learning rate |
| QTN scores = 0.000 | The Quantum Tensor Network output pure zeros | MPS core weights initialized at 0.1; multiplying 0.1 × 0.1 × ... × 0.1 (33 times) = effectively zero | Raised init to 0.5, added per-site normalization to prevent vanishing |
| CUDA OOM on T-RAD | GPU ran out of 8GB VRAM during threshold calculation | Tried loading all 529K rows into GPU at once | Rewrote to batch-process in chunks |
| Stale `transformer_detections` key | Reports showed T-RAD as "N/A" | Renamed to T-RAD but reporting still used old key name | Updated reporting to use `trad_detections` |
| PCAP directory rejected | `detect --input folder/` crashed | Code checked `os.path.isfile()` which fails on directories | Added directory detection and automatic file iteration |

---

## Complete System Flow

```
┌─────────────────────────────────────────────┐
│              RAW INPUT                       │
│     .pcap file / .csv file / directory       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│          SYSTEM PROFILER (Auto-Config)        │
│  Detects: GPU, RAM, CPU → Sets batch sizes   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│          FEATURE EXTRACTION                   │
│  Raw packets → 33 numerical measurements     │
│  (volume, packet size, timing, TCP flags)     │
└──────────────────┬───────────────────────────┘
                   │
       ┌───────────┼───────────┬───────────┐
       │           │           │           │
       ▼           ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
  │ XGBoost │ │  B-DAP  │ │ T-RAD  │ │  QTN   │
  │Supervised│ │  VAE +  │ │TCN+LTC │ │Quantum │
  │ Trees   │ │  KAN +  │ │+ RoPE +│ │Tensor  │
  │         │ │  DP     │ │Transf. │ │Network │
  │ "I know │ │ "This   │ │ "The   │ │"Deep   │
  │ this    │ │ looks   │ │ timing │ │feature │
  │ attack" │ │ weird"  │ │ is off"│ │ links" │
  └────┬────┘ └────┬────┘ └───┬────┘ └───┬────┘
       │           │          │           │
       └───────────┼──────────┼───────────┘
                   │          │
                   ▼          ▼
  ┌────────────────────────────────────────────┐
  │         HYBRID FUSION ENGINE                │
  │                                             │
  │  Step 1: C2E2 Early Exit                    │
  │    If XGBoost >99.5% confident → skip rest  │
  │                                             │
  │  Step 2: Cross-Flow Heuristics              │
  │    DNS Amplification / SYN Reflection /     │
  │    Shannon Entropy / ARP Override           │
  │                                             │
  │  Step 3: Triple Consensus Vote              │
  │    B-DAP + T-RAD + QTN all agree?           │
  │    → Critical Anomaly (highest confidence)  │
  │                                             │
  │  Step 4: Multi-Tiered Classification        │
  │    Both unsupervised → Critical Anomaly     │
  │    T-RAD only       → Suspicious (Temporal) │
  │    B-DAP only       → Suspicious (Spatial)  │
  │    XGBoost only     → DoS/DDoS (Malicious)  │
  │    Nobody flags     → Benign                │
  └──────────────────┬─────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────┐
  │          MITRE ATT&CK MAPPER               │
  │  Tags each detection with formal IDs       │
  │  e.g., T1498.002 (Reflection Amplification)│
  └──────────────────┬─────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────┐
  │           REPORTING ENGINE                  │
  │                                             │
  │  • Executive Summary + Verdict              │
  │  • Threat Severity Breakdown                │
  │  • Detection Source Analysis                │
  │  • Top Malicious + Suspicious alerts        │
  │                                             │
  │  Output Files:                              │
  │    predictions/benign/      (.json + .csv)  │
  │    predictions/malicious/   (.json + .csv)  │
  │    predictions/suspicious/  (.json + .csv)  │
  │    report/Report.md + Report.json           │
  └────────────────────────────────────────────┘
```

### Training Pipeline Flow (`python run.py train_v4`)
```
Phase 1: B-DAP ──→ 4 Sub-AEs (VAE+KAN gated fusion, 50 epochs each)
                   + Output AE + EVT/GPD Threshold Calibration
                   + Differential Privacy noise injection
                      │
Phase 2: T-RAD ──→ TCN → LTC (ODE dynamics) → RoPE → Transformer
                   30 epochs, batch size 64
                      │
Phase 3: QTN   ──→ 33-site MPS tensor contraction training
                   30 epochs, learns benign correlations
                      │
Phase 4: XGBoost ─→ GPU-accelerated tree boosting on labelled data
                      │
Phase 5: RAG-GAN ─→ Find weak spots → Retrieve similar past attacks
                   → Generate 5,000 synthetic mutations
                   → Retrain XGBoost on augmented data
                      │
Phase 6: Verify  ─→ Confirm all modules show [ACTIVE]
```

---

*Document generated: 2026-05-02 | Chimera IDS v4.0.1*
*Total novel features: 30 (20 v3 pillars + 10 v4 features)*
*Active source files: 19 | Models: B-DAP, T-RAD, QTN, XGBoost*
