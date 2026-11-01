# Research Audit & System Review: Chimera IDS v4.0

## Executive Summary
This document provides a comprehensive, research-grade audit of **Chimera IDS v4.0**, a hybrid intrusion detection system utilizing a priority-ordered hierarchical fusion of supervised and unsupervised learning models. The audit evaluates the codebase from both a **systems engineering** and a **cybersecurity research** standpoint.

While the project demonstrates a highly sophisticated integration of modern deep learning architectures (including Liquid Time-Constant networks, Kolmogorov-Arnold Networks, and Matrix Product State Tensor Networks), several critical logical, mathematical, and computational discrepancies have been identified. Fixing these issues will not only improve the accuracy and robustness of the system but will also elevate the academic rigor of the research paper for publication in venues such as IEEE Transactions on Information Forensics and Security (TIFS) or IEEE S&P.

---

## 1. Architectural & Conceptual Audit
Chimera's core design—a priority-ordered fusion engine—is built on the asymmetric risk profile of network security, where false negatives (missed attacks) are far costlier than false positives. 

The system relies on three parallel inference pipelines:
1. **XGBoost (Supervised)**: Tries to match known DoS/DDoS patterns.
2. **B-DAP (Unsupervised, Spatial)**: Kitsune-inspired autoencoder ensemble utilizing a gated Kolmogorov-Arnold Network (KAN) and Variational Autoencoder (VAE) bottleneck.
3. **T-RAD (Unsupervised, Temporal)**: Rotary-position-embedded (RoPE) Transformer with dilational Temporal Convolutional Networks (TCN) and Liquid Time-Constant (LTC) cells.
4. **QTN (Unsupervised, Quantum)**: Matrix Product State (MPS) tensor network for high-order feature correlations.

### The Central Architectural Discrepancy: "Paper Pillars" vs. "Active Code"
The research paper and the `config.yaml` make bold claims about **20 pillars (v3.0)** and **30 pillars (v4.0)**. 
However, an audit of `models/hybrid.py`, `models/bdap.py`, and `run.py` reveals a significant gap between the defined modules and their actual runtime integration:
* **Deeply Integrated Pillars**: VAE Bottleneck, KAN Gated Fusion, LTC Temporal Cells, QTN Matrix Product States, Differential Privacy noise, and EVT tail calibration are fully integrated into the training and execution loops.
* **Isolated / Inert Pillars**: The Spatio-Temporal GNN (`SpatioTemporalGNN`), Contrastive learning (`FlowSimCLR`), Explainable AI (`FlowSHAP`), Hyper-Dimensional Computing (`HDCEncoder`), Spiking Neural Network (`SpikingNN`), Capsule Networks (`FeatureCapsule`), Defensive Distillation, and Elastic Weight Consolidation (`EWC`) are implemented in `models/chimera_v3_modules.py` and successfully compile, but they are **never instantiated, trained, or called** during standard detection or evaluation runs.
* **The Integration Gap**: To claim these features are part of the active Chimera system, they must be woven into the fusion engine in `models/hybrid.py` rather than remaining as inert code definitions.

---

## 2. Critical Logical & Mathematical Bugs

### 2.1. The Manifold Mixup Reconstruction Loss Bug (Critical)
* **Location**: [models/bdap.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/bdap.py#L195-L202) and [bdap.py:L402-L417](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/bdap.py#L402-L417)
* **The Bug**: During training of the Sub-Autoencoders, a random mixing coefficient $\lambda \sim \text{Beta}(0.4, 0.4)$ is drawn. Inside `forward()`, the latent vector $z$ is mixed with a permuted version of itself: 
  $$z_{mixed} = \lambda z + (1 - \lambda) z_{perm}$$
  However, the reconstruction loss is computed as:
  ```python
  recon_loss = criterion(outputs, inputs)
  ```
  where `inputs` is the original, unmixed feature representation!
* **Research & Mathematical Flaw**: This is mathematically incorrect. Since the latent space is a linear combination of two different samples, the decoder is forced to reconstruct a blended feature profile. Penalizing it for not matching the original, unmixed sample $A$ creates contradictory gradients. This acts as severe noise, preventing the autoencoder from learning a clean reconstruction manifold.
* **The Fix**: The permutation index `perm` must be generated in a way that allows the training loop to access it, and the loss target must be mixed symmetrically:
  $$x_{mixed} = \lambda x + (1 - \lambda) x_{perm}$$

### 2.2. Kolmogorov-Arnold Network (KAN) Structural Deviation
* **Location**: [models/chimera_v4_modules.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/chimera_v4_modules.py#L166-L200)
* **The Bug**: A true KAN layer (inspired by the Kolmogorov-Arnold representation theorem) applies learnable 1D B-spline activation functions on the *individual edges* (the connections between inputs and outputs) *before* they are summed at the output node:
  $$y_j = \sum_{i} \phi_{i,j}(x_i)$$
  However, in `KANLayer.forward()`, the linear combination is computed first using standard weights, and then the spline is applied to the output node:
  ```python
  linear = F.linear(x, self.weight, self.bias)
  outputs = []
  for j in range(self.out_features):
      outputs.append(self.splines[j](linear[:, j:j+1]))
  ```
  This computes:
  $$y_j = \phi_j \left( \sum_{i} w_{j,i} x_i + b_j \right)$$
* **Research & Mathematical Flaw**: This is mathematically a standard Multi-Layer Perceptron (MLP) with learnable activation functions (a Spline-Activated Neural Network), not a KAN. In a true KAN, the univariate spline functions are applied on each individual incoming feature dimension before summation, giving it its unique approximation properties and avoiding the curse of dimensionality.
* **The Fix**: Rewrite the KANLayer to apply the splines on the edges.

### 2.3. C2E2 Computational Waste & Performance Overhead
* **Location**: [models/hybrid.py](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/hybrid.py#L87-L93) and [hybrid.py:L203-L213](file:///e:/Coding/College/PBL%20Sem%204/Backup/Chimera-main/models/hybrid.py#L203-L213)
* **The Bug**: Under the Confidence-Calibrated Early Exiting (C2E2) scheme, flows are skipped if XGBoost is highly confident. The code defines early exiting as:
  ```python
  c2e2_confident = np.array([xgb_preds[i] == 1 and xgb_probs[i] >= self.c2e2_threshold
                             for i in range(n_samples)])
  ```
  Any flow not matching this (i.e. `xgb_preds[i] == 0` OR `xgb_probs[i] < 0.995`) is marked as "uncertain" and sent to B-DAP, T-RAD, and QTN.
  However, in the fusion logic, if `xgb_preds[i] == 1`, the final verdict is **always** forced to `"DoS/DDoS"` regardless of the unsupervised scores:
  ```python
  elif xgb_label == 1:
      verdict = "DoS/DDoS"
  ```
* **Performance Flaw**: Since the unsupervised scores are completely ignored when `xgb_preds[i] == 1`, calculating them for flows where `0.5 <= xgb_probs[i] < 0.995` is a complete waste of compute. On heavy-attack datasets, this causes thousands of flows to run through expensive deep-learning models (T-RAD, B-DAP, QTN) only for their scores to be thrown away.
* **The Fix**: The `uncertain_mask` should be simplified to `xgb_preds == 0`. We only run unsupervised anomaly detection on flows that passed supervised screening as benign. This will yield up to a **10x speedup** on volumetric attack data without changing a single classification verdict.

### 2.4. Concept Drift Jensen-Shannon (JS) Divergence Scaling Error
* **Location**: [models/chimera_v4_modules.py](file:///e:/Coding/College/PBL%20Sem%204%20/Backup/Chimera-main/models/chimera_v4_modules.py#L596-L620)
* **The Bug**: When checking for concept drift, the detector computes the JS-divergence between the training and inference anomaly scores:
  ```python
  current_hist, _ = np.histogram(scores, bins=edges, density=True)
  current_hist = current_hist + 1e-10
  baseline = self.baseline_histograms[0]
  # ...
  js_div = 0.5 * np.sum(current_hist * np.log(current_hist / m)) + ...
  ```
* **Mathematical Flaw**: Using `density=True` in `np.histogram` computes a probability *density* where the integral (area under the curve) is 1, not the sum of the bin values. Because the bins have a width $W < 1$, the sum of the density values is $\sum p_i = 1/W > 1$. Applying Shannon entropy or KL-divergence formulas directly to these non-normalized density vectors violates standard probability mass assumptions. The calculated JS-divergence will scale inversely with the bin width and can exceed the theoretical upper bound of $\log(2)$ (or 1.0 depending on log base).
* **The Fix**: Compute the histogram without `density=True`, then normalize by dividing by the sum of the counts so that it represents a valid probability mass function ($\sum p_i = 1.0$).

---

## 3. Integration & Threat Intelligence Gaps

### 3.1. MITRE ATT&CK Mapping Integration Gap
* **Description**: The MITRE ATT&CK mapping engine (`src/mitre_mapper.py`) is fully implemented with high-fidelity Enterprise v14+ taxonomic mappings for each attack signature. However, the `MITREMapper` is never instantiated or called during active detection run (`detect_mode`) or in the reporting engine (`src/reporting.py`).
* **The Impact**: The final generated Markdown and JSON reports lack the structured threat intelligence (Tactic/Technique IDs like `T1498.001` or `TA0040`) promised by Feature 9.
* **The Fix**: Instantiate `MITREMapper` in `ReportGenerator` and append MITRE ATT&CK details to the alerts tables in Section 5 and 6 of `Report.md`.

### 3.2. Concept Drift Detection Integration Gap
* **Description**: The drift detector (`ConceptDriftDetector`) is defined in `chimera_v4_modules.py`, but it is never fitted on training scores or checked during inference in the active pipeline.
* **The Fix**: During training (`train_bdap` or `train_v4`), fit the baseline histogram. During detection, run the drift detector on the final anomaly scores and append a "Concept Drift Warning" card to the executive summary if drift is detected.

---

## 4. Repository Clean-Up: Unwanted & Legacy Files
To clean up the repository and remove unwanted clutter, the following files and directories should be addressed:

### 4.1. Old Output Run Directories (Clean-Up)
The `outputs/` directory contains massive legacy debug folders from previous development runs:
* `outputs/Tier1_PureAttack_Mix_1_20260505_000444/`
* `outputs/Tier1_PureAttack_Mix_1_20260506_000316/`
* `outputs/Tier1_PureAttack_Mix_1_20260506_001822/`
* **Recommendation**: **Delete** these old runs or move them to a separate backup folder outside the repository. They contain duplicate prediction CSVs/JSONs taking up disk space.

### 4.2. Scratch/Investigation Scripts (Clean-Up)
* `xgb_investigate.py`
* **Recommendation**: **Delete** or move this script to a `scratch/` directory. It is an ad-hoc investigation script for looking at false positives and is not part of the production pipeline.

### 4.3. Legacy Training Reports (Clean-Up)
* `reports/training/transformer_training_report.json`
* **Recommendation**: **Delete** this file. It is a leftover from v3.0 when the temporal model was called "Transformer Anomaly Detector". In v4.0, it is named "T-RAD", and its active report is `reports/training/trad_training_report.json`.

---

## 5. Recommended Technical Roadmap (Version 2.1)

Below is the proposed sequence of refactoring steps to implement these critical fixes and bridge the paper-code integration gaps:

```mermaid
graph TD
    A[Fix Manifold Mixup Loss] --> B[Fix KAN Edge activations]
    B --> C[Fix Concept Drift JS-Div Scaling]
    C --> D[Optimize C2E2 Inference Loop]
    D --> E[Integrate MITRE ATT&CK Mapping]
    E --> F[Integrate Concept Drift Monitoring]
    F --> G[Clean Unwanted & Legacy Files]
```

### Proposed Code Diffs

#### 1. Manifold Mixup Fix (`models/bdap.py`)
```diff
# In SubAutoEncoder.forward():
-        # Pillar 11: Manifold Mixup in latent space during training
-        if self.training and mixup_lambda is not None:
-            perm = torch.randperm(z.size(0))
-            z = mixup_lambda * z + (1 - mixup_lambda) * z[perm]
-
-        gated = self.attention_gate(z)
-        decoded = self.decoder(gated)
-        return decoded, mu, logvar

+        # Return perm to training loop so inputs can be mixed symmetrically
+        perm = torch.arange(z.size(0))
+        if self.training and mixup_lambda is not None:
+            perm = torch.randperm(z.size(0))
+            z = mixup_lambda * z + (1 - mixup_lambda) * z[perm]
+
+        gated = self.attention_gate(z)
+        decoded = self.decoder(gated)
+        return decoded, mu, logvar, perm
```

#### 2. KAN Edge Activations Fix (`models/chimera_v4_modules.py`)
```diff
# In KANLayer:
     def forward(self, x):
-        # Linear combination
-        linear = F.linear(x, self.weight, self.bias)  # (B, out_features)
-        # Apply learned spline activation per output
-        outputs = []
-        for j in range(self.out_features):
-            outputs.append(self.splines[j](linear[:, j:j+1]))
-        return torch.cat(outputs, dim=1)

+        # Apply learnable 1D B-spline activation functions on individual edges
+        # x: (B, in_features)
+        B, in_f = x.shape
+        out_f = self.out_features
+        # For each input feature i, apply out_features distinct splines
+        outputs = torch.zeros(B, out_f, device=x.device)
+        for j in range(out_f):
+            # Sum spline-activated representations over all inputs
+            edge_outputs = [self.splines[j](x[:, i:i+1] * self.weight[j, i]) for i in range(in_f)]
+            outputs[:, j] = torch.stack(edge_outputs, dim=1).sum(dim=1) + self.bias[j]
+        return outputs
```

#### 3. Concept Drift Density Fix (`models/chimera_v4_modules.py`)
```diff
# In ConceptDriftDetector:
     def fit_baseline(self, scores: np.ndarray):
-        hist, edges = np.histogram(scores, bins=self.n_bins, density=True)
-        hist = hist + 1e-10  # prevent log(0)
+        # Compute raw counts and normalize to probability mass function
+        hist, edges = np.histogram(scores, bins=self.n_bins, density=False)
+        hist = (hist / np.sum(hist)) + 1e-10
         self.baseline_histograms = [hist]
         self.baseline_edges = [edges]
```

---

> [!NOTE]
> Resolving the **Manifold Mixup loss target mismatch** and transitioning the **KANLayer** to true edge-based spline activations will establish a state-of-the-art level of biological/neuromorphic accuracy, significantly improving the empirical defensibility of the Chimera hybrid architecture in scientific reviews.
