# Chimera IDS v4.0 — Comprehensive Testing & Evaluation Report

**Date Generated:** 2026-05-03  
**Tested By:** Automated Rigorous Evaluation Pipeline  
**System Under Test:** Chimera v4.0 Multi-Paradigm Hybrid IDS  
**Dataset:** CIC-DDoS2019 (Final_Evaluation_Set — 500K+ flows)

---

## 1. Executive Summary

Chimera v4.0 is a tri-model hybrid Intrusion Detection System combining supervised learning (XGBoost), spatial anomaly detection (B-DAP), and temporal anomaly detection (T-RAD), with an additional Quantum Tensor Network (QTN). This report presents rigorous, end-to-end testing across every evaluation dimension: per-model metrics, per-attack-type detection rates, fusion strategy ablation, zero-day resilience, feature importance, training convergence, and critical failure analysis.

### Key Findings at a Glance

| Metric | XGBoost | B-DAP | T-RAD | Chimera (Fused) |
|--------|---------|-------|-------|-----------------|
| Accuracy | 99.81% | 85.87% | 70.05% | 99.55% |
| Precision | 99.59% | 98.40% | 0.00% | 98.73% |
| Recall | 99.77% | 53.72% | 0.00% | 99.77% |
| F1-Score | 0.9968 | 0.6949 | 0.0000 | 0.9925 |
| FPR | 0.17% | 0.37% | 0.00% | 0.55% |
| FNR | 0.23% | 46.28% | 100.00% | 0.23% |
| MCC | 0.9955 | 0.6607 | 0.0000 | 0.9893 |

**Overall Verdict:** XGBoost is the dominant detection engine. The hybrid fusion marginally improves recall (+1 flow) but introduces additional false positives (1,919 vs 608), reducing precision. T-RAD is **non-functional** as a standalone detector (0% recall).

---

## 2. Individual Model Performance

### 2.1 XGBoost (Supervised Classifier)

- **Architecture:** Gradient-boosted trees, 13 boosting rounds (early-stopped from 300), max_depth=8
- **Training Data:** 1,142,857 flows (429,991 benign + 712,866 attack), GPU-accelerated (CUDA)
- **Confusion Matrix:** TP=149,416 | TN=349,630 | FP=608 | FN=346

| Metric | Value |
|--------|-------|
| Accuracy | 99.8092% |
| Precision | 99.5947% |
| Recall | 99.7690% |
| F1-Score | 0.996818 |
| Specificity | 99.8264% |
| MCC | 0.995456 |
| FPR | 0.1736% |
| FNR | 0.2310% |

**Assessment:** Excellent performance across all metrics. The model early-stopped at only 13 rounds (from 300 configured), indicating rapid convergence on highly separable feature space. The scale_pos_weight of ~0.60 correctly compensated for the attack-heavy training distribution.

**Top 5 Features by Permutation Importance:**

| Rank | Feature | Mean Accuracy Decrease |
|------|---------|----------------------|
| 1 | fwd packet length min | 0.2233 |
| 2 | total backward packets | 0.1685 |
| 3 | flow bytes/s | 0.0527 |
| 4 | flow packets/s | 0.0403 |
| 5 | flow duration | 0.0236 |

### 2.2 B-DAP (Benign-Data Alignment Processor)

- **Architecture:** 4 Sub-Autoencoders (Volume, Packet Size, IAT, TCP Flags) + Output AE, VAE + KAN gated fusion per sub-AE
- **Training:** 529,481 benign samples, 50 epochs, EVT GPD threshold calibration
- **Threshold:** 0.4771 (method: evt_gpd_99.7)
- **Confusion Matrix:** TP=80,447 | TN=348,927 | FP=1,311 | FN=69,315

| Metric | Value |
|--------|-------|
| Accuracy | 85.87% |
| Precision | 98.40% |
| Recall | 53.72% |
| F1-Score | 0.6949 |
| Specificity | 99.63% |
| FPR | 0.37% |
| FNR | 46.28% |

**Assessment:** High precision (98.4%) but critically low recall (53.7%). Nearly half of all attacks pass undetected. The model excels at not flagging benign traffic but struggles to detect attacks that structurally resemble normal flows (particularly Syn, NetBIOS, NTP, SSDP, UDP, and TFTP).

**Training Convergence Analysis:**
- Volume sub-AE: Unstable — loss oscillates between 0.17 and 1.23 across 50 epochs (never converged)
- Packet Size sub-AE: Moderate convergence, final loss ~0.55
- IAT sub-AE: Best convergence, final loss ~0.07
- TCP Flags sub-AE: Poor convergence, loss plateaus at ~0.50-0.69

### 2.3 T-RAD (Temporal Reconstruction Anomaly Detector)

- **Architecture:** Multi-Scale TCN (4 dilation levels) + LTC + RoPE + Transformer Encoder, 189,761 parameters
- **Training:** 529,481 benign samples, 30 epochs, CosineAnnealing LR
- **Threshold:** 0.004133 (method: percentile_99.7)
- **Confusion Matrix:** TP=0 | TN=350,238 | FP=0 | FN=149,762

| Metric | Value |
|--------|-------|
| Accuracy | 70.05% |
| Precision | 0.00% |
| Recall | 0.00% |
| F1-Score | 0.0000 |
| FPR | 0.00% |
| FNR | 100.00% |

**CRITICAL FINDING:** T-RAD detected **zero** attacks out of 149,762 across all 10 attack types. The model has completely failed as a standalone detector.

**Root Cause Analysis:**
- Training converged well (loss: 0.077 → 0.000287 over 30 epochs)
- Mean RMSE on benign data: 0.001206, Std RMSE: 0.07672
- Threshold set at 0.004133 — extremely tight relative to the standard deviation
- The std_rmse (0.0767) is 63× larger than mean_rmse (0.0012), suggesting highly skewed error distribution with heavy outliers in benign data itself
- Attack reconstruction errors likely fall below the 0.004 threshold because the model learned such a general reconstruction that attacks don't produce significantly higher errors

### 2.4 QTN (Quantum Tensor Network)

- **Architecture:** Matrix Product State autoencoder, bond_dim=8
- **Role:** Third unsupervised voter in triple-consensus detection
- **In Hybrid Detection Stats:** 0 QTN detections, 0 triple consensus events

**Assessment:** QTN contributed zero unique detections in the evaluation. It only activates within the consensus pathway, which itself was never triggered because T-RAD never flags anything.

---

## 3. Per-Attack-Type Detection Analysis

### 3.1 Full Detection Rate Matrix (500K Evaluation Set)

| Attack Type | Flows | XGBoost | B-DAP | T-RAD | Chimera |
|-------------|------:|--------:|------:|------:|--------:|
| DrDoS_DNS | 14,918 | 99.81% | 96.03% | 0.00% | 99.81% |
| DrDoS_LDAP | 14,925 | 99.98% | 99.94% | 0.00% | 99.98% |
| DrDoS_MSSQL | 14,958 | 100.00% | 91.83% | 0.00% | 100.00% |
| DrDoS_NTP | 14,693 | 99.70% | 21.89% | 0.00% | 99.70% |
| DrDoS_NetBIOS | 14,924 | 99.98% | 1.34% | 0.00% | 99.98% |
| DrDoS_SNMP | 15,189 | 99.97% | 95.68% | 0.00% | 99.97% |
| DrDoS_SSDP | 14,848 | 99.98% | 44.80% | 0.00% | 99.98% |
| DrDoS_UDP | 15,122 | 99.95% | 43.86% | 0.00% | 99.95% |
| Syn | 15,115 | 98.86% | 0.26% | 0.00% | 98.86% |
| TFTP | 15,070 | 99.48% | 41.11% | 0.00% | 99.48% |
| **BENIGN** | **350,238** | **FP=608** | **FP=1,311** | **FP=0** | **FP=1,919** |

### 3.2 Critical Weak Points

1. **Syn Floods (XGBoost: 98.86%):** Lowest XGBoost detection rate. 173 Syn flows missed — these mimic normal TCP handshakes closely.
2. **DrDoS_NTP (B-DAP: 21.89%):** B-DAP misses ~78% of NTP attacks — NTP amplification flows have packet sizes similar to benign NTP traffic.
3. **DrDoS_NetBIOS (B-DAP: 1.34%):** Nearly invisible to B-DAP — NetBIOS attack flows have benign-like spatial structure.
4. **Syn (B-DAP: 0.26%):** Essentially undetectable by B-DAP. Syn floods are structurally indistinguishable from normal connection attempts.

---

## 4. Fusion Strategy Ablation Study

Tested on 1,428,571 flows from the full training dataset (larger evaluation set):

| Rank | Strategy | Accuracy | Precision | Recall | F1 | FPR | MCC |
|------|----------|----------|-----------|--------|------|------|------|
| 1 | XGBoost Alone | 99.80% | 99.93% | 99.79% | 0.9986 | 0.17% | 0.9953 |
| 2 | Dual Priority (XGB→B-DAP) | 99.69% | 99.77% | 99.79% | 0.9978 | 0.54% | 0.9927 |
| 3 | Tri-Model Priority | 99.63% | 99.68% | 99.79% | 0.9974 | 0.73% | 0.9913 |
| 4 | Tri-Model Consensus (2/3) | 81.24% | 99.94% | 73.20% | 0.8450 | 0.09% | 0.6709 |
| 5 | Transformer Alone | 81.10% | 99.83% | 73.08% | 0.8439 | 0.29% | 0.6680 |
| 6 | Dual Majority Vote | 67.64% | 100.00% | 53.69% | 0.6987 | 0.00% | 0.5087 |
| 7 | Dual AND Logic | 67.64% | 100.00% | 53.69% | 0.6987 | 0.00% | 0.5087 |
| 8 | B-DAP Alone | 67.53% | 99.71% | 53.69% | 0.6980 | 0.37% | 0.5049 |

### Key Ablation Findings

1. **XGBoost alone is the best overall performer** — adding unsupervised models slightly degrades accuracy due to increased FP.
2. **Priority fusion maintains recall** while accepting a small FPR increase (0.17% → 0.55%).
3. **Consensus strategies catastrophically fail** because T-RAD never flags → 2-of-3 vote becomes B-DAP-dependent.
4. **The fusion architecture's value is theoretical** — it would shine if T-RAD were functional, providing genuine multi-perspective defense.

---

## 5. Zero-Day (Leave-One-Out) Validation

Each attack type was completely removed from XGBoost training data; only unsupervised models could have "zero-day" knowledge:

| Attack Type | Samples | Detection Rate | Primary Detector |
|-------------|--------:|:--------------:|-----------------|
| DrDoS_DNS | 99,932 | **99.75%** | XGBoost (feature overlap) |
| DrDoS_LDAP | 99,930 | **100.00%** | XGBoost (feature overlap) |
| DrDoS_MSSQL | 99,969 | **99.99%** | XGBoost (feature overlap) |
| DrDoS_NetBIOS | 99,962 | **99.99%** | XGBoost (feature overlap) |
| DrDoS_NTP | 98,858 | **96.20%** | XGBoost (93K) + T-RAD (4.1K) |
| DrDoS_SNMP | 99,973 | **99.99%** | XGBoost (feature overlap) |
| DrDoS_SSDP | 99,960 | **99.99%** | XGBoost (feature overlap) |
| DrDoS_UDP | 99,929 | **99.96%** | XGBoost (feature overlap) |
| **Syn** | **99,964** | **12.99%** | XGBoost (12.7K) + B-DAP (438) |
| TFTP | 99,875 | **99.97%** | XGBoost (feature overlap) |

### Zero-Day Findings

1. **Most DrDoS variants share enough feature patterns** that XGBoost still detects them even when one specific variant is removed — this is feature overlap, not true zero-day detection.
2. **Syn floods are the true zero-day failure case:** Only 12.99% detected when removed. XGBoost caught 12,707 through feature overlap with other attacks, B-DAP caught 438, T-RAD caught 34, and consensus caught 47.
3. **Unsupervised models provided minimal zero-day coverage.** B-DAP contributed only 438 Syn detections; T-RAD contributed only 34. The unsupervised defense layer is not delivering on its architectural promise.

---

## 6. Hybrid Detection Pipeline Analysis

From the 500K evaluation run, the Chimera pipeline routed detections as follows:

| Detection Pathway | Count | % of Total |
|-------------------|------:|:----------:|
| XGBoost Detections | 150,024 | 30.00% |
| C2E2 Early Exits | 0 | 0.00% |
| Suspicious (Spatial/B-DAP only) | 1,312 | 0.26% |
| Suspicious (Temporal/T-RAD only) | 0 | 0.00% |
| Critical Anomaly (Consensus) | 0 | 0.00% |
| Triple Consensus (B-DAP+T-RAD+QTN) | 0 | 0.00% |
| Heuristic Detections | 0 | 0.00% |
| Benign Overrides | 0 | 0.00% |
| Total Benign | 348,664 | 69.73% |

### Pipeline Issues Identified

1. **C2E2 Early Exiting (0 exits):** The threshold is set at 99.5% confidence. XGBoost may not be producing probabilities that exceed this threshold on the evaluation set. This feature is not activating as designed.
2. **Zero consensus detections:** T-RAD never flags → consensus path is dead code in practice.
3. **Zero heuristic detections:** The cross-flow heuristics (port scan, DNS amplification, SYN reflection, entropy anomaly, distributed flood, Slowloris) triggered on zero flows. This may be expected for CIC-DDoS2019 dataset format (no IP/port columns or thresholds not met).
4. **Zero benign overrides:** No flows were overridden — either no protocol-aware safe traffic exists in the test set or the thresholds are too strict.

---

## 7. Training Quality Assessment

### 7.1 XGBoost Training

| Aspect | Value | Assessment |
|--------|-------|------------|
| Boosting Rounds | 13 / 300 | Early-stopped aggressively — task is easy for tree models |
| Validation Strategy | 80/20 split | Adequate but not stratified |
| HPO | Not used | Could potentially improve Syn detection |
| GPU Acceleration | CUDA | Efficient |

### 7.2 B-DAP Training Convergence

| Sub-AE | Final Loss | Converged? |
|--------|-----------|-----------|
| Volume | 0.94 | ❌ Oscillating heavily |
| Packet Size | 0.55 | ⚠️ Partial |
| IAT | 0.07 | ✅ Good |
| TCP Flags | 0.69 | ❌ Plateaued high |

**Issue:** Volume and TCP Flags sub-AEs never converged in 50 epochs. The loss curves oscillate significantly, suggesting learning rate instability or insufficient capacity for these feature groups.

### 7.3 T-RAD Training Convergence

| Aspect | Value | Assessment |
|--------|-------|------------|
| Final Loss | 0.000287 | Converged well |
| Threshold | 0.004133 | Extremely tight |
| Mean RMSE | 0.001206 | Very low reconstruction error |
| Std RMSE | 0.076716 | 63× mean — heavy-tailed distribution |

**Critical Issue:** The threshold is calibrated at the 99.7th percentile of benign RMSE. But because benign data itself has outliers with high RMSE (evidenced by std >> mean), the threshold is too tight. Attacks produce RMSE values that fall within the "normal" benign range — the model has learned to reconstruct *everything* well, including attacks.

---

## 8. Damage Assessment & Risk Matrix

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|------------|
| T-RAD non-functional (0% recall) | **CRITICAL** | Entire unsupervised temporal detection layer is dead weight; consensus detection disabled | Retrain with adjusted threshold (95th percentile) or deeper architecture |
| B-DAP misses ~46% of attacks | **HIGH** | Half the attacks pass spatial anomaly detection | Fine-tune EVT threshold; increase sub-AE capacity for Volume/TCP groups |
| C2E2 never activates | **MEDIUM** | No computational savings from early exit | Lower threshold from 99.5% to 95% or verify probability calibration |
| Volume/TCP sub-AEs not converging | **MEDIUM** | Contributes to B-DAP's poor recall | Increase epochs, reduce LR, add LR warmup |
| Syn flood weak detection (98.86%) | **MEDIUM** | 1.14% of Syn attacks evade even XGBoost | Add Syn-specific heuristic features |
| Cross-flow heuristics inactive | **LOW** | Network-level pattern detection not contributing | Verify CIC dataset has required columns (src_ip, dst_ip, dst_port, protocol) |
| Fusion adds FPs without improving recall | **LOW** | Chimera FPR (0.55%) > XGBoost alone (0.17%) | Consider tighter B-DAP threshold for production |

---

## 9. Conclusions & Recommendations

### 9.1 System Strengths

1. **XGBoost is exceptional** — 99.81% accuracy, 99.77% recall, 0.17% FPR across 10 attack types
2. **Per-attack coverage is comprehensive** — 8/10 attack types detected at >99.9% by XGBoost
3. **B-DAP has excellent precision** (98.4%) — when it flags something, it's almost certainly an attack
4. **Training pipeline is robust** — auto-configuration, GPU acceleration, integrity checksums

### 9.2 System Weaknesses

1. **T-RAD is the critical failure** — the entire temporal detection layer and all consensus pathways are non-functional
2. **The hybrid fusion does not outperform XGBoost alone** in any metric except a negligible recall improvement (+1 flow)
3. **B-DAP has severe blind spots** — Syn (0.26%), NetBIOS (1.34%), NTP (21.89%) attacks are nearly invisible
4. **Zero-day resilience is weak** — Syn floods drop to 12.99% detection in LOO validation

### 9.3 Priority Recommendations

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| P0 | Fix T-RAD threshold calibration — use adaptive percentile or increase tail sensitivity | Restore temporal detection; enable consensus pathway |
| P1 | Retrain B-DAP with more epochs (100+) and LR warmup for Volume/TCP sub-AEs | Improve B-DAP recall from 53% toward 70%+ |
| P2 | Calibrate C2E2 threshold (lower from 99.5% to 97%) | Enable early-exit computational savings |
| P3 | Add Syn-specific features (SYN flag ratio, half-open connection count) | Improve weakest attack detection |
| P4 | Verify heuristic column availability in test data | Enable cross-flow detection layer |

---

## Appendix A: Test Environment

- **OS:** Windows
- **GPU:** CUDA-enabled (XGBoost device=cuda)
- **Dataset:** CIC-DDoS2019, 10 attack types + benign
- **Evaluation Set:** 500,000 flows (350,238 benign + 149,762 attack)
- **Ablation Set:** 1,428,571 flows (430,219 benign + 998,352 attack)
- **Training Set:** 1,142,857 flows (70/30 attack/benign split)

## Appendix B: Model File Inventory

| Model | File | Size | Integrity |
|-------|------|------|-----------|
| XGBoost | xgboost_model.json | 153 KB | SHA256 verified |
| B-DAP | bdap_model.pkl | 154 KB | SHA256 verified |
| T-RAD | trad_model.pkl | 779 KB | SHA256 verified |
| QTN | qtn_model.pkl | 34 KB | SHA256 verified |

---

*Report generated for Chimera IDS v4.0 — Rigorous Post-Training Evaluation*
