# Benchmark & Evaluation Outputs

This directory stores empirical validation results, profiling traces, and runtime detection reports produced by **Typhon**.

---

## 1. Verified Experimental Results

- **`standardized_500k_results.json`**:
  Comprehensive multi-metric evaluation of Typhon vs. standalone supervised XGBoost on 500,000 standardized flows from CIC-DDoS2019 under strict feature-leakage controls.
  - Standalone XGBoost: 99.81% accuracy, 99.91% recall, 608 False Positives (0.17% FPR).
  - Typhon Hybrid Framework: 99.55% accuracy, 99.77% recall, 1,919 False Positives (0.55% FPR) — maintaining an intentional safety net for out-of-distribution traffic.

- **`latency_benchmark_results.json`**:
  Edge gateway latency profiling and C2E2 fast-path hardware throughput metrics:
  - Supervised Fast-Path (XGBoost): ~56.4 µs/flow.
  - Neuromorphic ST3 / CT-DGNN Full Inference: ~2.4 ms/flow.
  - Confidence-Calibrated Early Exit (C2E2, $p > 0.995$): Effective weighted latency of **288.89 µs/flow**, sustaining **3,461.5 flows/sec** on commodity GPU hardware.

- **`fresh_evaluation_results.json`**:
  Drift and zero-day perturbation test suite results demonstrating Typhon's **99.83% zero-day recall** under novel protocol shifts where static classifiers drop to 64.79%.

---

## 2. Real-Time Detection Logs & Reports

Whenever `--mode detect` is executed:
- A timestamped subdirectory `run_YYYYMMDD_HHMMSS/` is created.
- Detailed classification breakdowns (`benign/`, `suspicious/`, `malicious/`) are partitioned.
- Comprehensive JSON and Markdown summary reports (`Report.json`, `Report.md`) with MITRE ATT&CK mapping are automatically generated.
