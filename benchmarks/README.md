# Typhon Evaluation & Benchmarking Suite

This directory contains standalone benchmark scripts, latency profiling tools, and comparative evaluation scenarios.

---

## Benchmark Scripts

1. **`eval_comprehensive_500k.py`**:
   Evaluates Typhon and baseline models over the 500,000-flow CIC-DDoS2019 standardized test set.
   - Computes Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR), and Matthews Correlation Coefficient (MCC).
   - Usage:
     ```bash
     python benchmarks/eval_comprehensive_500k.py
     ```

2. **`eval_latency_benchmarks.py`**:
   Profiles inference latency, hardware resource utilization (CPU/GPU/RAM), and C2E2 early-exit throughput:
   - Measures per-flow latency across XGBoost fast-path, ST3 anomaly detector, CT-DGNN graph propagation, and RL-Fusion decision logic.
   - Usage:
     ```bash
     python benchmarks/eval_latency_benchmarks.py
     ```

3. **`eval_st3_zeroday.py`**:
   Evaluates the Spatio-Temporal Tensor-Transformer (ST3) anomaly detector specifically on unseen zero-day attacks excluded from training.
   - Usage:
     ```bash
     python benchmarks/eval_st3_zeroday.py
     ```

4. **`run_scenario.py`**:
   Simulates end-to-end traffic scenarios combining varying proportions of benign, known volumetric DoS, and stealthy zero-day flows.
   - Usage:
     ```bash
     python benchmarks/run_scenario.py
     ```

5. **`train_baselines.py`**:
   Trains classical baseline classifiers (Random Forest, SVM, standalone XGBoost) for empirical comparison against Typhon.
   - Results are serialized to `results/baseline_results.json`.
