import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import psutil
from xgboost import XGBClassifier

# Add current dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_config, load_csv
import pickle
from src.utils import get_device
from models.st3 import ST3_Autoencoder
from models.ct_dgnn import CT_DGNN
from models.fusion import RLFusionAgent

def benchmark_latency_and_throughput():
    print("=" * 80)
    print("TYPHON IDS: EMPIRICAL LATENCY & THROUGHPUT BENCHMARKING ENGINE")
    print("=" * 80)
    
    config = load_config()
    device = get_device()
    features = [f.strip().lower() for f in config["selected_features"]]
    
    test_path = "Dataset/Testing CIC/Tier1_PureAttack_Mix_1.csv"
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return
        
    df_test = load_csv(test_path, max_rows=50000)
    X_raw = df_test[features].values
    y_test = df_test["label"].apply(lambda x: 0 if str(x).upper() in ["BENIGN", "0", "0.0"] else 1).values
    
    models_dir = config["directories"]["models"]
    obj_path = os.path.join(models_dir, "typhon_objects.pkl")
    with open(obj_path, "rb") as f:
        objs = pickle.load(f)
        scaler = objs["scaler"]
        evt = objs["evt"]
        xgb_detector = objs["xgb"]
        
    X_scaled = np.clip(scaler.transform(X_raw), -5.0, 5.0)
    N = len(X_scaled)
    print(f"Loaded {N:,} benchmark flows with {X_scaled.shape[1]} features.", flush=True)
    print(f"Compute Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    
    st3_model = ST3_Autoencoder(config=config["st3"], device=device.type).to(device)
    st3_model.load_state_dict(torch.load(os.path.join(models_dir, "st3.pt"), map_location=device))
    st3_model.eval()
    
    dgnn_model = CT_DGNN(input_dim=33, device=device.type).to(device)
    dgnn_model.eval()
    
    rl_agent = RLFusionAgent(config=config, device=device.type).to(device)
    rl_agent.q_net.load_state_dict(torch.load(os.path.join(models_dir, "rl_qnet.pt"), map_location=device))
    rl_agent.q_net.eval()
    
    # Warmup runs
    dummy_input = torch.tensor(X_scaled[:10], dtype=torch.float32).to(device)
    dummy_seq = dummy_input.unsqueeze(1).repeat(1, 10, 1)
    dummy_src = [f"192.168.1.{i%250}" for i in range(10)]
    dummy_dst = [f"10.0.0.{i%250}" for i in range(10)]
    dummy_ts = np.arange(10, dtype=np.float64)
    with torch.no_grad():
        _ = st3_model(dummy_seq)
        _ = dgnn_model(dummy_src, dummy_dst, dummy_ts, dummy_input)
        _ = rl_agent.q_net(torch.randn(10, 6).to(device))
        
    # --- 1. Fast-Path (C2E2) Latency Measurement ---
    print("\n[1] Benchmarking Fast-Path (C2E2 - XGBoost Early Exit)...", flush=True)
    fast_latencies = []
    
    # Warmup XGB
    _ = xgb_detector.predict_proba(X_scaled[:100])
    
    t0 = time.perf_counter()
    for i in range(0, min(10000, N), 100):
        chunk = X_scaled[i:i+100]
        tc0 = time.perf_counter()
        _ = xgb_detector.predict_proba(chunk)
        tc1 = time.perf_counter()
        fast_latencies.append((tc1 - tc0) / 100.0 * 1e6) # microseconds per flow
    t1 = time.perf_counter()
    
    avg_fast_us = np.mean(fast_latencies)
    p95_fast_us = np.percentile(fast_latencies, 95)
    p99_fast_us = np.percentile(fast_latencies, 99)
    print(f"Fast-Path Latency (C2E2): Mean = {avg_fast_us:.2f} µs/flow | P95 = {p95_fast_us:.2f} µs/flow | P99 = {p99_fast_us:.2f} µs/flow", flush=True)
    
    # --- 2. Deep-Path Component Latency Breakdown ---
    print("\n[2] Benchmarking Deep-Path Component Breakdown...", flush=True)
    
    # ST3 Autoencoder Latency
    st3_latencies = []
    with torch.no_grad():
        for i in range(0, 1000, 10):
            seq = torch.tensor(X_scaled[i:i+10], dtype=torch.float32).unsqueeze(1).repeat(1, 10, 1).to(device)
            tc0 = time.perf_counter()
            _ = st3_model(seq)
            if device.type == "cuda": torch.cuda.synchronize()
            tc1 = time.perf_counter()
            st3_latencies.append((tc1 - tc0) / 10.0 * 1e6)
    avg_st3_us = np.mean(st3_latencies)
    
    # CT-DGNN Latency
    dgnn_latencies = []
    dummy_src = [f"192.168.1.{i%250}" for i in range(10)]
    dummy_dst = [f"10.0.0.{i%250}" for i in range(10)]
    dummy_ts = np.arange(10, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, 1000, 10):
            inp = torch.tensor(X_scaled[i:i+10], dtype=torch.float32).to(device)
            tc0 = time.perf_counter()
            _ = dgnn_model(dummy_src, dummy_dst, dummy_ts, inp)
            if device.type == "cuda": torch.cuda.synchronize()
            tc1 = time.perf_counter()
            dgnn_latencies.append((tc1 - tc0) / 10.0 * 1e6)
    avg_dgnn_us = np.mean(dgnn_latencies)
    
    # DQN Routing Latency
    rl_latencies = []
    dummy_states = torch.randn(10, 6).to(device)
    with torch.no_grad():
        for _ in range(100):
            tc0 = time.perf_counter()
            _ = rl_agent.q_net(dummy_states)
            if device.type == "cuda": torch.cuda.synchronize()
            tc1 = time.perf_counter()
            rl_latencies.append((tc1 - tc0) / 10.0 * 1e6)
    avg_rl_us = np.mean(rl_latencies)
    
    total_deep_us = avg_fast_us + avg_st3_us + avg_dgnn_us + avg_rl_us
    print(f"  - XGBoost Gating Latency: {avg_fast_us:.2f} µs/flow")
    print(f"  - ST3 Tensor-Autoencoder Latency: {avg_st3_us:.2f} µs/flow")
    print(f"  - CT-DGNN Graph Temporal Latency: {avg_dgnn_us:.2f} µs/flow")
    print(f"  - DQN Neuro-Symbolic Routing Latency: {avg_rl_us:.2f} µs/flow")
    print(f"Total Deep-Path Latency: {total_deep_us:.2f} µs/flow ({total_deep_us/1000:.3f} ms/flow)", flush=True)
    
    # --- 3. Weighted Hybrid System Throughput ---
    # 98.22% fast-path exits, 1.78% deep-path exits
    weighted_latency_us = (0.9822 * avg_fast_us) + (0.0178 * total_deep_us)
    throughput_fps = 1e6 / weighted_latency_us
    
    print("\n[3] System Throughput & Memory Benchmark...", flush=True)
    print(f"Weighted Hybrid Latency: {weighted_latency_us:.2f} µs/flow")
    print(f"System Operational Throughput: {throughput_fps:,.0f} flows/second", flush=True)
    
    # Memory Footprint
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(f"Host RAM Usage: {ram_mb:.2f} MB")
    print(f"GPU VRAM Allocation: {vram_mb:.2f} MB", flush=True)
    
    # Save latency payload
    import json
    latency_payload = {
        "hardware": str(device),
        "fast_path_latency_us": {"mean": avg_fast_us, "p95": p95_fast_us, "p99": p99_fast_us},
        "deep_path_breakdown_us": {
            "xgb_gating": avg_fast_us,
            "st3_kan_mps": avg_st3_us,
            "ct_dgnn_ltc": avg_dgnn_us,
            "dqn_routing": avg_rl_us,
            "total_deep_path": total_deep_us
        },
        "weighted_latency_us": weighted_latency_us,
        "throughput_flows_per_sec": throughput_fps,
        "memory_mb": {"ram": ram_mb, "vram": vram_mb}
    }
    
    out_json = "outputs/latency_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(latency_payload, f, indent=2)
    print(f"\nLatency benchmark results saved to {out_json}", flush=True)
    print("=" * 80)

if __name__ == "__main__":
    benchmark_latency_and_throughput()
