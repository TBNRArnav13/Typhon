import os
import sys
import yaml
import pickle
import time
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.st3 import ST3_Autoencoder
from models.ct_dgnn import CT_DGNN
from models.fusion import XGBoostDetector, RLFusionAgent

# Backwards compatibility for unpickling QST3 models
import models.st3
sys.modules['models.qst3'] = models.st3

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def make_sequences(data, seq_len=10):
    N, D = data.shape
    pad = np.repeat(data[0:1], seq_len - 1, axis=0)
    padded = np.concatenate([pad, data], axis=0)
    from numpy.lib.stride_tricks import as_strided
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    shape = (N, seq_len, D)
    return as_strided(padded, shape=shape, strides=strides).copy()

def main():
    config = load_config()
    features = [f.strip().lower() for f in config["selected_features"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")
    
    # Load scaling and objects
    with open("../models_saved/typhon_objects.pkl", "rb") as f:
        objs = pickle.load(f)
        scaler = objs["scaler"]
        evt = objs["evt"]
        
    # Load models
    ct_dgnn = CT_DGNN(device=device.type).to(device)
    if os.path.exists("../models_saved/ct_dgnn.pt"):
        ct_dgnn.load_state_dict(torch.load("../models_saved/ct_dgnn.pt", map_location=device))
    ct_dgnn.eval()
    
    st3 = ST3_Autoencoder(config=config["st3"], device=device.type).to(device)
    if os.path.exists("../models_saved/st3.pt"):
        st3.load_state_dict(torch.load("../models_saved/st3.pt", map_location=device))
    st3.eval()
    
    xgb = objs["xgb"]
    
    # Load 10,000 flows for latency testing
    df = pd.read_csv("../Dataset/Testing CIC/Tier3_Mixed_Ratio_3.csv", nrows=10000, on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df_feats = df[features]
    df_feats.replace([float("inf"), float("-inf")], np.nan, inplace=True)
    df_feats.dropna(inplace=True)
    
    N = len(df_feats)
    print(f"Loaded {N} flows for latency profiling.")
    
    # Measure Preprocessing
    t0 = time.time()
    X_scaled = np.clip(scaler.transform(df_feats.values), -5.0, 5.0)
    t_prep = (time.time() - t0) * 1000.0 / N
    
    # Measure XGBoost
    t0 = time.time()
    xgb_probs = xgb.predict_proba(X_scaled)
    t_xgb = (time.time() - t0) * 1000.0 / N
    
    # Measure CT-DGNN
    src_ips = [str(x) for x in df.get("src_ip", np.arange(N))][:N]
    dst_ips = [str(x) for x in df.get("dst_ip", np.arange(N) + 10)][:N]
    timestamps = df.get("timestamp", np.linspace(0, N * 0.1, N))[:N]
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.values
        
    X_scaled_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    
    # Warmup
    _ = ct_dgnn(src_ips[:100], dst_ips[:100], timestamps[:100], X_scaled_t[:100])
    
    t0 = time.time()
    with torch.no_grad():
        _ = ct_dgnn(src_ips, dst_ips, timestamps, X_scaled_t)
    t_graph = (time.time() - t0) * 1000.0 / N
    
    # Measure ST3
    X_seq = make_sequences(X_scaled, seq_len=10)
    dataset_seq = torch.tensor(X_seq, dtype=torch.float32, device=device)
    
    # Warmup
    _ = st3(dataset_seq[:100])
    
    t0 = time.time()
    with torch.no_grad():
        _ = st3(dataset_seq)
    t_st3 = (time.time() - t0) * 1000.0 / N
    
    print("\n" + "="*50)
    print(f"{'Pipeline Component':<25} | {'Latency (us/flow)':<20}")
    print("="*50)
    print(f"{'Preprocessing':<25} | {t_prep * 1000:.3f} us")
    print(f"{'XGBoost':<25} | {t_xgb * 1000:.3f} us")
    print(f"{'CT-DGNN (LTC-ODE Graph)':<25} | {t_graph * 1000:.3f} us")
    print(f"{'ST3 (KAN-Trans-MPS)':<25} | {t_st3 * 1000:.3f} us")
    print("="*50)
    
    total_lat_us = (t_prep + t_xgb + t_graph + t_st3) * 1000
    throughput = 1e6 / total_lat_us
    print(f"Total pipeline latency: {total_lat_us:.3f} us per flow")
    print(f"Estimated throughput:   {throughput:,.1f} flows/sec")
    print("="*50)

if __name__ == "__main__":
    main()
