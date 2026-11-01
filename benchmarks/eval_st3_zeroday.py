import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import torch

# Add parent directory to path so we can import models and src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.st3 import ST3_Autoencoder, EVTDetector

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
    print(f"Using device: {device}")
    
    # 1. Load RobustScaler, EVT threshold and ST3 model
    print("Loading saved pipeline objects...")
    with open("../models_saved/typhon_objects.pkl", "rb") as f:
        objs = pickle.load(f)
        scaler = objs["scaler"]
        evt = objs["evt"]
        
    print(f"Loaded EVT threshold: {evt.threshold:.6f}")
    
    st3 = ST3_Autoencoder(config=config["st3"], device=device.type).to(device)
    st3.load_state_dict(torch.load("../models_saved/st3.pt", map_location=device))
    st3.eval()
    
    # 2. Load the full training dataset to identify slices for each attack class
    print("Loading unified training dataset for evaluation...")
    df_full = pd.read_csv("../Dataset/Traning/Final_70_30_Dataset.csv", on_bad_lines="skip", low_memory=False)
    df_full.columns = df_full.columns.str.strip().str.lower()
    
    # Unique labels (excluding benign)
    labels = [l for l in df_full["label"].unique() if l != "BENIGN"]
    
    print("\nEvaluating unsupervised ST3 detection rates on each attack class...")
    print(f"{'Attack Class':<20} | {'Total Flows':<12} | {'Detected':<12} | {'Rate':<10}")
    print("="*60)
    
    results = {}
    
    for label in labels:
        df_label = df_full[df_full["label"] == label].copy()
        # Keep only the target features
        df_label = df_label[features]
        df_label.replace([float("inf"), float("-inf")], np.nan, inplace=True)
        df_label.dropna(inplace=True)
        
        N = len(df_label)
        if N == 0:
            continue
            
        X_scaled = np.clip(scaler.transform(df_label.values), -5.0, 5.0)
        X_seq = make_sequences(X_scaled, seq_len=10)
        
        dataset_seq = torch.tensor(X_seq, dtype=torch.float32, device=device)
        
        # Compute anomaly scores in chunks
        st3_scores = []
        with torch.no_grad():
            for i in range(0, len(dataset_seq), 512):
                batch_x = dataset_seq[i:i+512]
                scores = st3.compute_anomaly_score(batch_x)
                st3_scores.extend(scores.cpu().numpy())
                
        st3_scores = np.array(st3_scores)
        
        # Count anomalies exceeding the threshold
        detected = (st3_scores > evt.threshold).sum()
        rate = (detected / N) * 100.0
        
        print(f"{label:<20} | {N:<12,} | {detected:<12,} | {rate:.2f}%")
        results[label] = {
            "total": N,
            "detected": int(detected),
            "rate": float(rate)
        }
        
    print("="*60)

if __name__ == "__main__":
    main()
