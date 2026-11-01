import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_csv, load_config

def build_standardized_500k_dataset():
    print("=" * 80)
    print("BUILDING STANDARDIZED 500,000-FLOW EVALUATION DATASET (70% BENIGN / 30% ATTACK)")
    print("=" * 80)
    
    benign_path = "Dataset/Traning/1_Million_Benign.csv"
    attack_path = "Dataset/Traning/Final_70_30_Dataset.csv"
    output_path = "Dataset/Testing CIC/Standardized_500K_Test_Set.csv"
    
    print(f"[1] Loading Benign Flows from {benign_path}...", flush=True)
    df_benign = load_csv(benign_path, max_rows=500000)
    # Ensure benign labels are explicitly 'BENIGN'
    df_benign = df_benign.sample(n=350000, random_state=42).reset_index(drop=True)
    df_benign["label"] = "BENIGN"
    print(f"Loaded {len(df_benign):,} Benign flows.", flush=True)
    
    print(f"\n[2] Loading Attack Flows from {attack_path}...", flush=True)
    df_attack_full = load_csv(attack_path, max_rows=1500000)
    
    # Filter out benign from attack file if any
    df_attack_only = df_attack_full[~df_attack_full["label"].astype(str).str.upper().isin(["BENIGN", "0", "0.0"])].copy()
    print(f"Total available attack flows: {len(df_attack_only):,}", flush=True)
    
    # Target 10 attack classes: 15,000 flows each = 150,000 attack flows
    attack_classes = df_attack_only["label"].unique()
    print(f"Discovered attack classes ({len(attack_classes)}): {attack_classes}", flush=True)
    
    selected_attack_dfs = []
    target_per_class = 15000
    
    for attack_cls in attack_classes:
        sub_df = df_attack_only[df_attack_only["label"] == attack_cls]
        n_available = len(sub_df)
        n_sample = min(target_per_class, n_available)
        sampled_sub = sub_df.sample(n=n_sample, random_state=42)
        selected_attack_dfs.append(sampled_sub)
        print(f"  - Class '{attack_cls}': sampled {n_sample:,} / {n_available:,} flows")
        
    df_attacks_combined = pd.concat(selected_attack_dfs, ignore_index=True)
    
    # If total attack flows sampled < 150,000 due to class size differences, top up to exactly 150,000
    if len(df_attacks_combined) < 150000:
        needed = 150000 - len(df_attacks_combined)
        remaining_pool = df_attack_only.drop(df_attacks_combined.index, errors="ignore")
        top_up = remaining_pool.sample(n=min(needed, len(remaining_pool)), random_state=42)
        df_attacks_combined = pd.concat([df_attacks_combined, top_up], ignore_index=True)
    elif len(df_attacks_combined) > 150000:
        df_attacks_combined = df_attacks_combined.sample(n=150000, random_state=42).reset_index(drop=True)
        
    print(f"\nFinal Combined Attack Corpus: {len(df_attacks_combined):,} flows across 10 attack types.", flush=True)
    
    # Concatenate 350,000 Benign + 150,000 Attacks = 500,000 total flows
    df_final_500k = pd.concat([df_benign, df_attacks_combined], ignore_index=True)
    # Shuffle dataset deterministically
    df_final_500k = df_final_500k.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print("\n[3] Standardized 500,000-Flow Dataset Summary:", flush=True)
    print(f"  - Total Rows: {len(df_final_500k):,}")
    print(f"  - Benign Flows (70.0%): {(df_final_500k['label'] == 'BENIGN').sum():,} flows")
    print(f"  - Malicious Flows (30.0%): {(df_final_500k['label'] != 'BENIGN').sum():,} flows")
    
    print("\nDetailed Per-Attack Breakdown:")
    print(df_final_500k["label"].value_counts())
    
    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final_500k.to_csv(output_path, index=False)
    print(f"\nSuccessfully written standardized evaluation dataset to: {output_path}", flush=True)
    print("=" * 80)

if __name__ == "__main__":
    build_standardized_500k_dataset()
