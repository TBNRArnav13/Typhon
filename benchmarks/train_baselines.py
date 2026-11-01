import os
import sys
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_config(config_path="../config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_csv_simple(filepath, max_rows=None):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    print(f"Loading {filepath} (max_rows={max_rows})...")
    df = pd.read_csv(filepath, nrows=max_rows, on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    features = [f.strip().lower() for f in config["selected_features"]]
    
    cols_to_keep = [c for c in features if c in df.columns]
    if "label" in df.columns:
        cols_to_keep.append("label")
    df = df[cols_to_keep]
    df.replace([float("inf"), float("-inf")], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def main():
    config = load_config()
    features = [f.strip().lower() for f in config["selected_features"]]
    
    # 1. Fit robust scaler on benign baseline
    print("Fitting RobustScaler on benign baseline...")
    df_benign = load_csv_simple("../Dataset/Traning/1_Million_Benign.csv", max_rows=100000)
    scaler = RobustScaler()
    scaler.fit(df_benign[features].values)
    
    # 2. Load training dataset
    print("Loading training dataset...")
    df_train = load_csv_simple("../Dataset/Traning/Final_70_30_Dataset.csv", max_rows=None)
    # Shuffle dataset to ensure we get a balanced mix of classes
    df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X_train_raw = df_train[features].values
    X_train = np.clip(scaler.transform(X_train_raw), -5.0, 5.0)
    y_train = df_train["label"].astype(str).str.strip().str.upper().apply(
        lambda x: 0 if x in ["BENIGN", "0", "0.0"] else 1
    ).values
    
    # 3. Load testing dataset (Tier3_Mixed_Ratio_3.csv)
    print("Loading testing dataset...")
    df_test = load_csv_simple("../Dataset/Testing CIC/Tier3_Mixed_Ratio_3.csv", max_rows=500000)
    
    X_test_raw = df_test[features].values
    X_test = np.clip(scaler.transform(X_test_raw), -5.0, 5.0)
    y_test = df_test["label"].astype(str).str.strip().str.upper().apply(
        lambda x: 0 if x in ["BENIGN", "0", "0.0"] else 1
    ).values
    
    results = {}
    
    # Define models
    # To prevent SVM RBF from freezing or taking hours, we train SVM on a smaller representative subset of training data (20,000 samples)
    # MLP and others will use 100,000 samples for training.
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "DNN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=30, random_state=42)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == "SVM (RBF)":
            # Subset training to 20,000 samples for SVM RBF efficiency
            subset_size = min(len(X_train), 20000)
            X_tr = X_train[:subset_size]
            y_tr = y_train[:subset_size]
        else:
            # Use 100,000 samples for others
            subset_size = min(len(X_train), 100000)
            X_tr = X_train[:subset_size]
            y_tr = y_train[:subset_size]
            
        model.fit(X_tr, y_tr)
        
        print(f"Evaluating {name} on 500K test flows...")
        # Evaluate in chunks for KNN to avoid memory issues
        if name == "KNN (k=5)":
            preds = []
            chunk_size = 20000
            for i in range(0, len(X_test), chunk_size):
                chunk_preds = model.predict(X_test[i:i+chunk_size])
                preds.extend(chunk_preds)
            preds = np.array(preds)
        else:
            preds = model.predict(X_test)
            
        # Compute metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        print(f"Results for {name}:")
        print(f"  Accuracy:  {acc:.6f}")
        print(f"  Precision: {prec:.6f}")
        print(f"  Recall:    {rec:.6f}")
        print(f"  F1-Score:  {f1:.6f}")
        print(f"  MCC:       {mcc:.6f}")
        print(f"  CM:        TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "mcc": float(mcc),
            "confusion_matrix": {
                "TN": int(cm[0,0]),
                "FP": int(cm[0,1]),
                "FN": int(cm[1,0]),
                "TP": int(cm[1,1])
            }
        }
        
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll baseline models trained and evaluated successfully. Results saved to results/baseline_results.json.")

if __name__ == "__main__":
    main()
