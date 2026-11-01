import os
import sys
import argparse
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, Tuple, List, Optional

from src.utils import logger, get_device
from src.data_loader import load_csv, combine_datasets
from src.drift_detector import ConceptDriftDetector
from src.input_handler import InputHandler
from src.reporting import ReportGenerator

# Import models
from models.ct_dgnn import CT_DGNN
from models.st3 import ST3_Autoencoder, EVTDetector
from models.fusion import XGBoostDetector, RLFusionAgent

# Backwards compatibility for unpickling QST3 models
import models.st3
sys.modules['models.qst3'] = models.st3



def parse_args():
    parser = argparse.ArgumentParser(description="Typhon Intrusion Detection System Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "detect", "evaluate", "ablation"],
                        help="Execution mode: train, detect, evaluate, or ablation")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to PCAP or CSV file for detection/evaluation")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    return parser.parse_args()


def make_sequences(data: np.ndarray, seq_len: int = 10) -> np.ndarray:
    """Transforms flat tabular feature rows into rolling temporal sequences using vectorized padding."""
    N, D = data.shape
    # Pad the beginning with repeated first row so every index produces a full window
    pad = np.repeat(data[0:1], seq_len - 1, axis=0)
    padded = np.concatenate([pad, data], axis=0)  # (N + seq_len - 1, D)
    # Use stride_tricks for zero-copy sliding windows
    from numpy.lib.stride_tricks import as_strided
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    shape = (N, seq_len, D)
    sequences = as_strided(padded, shape=shape, strides=strides)
    return sequences.copy()  # copy to own memory so padded can be freed


def extract_rl_states(xgb_probs: np.ndarray, st3_scores: np.ndarray, 
                      scaled_features: np.ndarray, df: pd.DataFrame, 
                      js_drift: float) -> np.ndarray:
    """
    Constructs the 6-dimensional state vector S for the DQN RL-Fusion agent.
    S = [p_xgb, e_st3, H_entropy, V_flow, L_system, delta_drift]
    """
    N = len(xgb_probs)
    states = []
    
    # Pre-extract flow packets/s column to optimize speed
    col_name = "flow packets/s"
    if col_name in df.columns:
        flow_packets = df[col_name].values
    else:
        # Fallback to feature index if raw column missing (flow packets/s is index 14)
        flow_packets = scaled_features[:, 14] if scaled_features.shape[1] > 14 else np.zeros(N)
        
    for i in range(N):
        p_xgb = float(xgb_probs[i])
        e_st3 = float(st3_scores[i])
        
        # 1. Feature Entropy
        feats = np.abs(scaled_features[i])
        p = (feats + 1e-9) / (feats.sum() + 1e-9)
        h_entropy = float(-np.sum(p * np.log2(p)))
        
        # Autoencoder score
        e_st3 = float(st3_scores[i])
        
        # 2. Flow Velocity
        v_flow = float(np.log1p(np.abs(flow_packets[i])))
        
        # 3. System traffic load proxy (dynamic traffic load/rate pressure signal)
        l_system = float(np.clip(np.abs(flow_packets[i]) / 10000.0, 0.0, 1.0))
        
        # 4. Drift metric
        delta_drift = float(js_drift)
        
        state = [p_xgb, e_st3, h_entropy, v_flow, l_system, delta_drift]
        states.append(state)
        
    return np.array(states, dtype=np.float32)


def apply_symbolic_overrides(df: pd.DataFrame, predictions: List[str], reasons: List[str]) -> Tuple[List[str], int]:
    """Applies protocol/symbolic safety overrides for benign local broadcast and mDNS discovery."""
    overrides_triggered = 0
    overridden_preds = []
    
    # Check if necessary network attributes are present
    has_ports = "src_port" in df.columns and "dst_port" in df.columns
    has_ip = "dst_ip" in df.columns
    
    for i in range(len(predictions)):
        pred = predictions[i]
        reason = reasons[i]
        
        if pred != "Benign":
            # Rule 1: Clear local multicast DNS (port 5353)
            is_mdns = False
            if has_ports:
                sport = int(df.iloc[i]["src_port"])
                dport = int(df.iloc[i]["dst_port"])
                if sport == 5353 or dport == 5353:
                    is_mdns = True
                    
            # Rule 2: Clear local broadcast/multicast target subnets
            is_local = False
            if has_ip:
                dip = str(df.iloc[i]["dst_ip"])
                if dip.startswith("224.") or dip.startswith("239.") or dip == "255.255.255.255":
                    is_local = True
                    
            if is_mdns or is_local:
                overridden_preds.append("Benign")
                reasons[i] = "Symbolic Override -- Benign Local Discovery / Multicast"
                overrides_triggered += 1
            else:
                overridden_preds.append(pred)
        else:
            overridden_preds.append(pred)
            
    return overridden_preds, overrides_triggered


class TyphonPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        from src.data_loader import load_config
        self.config = load_config(config_path)
        self.device = get_device()
        
        # Save directories
        self.models_dir = self.config["directories"]["models"]
        self.outputs_dir = self.config["directories"]["outputs"]
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Pipeline modules placeholder
        self.scaler: Optional[RobustScaler] = None
        self.ct_dgnn: Optional[CT_DGNN] = None
        self.st3: Optional[ST3_Autoencoder] = None
        self.evt: Optional[EVTDetector] = None
        self.xgb: Optional[XGBoostDetector] = None
        self.rl_agent: Optional[RLFusionAgent] = None
        self.drift_detector: Optional[ConceptDriftDetector] = None

    def save_pipeline(self):
        """Saves fitted models and scaler configuration to disk."""
        logger.info("Persisting Typhon pipeline states...")
        
        # PyTorch model weights
        if self.ct_dgnn:
            torch.save(self.ct_dgnn.state_dict(), os.path.join(self.models_dir, "ct_dgnn.pt"))
        if self.st3:
            torch.save(self.st3.state_dict(), os.path.join(self.models_dir, "st3.pt"))
        if self.rl_agent:
            torch.save(self.rl_agent.q_net.state_dict(), os.path.join(self.models_dir, "rl_qnet.pt"))
            
        # Non-PyTorch serializable objects
        objs = {
            "scaler": self.scaler,
            "evt": self.evt,
            "xgb": self.xgb,
            "drift_detector": self.drift_detector
        }
        with open(os.path.join(self.models_dir, "typhon_objects.pkl"), "wb") as f:
            pickle.dump(objs, f)
            
        logger.info("Pipeline states successfully written to disk.")

    def load_pipeline(self) -> bool:
        """Loads fitted pipeline states from disk."""
        logger.info("Loading Typhon pipeline states...")
        obj_path = os.path.join(self.models_dir, "typhon_objects.pkl")
        if not os.path.exists(obj_path):
            logger.error("Fitted pipeline objects not found. Run pipeline training first.")
            return False
            
        with open(obj_path, "rb") as f:
            objs = pickle.load(f)
            self.scaler = objs["scaler"]
            self.evt = objs["evt"]
            self.xgb = objs["xgb"]
            self.drift_detector = objs["drift_detector"]
            
        # Re-initialize modules on appropriate device
        self.ct_dgnn = CT_DGNN(device=self.device.type).to(self.device)
        ct_path = os.path.join(self.models_dir, "ct_dgnn.pt")
        if os.path.exists(ct_path):
            self.ct_dgnn.load_state_dict(torch.load(ct_path, map_location=self.device))
        self.ct_dgnn.eval()
        
        self.st3 = ST3_Autoencoder(config=self.config["st3"], device=self.device.type).to(self.device)
        st3_path = os.path.join(self.models_dir, "st3.pt")
        if os.path.exists(st3_path):
            self.st3.load_state_dict(torch.load(st3_path, map_location=self.device))
        self.st3.eval()
        
        self.rl_agent = RLFusionAgent(config=self.config, device=self.device.type).to(self.device)
        rl_path = os.path.join(self.models_dir, "rl_qnet.pt")
        if os.path.exists(rl_path):
            self.rl_agent.q_net.load_state_dict(torch.load(rl_path, map_location=self.device))
            self.rl_agent.update_target_network()
            
        logger.info("Pipeline loaded and ready.")
        return True

    def train_pipeline(self):
        """Fits all modular components of Typhon from scratch."""
        logger.info("========================================")
        logger.info("INITIATING PIPELINE TRAINING (CLEAN SLATE)")
        logger.info("========================================")
        
        # 1. Load Datasets
        benign_path = self.config["datasets"]["benign"]
        dos_dir = self.config["datasets"]["dos"]
        
        # Load Benign for unsupervised training
        df_benign = load_csv(benign_path)
        if df_benign is None:
            logger.error("Failed to load benign dataset. Aborting.")
            return
            
        # Fit Scaler
        logger.info("Fitting RobustScaler on benign baseline...")
        features = [f.strip().lower() for f in self.config["selected_features"]]
        self.scaler = RobustScaler()
        
        X_benign_raw = df_benign[features].values
        self.scaler.fit(X_benign_raw)
        X_benign = np.clip(self.scaler.transform(X_benign_raw), -5.0, 5.0)
        
        # Fit concept drift reference PMF
        self.drift_detector = ConceptDriftDetector(self.config)
        self.drift_detector.fit_baseline(df_benign[features])
        
        # 2. Train Spatio-Temporal Autoencoder (ST3) strictly on Benign (Unsupervised)
        logger.info("--- Training Unsupervised ST3 Branch ---")
        self.st3 = ST3_Autoencoder(config=self.config["st3"], device=self.device.type).to(self.device)
        optimizer_st3 = torch.optim.Adam(self.st3.parameters(), lr=self.config["st3"]["learning_rate"])
        
        # Prepare sequential windows: (N, seq_len, 33)
        seq_len = 10
        X_benign_seq = make_sequences(X_benign, seq_len=seq_len)
        
        # Use representative subset to fit in GPU/RAM (50K sequences * 10 * 33 * 4B = ~66MB)
        st3_train_slice = min(len(X_benign_seq), 50000)
        dataset_st3 = torch.tensor(X_benign_seq[:st3_train_slice], dtype=torch.float32)
        batch_size = self.config["st3"]["batch_size"]
        epochs_st3 = self.config["st3"]["epochs"]
        
        self.st3.train()
        for epoch in range(1, epochs_st3 + 1):
            permutation = torch.randperm(dataset_st3.size(0))
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, dataset_st3.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = dataset_st3[indices].to(self.device)
                
                optimizer_st3.zero_grad()
                recon, target = self.st3(batch_x)
                loss = torch.mean((recon - target) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.st3.parameters(), max_norm=1.0)
                optimizer_st3.step()
                
                epoch_loss += loss.item() * len(indices)
                n_batches += 1
            avg_loss = epoch_loss / len(dataset_st3)
            logger.info(f"ST3 Epoch {epoch}/{epochs_st3} -- Mean Reconstruction MSE: {avg_loss:.6f}")
            
        # Fit EVT Extreme Value Threshold Detector on ST3 Benign Reconstruction Errors
        logger.info("--- Fitting EVT Threshold Detector ---")
        self.st3.eval()
        benign_errors = []
        with torch.no_grad():
            for i in range(0, len(dataset_st3), batch_size):
                batch_x = dataset_st3[i:i+batch_size].to(self.device)
                scores = self.st3.compute_anomaly_score(batch_x)
                benign_errors.extend(scores.cpu().numpy())
                
        benign_errors = np.array(benign_errors)
        self.evt = EVTDetector(
            tail_percentile=self.config["st3"]["evt"]["tail_percentile"],
            sensitivity_sigma=self.config["st3"]["evt"]["sensitivity_sigma"]
        )
        self.evt.fit(benign_errors)
        
        # 3. Train Continuous-Time Graph Neural Network (CT-DGNN)
        logger.info("--- Training Continuous-Time Graph Engine (CT-DGNN) ---")
        self.ct_dgnn = CT_DGNN(device=self.device.type).to(self.device)
        optimizer_graph = torch.optim.Adam(self.ct_dgnn.parameters(), lr=self.config["ct_dgnn"]["learning_rate"])
        
        # Use real IPs if available, otherwise generate synthetic node IDs
        graph_slice = min(len(df_benign), 5000)
        if "src_ip" in df_benign.columns:
            graph_src = [str(x) for x in df_benign["src_ip"].iloc[:graph_slice]]
            graph_dst = [str(x) for x in df_benign["dst_ip"].iloc[:graph_slice]]
        else:
            graph_src = [str(x) for x in np.arange(graph_slice)]
            graph_dst = [str(x + 10) for x in np.arange(graph_slice)]
        graph_times = np.linspace(0, graph_slice * 0.1, graph_slice)
        
        graph_epochs = self.config["ct_dgnn"]["epochs"]
        X_benign_t = torch.tensor(X_benign[:graph_slice], dtype=torch.float32, device=self.device)
        chunk_size = 500
        
        self.ct_dgnn.train()
        for epoch in range(1, graph_epochs + 1):
            self.ct_dgnn.reset_states()
            total_loss = 0.0
            for i in range(0, graph_slice, chunk_size):
                end_idx = min(i + chunk_size, graph_slice)
                src = graph_src[i:end_idx]
                dst = graph_dst[i:end_idx]
                times = graph_times[i:end_idx]
                edges = X_benign_t[i:end_idx]
                
                optimizer_graph.zero_grad()
                features_g = self.ct_dgnn(src, dst, times, edges)
                loss = torch.mean(features_g ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ct_dgnn.parameters(), max_norm=1.0)
                optimizer_graph.step()
                total_loss += loss.item()
            logger.info(f"CT-DGNN Epoch {epoch}/{graph_epochs} -- Topo L2 Conservation: {total_loss:.6f}")
            
        # 4. Train Supervised XGBoost Classifier on Combined Dataset
        logger.info("--- Training Supervised XGBoost Branch ---")
        combined_path = self.config["datasets"]["training_combined"]
        df_combined = load_csv(combined_path)
        if df_combined is None:
            # Fallback to combine_datasets if pre-combined file not available
            df_combined = combine_datasets(benign_path, dos_dir)
            
        if df_combined is None:
            logger.error("Could not load combined training set. XGBoost cannot fit.")
            return
            
        X_comb_raw = df_combined[features].values
        X_comb = np.clip(self.scaler.transform(X_comb_raw), -5.0, 5.0)
        
        y_comb = df_combined["label"].astype(str).str.strip().str.upper().apply(
            lambda x: 0 if x in ["BENIGN", "0", "0.0"] else 1
        ).values
        
        self.xgb = XGBoostDetector(self.config)
        self.xgb.fit(X_comb, y_comb)
        
        # 5. Train RL-Fusion DQN Agent
        logger.info("--- Training RL DQN-Fusion Agent ---")
        self.rl_agent = RLFusionAgent(self.config, device=self.device.type).to(self.device)
        
        # Collect model predictions on training combined slice to build experiences
        self.st3.eval()
        self.ct_dgnn.eval()
        
        # SHUFFLE before slicing so RL sees both benign and malicious
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(X_comb))
        X_comb_shuf = X_comb[shuffle_idx]
        y_comb_shuf = y_comb[shuffle_idx]
        df_shuf = df_combined.iloc[shuffle_idx]
        
        eval_slice = min(len(X_comb), 50000)
        X_comb_slice = X_comb_shuf[:eval_slice]
        y_comb_slice = y_comb_shuf[:eval_slice]
        df_slice = df_shuf.iloc[:eval_slice]
        
        # Extract features for ST3
        X_comb_seq = make_sequences(X_comb_slice, seq_len=10)
        dataset_comb_seq = torch.tensor(X_comb_seq, dtype=torch.float32, device=self.device)
        
        st3_scores = []
        with torch.no_grad():
            for i in range(0, len(dataset_comb_seq), 128):
                batch_x = dataset_comb_seq[i:i+128]
                scores = self.st3.compute_anomaly_score(batch_x)
                st3_scores.extend(scores.cpu().numpy())
                
        st3_scores = np.array(st3_scores)
        xgb_probs = self.xgb.predict_proba(X_comb_slice)
        
        # State vector representation
        js_drift = 0.0 # training baseline has 0 drift
        states = extract_rl_states(xgb_probs, st3_scores, X_comb_slice, df_slice, js_drift)
        
        # Train DQN Agent over experiences
        episodes = 15
        logger.info(f"Running DQN training loop over {episodes} episodes...")
        for ep in range(1, episodes + 1):
            total_reward = 0.0
            losses = []
            
            # Sequence simulation
            for i in range(len(states) - 1):
                state = states[i]
                next_state = states[i+1]
                
                # Select action
                action = self.rl_agent.select_action(state, evaluate=False)
                
                # Dynamic prediction
                pred = self.rl_agent.get_fusion_decision(
                    action, xgb_probs[i], st3_scores[i], self.evt.threshold
                )
                
                # Asymmetric reward with zero-day incentive
                reward = self.rl_agent.compute_asymmetric_reward(
                    pred, y_comb_slice[i], xgb_probs[i], st3_scores[i], self.evt.threshold
                )
                
                # Store transition
                done = (i == len(states) - 2)
                self.rl_agent.store_transition(state, action, reward, next_state, done)
                
                # Train step every 4 steps for optimal experience replay sampling speed
                if i % 4 == 0:
                    loss = self.rl_agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                    
                total_reward += reward
                
            self.rl_agent.decay_epsilon()
            self.rl_agent.update_target_network()
            mean_loss = np.mean(losses) if losses else 0.0
            logger.info(f"Episode {ep}/{episodes} -- Total Reward: {total_reward:.1f}, Epsilon: {self.rl_agent.epsilon:.3f}, Loss: {mean_loss:.5f}")
            
        # Save training state
        self.save_pipeline()
        logger.info("Pipeline training completely finalized.")

    def run_detection(self, filepath: str) -> Optional[str]:
        """Runs the detection engine over an input file and outputs reports."""
        if not self.load_pipeline():
            return None
            
        logger.info(f"Processing target file for intrusion detection: {filepath}")
        input_handler = InputHandler(self.config)
        df_raw = input_handler.process_file(filepath)
        
        if df_raw is None or df_raw.empty:
            logger.error("No valid flow records extracted from input file.")
            return None
            
        features = [f.strip().lower() for f in self.config["selected_features"]]
        X_scaled = np.clip(self.scaler.transform(df_raw[features].values), -5.0, 5.0)
        
        # 1. Concept Drift assessment
        drift_status = self.drift_detector.detect_drift(df_raw[features])
        js_drift = drift_status.get("js_divergence", 0.0)
        
        # 2. Sequential Spatio-Temporal branch (ST3)
        X_seq = make_sequences(X_scaled, seq_len=10)
        dataset_seq = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
        
        self.st3.eval()
        st3_scores = []
        with torch.no_grad():
            for i in range(0, len(dataset_seq), 128):
                batch_x = dataset_seq[i:i+128]
                scores = self.st3.compute_anomaly_score(batch_x)
                st3_scores.extend(scores.cpu().numpy())
        st3_scores = np.array(st3_scores)
        
        # 3. Supervised branch (XGBoost)
        xgb_probs = self.xgb.predict_proba(X_scaled)
        
        # 4. Continuous-Time Dynamic Graph branch (CT-DGNN)
        self.ct_dgnn.eval()
        self.ct_dgnn.reset_states()
        
        # If real IPs present, extract them; otherwise fallback to indexing
        src_ips = [str(x) for x in df_raw.get("src_ip", np.arange(len(df_raw)))]
        dst_ips = [str(x) for x in df_raw.get("dst_ip", np.arange(len(df_raw)) + 10)]
        timestamps = df_raw.get("timestamp", np.linspace(0, len(df_raw) * 0.1, len(df_raw)))
        if isinstance(timestamps, pd.Series):
            timestamps = timestamps.values
            
        # Run graph updates in batches to preserve GPU memory
        graph_features = []
        chunk_size = 1000
        X_scaled_t = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for i in range(0, len(df_raw), chunk_size):
                end_idx = min(i + chunk_size, len(df_raw))
                src = src_ips[i:end_idx]
                dst = dst_ips[i:end_idx]
                times = timestamps[i:end_idx]
                edges = X_scaled_t[i:end_idx]
                
                g_out = self.ct_dgnn(src, dst, times, edges)
                graph_features.extend(g_out.cpu().numpy())
        graph_features = np.array(graph_features)
        
        # 5. Neuro-Symbolic Reinforcement Learning Fusion Decision Loop
        states = extract_rl_states(xgb_probs, st3_scores, X_scaled, df_raw, js_drift)
        
        final_preds = []
        reasons = []
        
        c2e2_enabled = self.config.get("c2e2", {}).get("enabled", True)
        c2e2_threshold = self.config.get("c2e2", {}).get("threshold", 0.995)
        
        override_benign_count = 0
        
        for i in range(len(df_raw)):
            p_xgb = xgb_probs[i]
            e_st3 = st3_scores[i]
            
            # C2E2 Optimization: If XGBoost is highly confident of DoS/DDoS, exit early
            if c2e2_enabled and p_xgb > c2e2_threshold:
                final_preds.append("DoS/DDoS")
                reasons.append("C2E2 Confirmed Attack -- Supervised Core")
                continue
                
            # Otherwise, invoke full RL-Fusion
            action = self.rl_agent.select_action(states[i], evaluate=True)
            decision = self.rl_agent.get_fusion_decision(
                action, p_xgb, e_st3, self.evt.threshold
            )
            
            # Map action back to classification strings
            if decision == 1:
                if p_xgb > 0.5:
                    final_preds.append("DoS/DDoS")
                    reasons.append(f"RL-Fusion Consensus (Action #{action}) -- Known Signature")
                else:
                    final_preds.append("Critical Anomaly")
                    reasons.append(f"RL-Fusion Consensus (Action #{action}) -- Zero-Day Outlier")
            else:
                if e_st3 > self.evt.threshold * 0.7:
                    final_preds.append("Suspicious")
                    reasons.append("Dynamic Graph/Spline Reconstruction Outlier")
                else:
                    final_preds.append("Benign")
                    reasons.append("Operational normal profile")
                    
        # Apply Symbolic overrides
        final_preds, override_benign_count = apply_symbolic_overrides(df_raw, final_preds, reasons)
        
        # Build results package
        results = {
            "final_preds": final_preds,
            "reasons": reasons,
            "detection_stats": {
                "override_benign": override_benign_count
            }
        }
        
        # Save output records
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.outputs_dir, f"run_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        generator = ReportGenerator(self.config, self.outputs_dir)
        report_path = generator.generate_report(
            results=results,
            input_file=filepath,
            models_used={
                "ct_dgnn": "Continuous-Time Dynamic Graph Neural Network",
                "st3": "Spatio-Temporal Tensor-Transformer",
                "xgb": "Supervised XGBoost Detector",
                "rl_fusion": "Neuro-Symbolic Reinforcement Learning Fusion"
            },
            run_dir=run_dir,
            df=df_raw,
            drift_status=drift_status
        )
        generator.save_predictions(df_raw, results, run_dir)
        
        return report_path

    def run_ablation_study(self, test_path: str = None):
        """Runs ablation verification to prove scientific contribution of each model layer."""
        if not self.load_pipeline():
            return
            
        if test_path is None:
            test_dir = self.config["datasets"]["test_dir"]
            # If test_dir is a directory, pick the first suitable CSV file
            if os.path.isdir(test_dir):
                csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
                if csv_files:
                    # Prefer smaller Tier files for ablation (~163-185MB vs MixSet ~4GB)
                    tier_files = [f for f in csv_files if f.startswith('Tier')]
                    chosen = tier_files[0] if tier_files else csv_files[0]
                    test_path = os.path.join(test_dir, chosen)
                    logger.info(f"Ablation: selected test file '{chosen}' from directory")
            elif os.path.isfile(test_dir):
                test_path = test_dir
        
        if test_path is None:
            test_path = self.config["datasets"]["training_combined"]
            
        logger.info("========================================")
        logger.info("INITIATING SCIENTIFIC ABLATION VERIFICATION")
        logger.info("========================================")
        
        df_test = load_csv(test_path, max_rows=5000)
        if df_test is None:
            logger.error("Failed to load ablation test target dataset.")
            return
            
        features = [f.strip().lower() for f in self.config["selected_features"]]
        X_scaled = np.clip(self.scaler.transform(df_test[features].values), -5.0, 5.0)
        y_true = df_test["label"].astype(str).str.strip().str.upper().apply(
            lambda x: 0 if x in ["BENIGN", "0", "0.0"] else 1
        ).values
        
        # Configuration runs
        configs = [
            {"name": "XGBoost Core Only", "use_xgb": True, "use_st3": False, "use_rl": False},
            {"name": "ST3 Unsupervised Only", "use_xgb": False, "use_st3": True, "use_rl": False},
            {"name": "XGBoost + ST3 (Static OR)", "use_xgb": True, "use_st3": True, "use_rl": False},
            {"name": "Typhon Hybrid (Full DQN Fusion)", "use_xgb": True, "use_st3": True, "use_rl": True}
        ]
        
        # Spatio-Temporal features extraction for ST3
        X_seq = make_sequences(X_scaled, seq_len=10)
        dataset_seq = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
        self.st3.eval()
        st3_scores = []
        with torch.no_grad():
            for i in range(0, len(dataset_seq), 128):
                batch_x = dataset_seq[i:i+128]
                scores = self.st3.compute_anomaly_score(batch_x)
                st3_scores.extend(scores.cpu().numpy())
        st3_scores = np.array(st3_scores)
        
        xgb_probs = self.xgb.predict_proba(X_scaled)
        states = extract_rl_states(xgb_probs, st3_scores, X_scaled, df_test, 0.0)
        
        print("\n" + "="*80)
        print(f"{'Ablation Test Layer':<35} | {'Accuracy':<10} | {'Recall':<10} | {'FPR':<10}")
        print("="*80)
        
        for cfg in configs:
            preds = []
            for i in range(len(df_test)):
                p_xgb = xgb_probs[i]
                e_st3 = st3_scores[i]
                
                if cfg["use_rl"]:
                    action = self.rl_agent.select_action(states[i], evaluate=True)
                    decision = self.rl_agent.get_fusion_decision(
                        action, p_xgb, e_st3, self.evt.threshold
                    )
                else:
                    xgb_flag = p_xgb > 0.5
                    st3_flag = e_st3 > self.evt.threshold
                    if cfg["use_xgb"] and cfg["use_st3"]:
                        decision = int(xgb_flag or st3_flag)
                    elif cfg["use_xgb"]:
                        decision = int(xgb_flag)
                    else:
                        decision = int(st3_flag)
                        
                preds.append(decision)
                
            preds = np.array(preds)
            tp = int(((y_true == 1) & (preds == 1)).sum())
            tn = int(((y_true == 0) & (preds == 0)).sum())
            fp = int(((y_true == 0) & (preds == 1)).sum())
            fn = int(((y_true == 1) & (preds == 0)).sum())
            
            acc = (tp + tn) / len(df_test)
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            print(f"{cfg['name']:<35} | {acc:.6f}   | {rec:.6f}   | {fpr:.6f}")
            
        print("="*80 + "\n")
        logger.info("Scientific Ablation verification complete.")


if __name__ == "__main__":
    args = parse_args()
    
    pipeline = TyphonPipeline(args.config)
    
    if args.mode == "train":
        pipeline.train_pipeline()
    elif args.mode == "detect":
        if not args.file:
            logger.error("Error: --file argument is required for detect mode.")
            sys.exit(1)
        pipeline.run_detection(args.file)
    elif args.mode == "evaluate":
        if not args.file:
            logger.error("Error: --file argument is required for evaluate mode.")
            sys.exit(1)
        pipeline.run_detection(args.file)
    elif args.mode == "ablation":
        if args.file:
            pipeline.run_ablation_study(args.file)
        else:
            pipeline.run_ablation_study()
