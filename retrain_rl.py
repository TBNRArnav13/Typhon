import os
import sys
import torch
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_csv
from main import TyphonPipeline, make_sequences, extract_rl_states
from models.fusion import RLFusionAgent

logging.basicConfig(level=logging.INFO)
print("Loading Typhon pipeline states...")
pipeline = TyphonPipeline()
if not pipeline.load_pipeline():
    print("Could not load pipeline!")
    sys.exit(1)

print("Loading training dataset for RL experience collection...")
train_path = "Dataset/Traning/Final_70_30_Dataset.csv"
df_train = load_csv(train_path, max_rows=60000)

labels_str = df_train["label"].astype(str).str.strip().str.upper().values
y_true = np.where(np.isin(labels_str, ["BENIGN", "0", "0.0"]), 0, 1)

features = [f.strip().lower() for f in pipeline.config["selected_features"]]
X_raw = df_train[features].values
X_scaled = np.clip(pipeline.scaler.transform(X_raw), -5.0, 5.0)

print("Extracting ST3 anomaly scores...")
X_seq = make_sequences(X_scaled, seq_len=10)
dataset_seq = torch.tensor(X_seq, dtype=torch.float32, device=pipeline.device)

pipeline.st3.eval()
st3_scores = []
with torch.no_grad():
    for i in range(0, len(dataset_seq), 512):
        batch_x = dataset_seq[i:i+512]
        scores = pipeline.st3.compute_anomaly_score(batch_x)
        st3_scores.extend(scores.cpu().numpy())
st3_scores = np.array(st3_scores)

print("Extracting XGBoost predictions...")
xgb_probs = pipeline.xgb.predict_proba(X_scaled)
st3_thresh = pipeline.evt.threshold

# Build RL training states over representative dataset
print("Constructing 6D RL state matrices...")
states = extract_rl_states(xgb_probs, st3_scores, X_scaled, df_train, js_drift=0.0)
N = len(states)

print("Initializing RL Agent for Q-network optimization...")
pipeline.rl_agent = RLFusionAgent(pipeline.config, device=pipeline.device.type).to(pipeline.device)

xgb_flags = (xgb_probs > 0.5).astype(int)
st3_flags = (st3_scores > st3_thresh).astype(int)

episodes = 25
states_t = torch.tensor(states, dtype=torch.float32, device=pipeline.device)

for ep in range(1, episodes + 1):
    with torch.no_grad():
        q_vals = pipeline.rl_agent.q_net(states_t)
        greedy_actions = torch.argmax(q_vals, dim=-1).cpu().numpy()
        
    rand_mask = np.random.random(N) < pipeline.rl_agent.epsilon
    random_actions = np.random.randint(0, 4, size=N)
    actions = np.where(rand_mask, random_actions, greedy_actions)
    
    preds = np.where(actions == 0, xgb_flags,
            np.where(actions == 1, st3_flags,
            np.where(actions == 2, xgb_flags | st3_flags,
                                   xgb_flags & st3_flags)))
                                   
    tp = (preds == 1) & (y_true == 1)
    tn = (preds == 0) & (y_true == 0)
    fn = (preds == 0) & (y_true == 1)
    fp = (preds == 1) & (y_true == 0)
    
    rewards = (tp * 15.0) + (tn * 5.0) + (fn * -150.0) + (fp * -20.0)
    total_reward = float(np.sum(rewards))
    
    sample_indices = np.random.choice(N - 1, size=3000, replace=False)
    for idx in sample_indices:
        pipeline.rl_agent.store_transition(states[idx], actions[idx], rewards[idx], states[idx+1], False)
        
    losses = []
    for _ in range(60):
        loss = pipeline.rl_agent.train_step()
        if loss is not None:
            losses.append(loss)
            
    pipeline.rl_agent.decay_epsilon()
    pipeline.rl_agent.update_target_network()
    mean_loss = np.mean(losses) if losses else 0.0
    print(f"Episode {ep:2d}/{episodes:2d} | Total Reward: {total_reward:10.1f} | Epsilon: {pipeline.rl_agent.epsilon:.3f} | Loss: {mean_loss:.5f}", flush=True)

pipeline.save_pipeline()
print("Saved retrained RL agent successfully!", flush=True)
