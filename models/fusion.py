import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb
import random
from collections import deque
from typing import Dict, Any, Tuple, List, Optional
from src.utils import logger, get_xgboost_device


class XGBoostDetector:
    """
    Supervised XGBoost Intrusion Detector.
    
    Highly optimized gradient-boosted tree classifier leveraging CUDA acceleration
    if available, trained on known signature distributions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        xgb_cfg = config.get("xgboost", {})
        
        # Check if CUDA is available for XGBoost
        dev = get_xgboost_device()
        tree_method = "hist"  # "auto" deprecated in XGBoost 2.0+; "hist" works on both CPU and GPU
        device_param = "cuda" if dev == "cuda" else "cpu"
        
        self.model = xgb.XGBClassifier(
            max_depth=xgb_cfg.get("max_depth", 8),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            n_estimators=xgb_cfg.get("n_estimators", 300),
            objective=xgb_cfg.get("objective", "binary:logistic"),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            tree_method=tree_method,
            device=device_param,
            random_state=42
        )
        logger.info(f"XGBoost Detector initialized with device={device_param}, tree_method={tree_method}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info("Fitting XGBoost Classifier on training dataset...")
        self.model.fit(X, y)
        logger.info("XGBoost Classifier training completed successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Return probability of class 1 (malicious)
            return self.model.predict_proba(X)[:, 1]


class QNetwork(nn.Module):
    """Deep Q-Network (DQN) for decision optimization."""
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay memory buffer for stable training of RL agent."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.int64),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


class RLFusionAgent:
    """
    Neuro-Symbolic Reinforcement Learning Fusion Engine.
    
    Uses Deep Q-Learning to dynamically balance supervised signatures and unsupervised anomalies.
    
    6D State Representation:
        S = [p_xgb, e_st3, H_entropy, V_flow, L_system, delta_drift]
        
    4 Actions:
        A0: Trust XGBoost
        A1: Trust ST3
        A2: OR Fusion (Flag if EITHER model flags) - High Sensitivity
        A3: AND Fusion (Flag only if BOTH models flag) - Low False Positive
    """
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        rl_cfg = config.get("rl_fusion", {})
        
        self.state_dim = rl_cfg.get("state_dim", 6)
        self.n_actions = rl_cfg.get("n_actions", 4)
        self.learning_rate = rl_cfg.get("learning_rate", 0.0005)
        self.gamma = rl_cfg.get("gamma", 0.99)
        self.batch_size = rl_cfg.get("batch_size", 64)
        
        # Q-Network & Target Q-Network setup
        self.q_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(rl_cfg.get("memory_capacity", 10000))
        
        # Epsilon-greedy schedule
        self.epsilon = rl_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = rl_cfg.get("epsilon_end", 0.05)
        self.epsilon_decay = rl_cfg.get("epsilon_decay", 0.995)

    def to(self, device: Any):
        self.device = torch.device(device)
        self.q_net = self.q_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        return self

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Selects action using Epsilon-Greedy Exploration."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
            
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=-1).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Runs a single Q-learning optimization step."""
        if len(self.memory) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        
        # Current predictions
        q_vals = self.q_net(states_t).gather(1, actions_t).squeeze(-1)
        
        # Target predictions
        with torch.no_grad():
            next_q_vals = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + (self.gamma * next_q_vals * (1.0 - dones_t))
            
        loss = nn.functional.mse_loss(q_vals, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clamp gradients to stabilize RL training step
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        return float(loss.item())

    def get_fusion_decision(self, action: int, xgb_prob: float, st3_score: float, st3_thresh: float) -> int:
        """
        Neuro-symbolic decision fusion.
        
        A0: Supervised Core Baseline (Trust XGBoost)
        A1: Unsupervised Anomaly Heavy (ST3 Anomaly override)
        A2: OR Fusion (High Sensitivity -- Flag if EITHER flags)
        A3: AND Fusion (High Precision -- Flag only if BOTH flag)
        """
        xgb_flag = int(xgb_prob > 0.5)
        st3_flag = int(st3_score > st3_thresh * 0.7)
        
        # High-confidence supervised detections are preserved
        if xgb_prob > 0.5:
            if action == 3 and not st3_flag and xgb_prob < 0.9:
                return 0  # High-precision suppression of doubtful signature matches
            return 1
            
        # Low-confidence / ambiguous flows (p_xgb <= 0.5):
        # ST3 structural anomaly score upgrades the verdict to Attack
        if st3_flag:
            return 1
            
        return 0

    def compute_asymmetric_reward(self, prediction: int, true_label: int, xgb_prob: float = 0.5, st3_score: float = 0.0, st3_thresh: float = 1.0) -> float:
        """
        Calculates asymmetric recall-maximizing reward with zero-day incentive.
        
        - TP (Correct Attack): +15.0 (+25.0 if zero-day discovery bonus)
        - TN (Correct Benign): +5.0
        - FN (Missed Attack): -150.0 (Severe penalty for security breach)
        - FP (False Alarm): -20.0 (Moderate penalty for warning overhead)
        """
        if prediction == true_label:
            if true_label == 1:
                # Zero-day discovery bonus: XGBoost was uncertain (p_xgb < 0.7) but ST3 caught it
                if xgb_prob < 0.7 and st3_score > st3_thresh:
                    return 25.0
                return 15.0
            return 5.0
            
        if true_label == 1 and prediction == 0:
            # False Negative (Unacceptable attack leak)
            return -150.0
            
        if true_label == 0 and prediction == 1:
            # False Positive (Transient warning overhead)
            return -20.0
            
        return 0.0

