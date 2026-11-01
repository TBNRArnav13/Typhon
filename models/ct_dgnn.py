import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) ODE Cell.
    
    Solves the continuous-time dynamics:
        dh/dt = -h/tau + f(h, x)
    using stable Euler discretization and log-stabilized time deltas.
    """
    def __init__(self, node_dim: int, input_dim: int, hidden_dim: int):
        super().__init__()
        # Parameterize time-constant tau
        self.w_tau = nn.Linear(node_dim + input_dim, node_dim)
        # Parameterize the state transition function f
        self.w_f = nn.Linear(node_dim + input_dim, node_dim)
        
    def forward(self, h_prev: torch.Tensor, x: torch.Tensor, dt: float) -> torch.Tensor:
        # Combine historical state and incoming flow features
        combined = torch.cat([h_prev, x], dim=-1)
        
        # tau must be strictly positive. Softplus + epsilon prevents divide-by-zero
        tau = torch.nn.functional.softplus(self.w_tau(combined)) + 1e-4
        
        # Derivative transition bound by tanh
        f_val = torch.tanh(self.w_f(combined))
        
        # Stabilize long intervals via log-scaling to prevent Euler discretization explosion
        dt_eff = torch.log1p(torch.tensor(dt, dtype=torch.float32, device=h_prev.device))
        
        # Solve ODE step: dh/dt = -h/tau + f
        dh_dt = -h_prev / tau + f_val
        h_new = h_prev + dt_eff * dh_dt
        
        # Stability clamping
        return torch.clamp(h_new, -10.0, 10.0)


class CT_DGNN(nn.Module):
    """
    Continuous-Time Dynamic Graph Neural Network (CT-DGNN).
    
    Maintains continuous node states for active IP addresses in the network,
    updating them dynamically upon the arrival of packets/flows via LTC-ODEs.
    """
    def __init__(self, input_dim: int = 33, node_dim: int = 16, edge_dim: int = 8, hidden_dim: int = 32, device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        self.ltc_cell = LTCCell(node_dim, input_dim, hidden_dim)
        
        # Fuses source state, destination state, and raw features into an enriched edge representation
        self.edge_projector = nn.Sequential(
            nn.Linear(node_dim * 2 + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        self.reset_states()

    def reset_states(self):
        """Clears topological context maps and resets all IP state tensors."""
        self.ip_to_idx: Dict[str, int] = {}
        self.max_nodes = 20000
        self.node_states = torch.zeros(self.max_nodes, self.node_dim, device=self.device)
        self.last_update_times = torch.zeros(self.max_nodes, device=self.device)

    def to(self, device: Any):
        self.device = torch.device(device)
        self.node_states = self.node_states.to(self.device)
        self.last_update_times = self.last_update_times.to(self.device)
        return super().to(self.device)

    def _get_node_indices(self, ips: List[str]) -> torch.Tensor:
        """Dynamically registers new IP addresses and handles memory tensor expansion."""
        indices = []
        for ip in ips:
            if ip not in self.ip_to_idx:
                idx = len(self.ip_to_idx)
                self.ip_to_idx[ip] = idx
                if idx >= self.max_nodes:
                    # Dynamically double internal state storage capacity
                    new_max = self.max_nodes * 2
                    new_states = torch.zeros(new_max, self.node_dim, device=self.device)
                    new_times = torch.zeros(new_max, device=self.device)
                    new_states[:self.max_nodes] = self.node_states
                    new_times[:self.max_nodes] = self.last_update_times
                    self.node_states = new_states
                    self.last_update_times = new_times
                    self.max_nodes = new_max
            indices.append(self.ip_to_idx[ip])
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def forward(self, src_ips: List[str], dst_ips: List[str], timestamps: np.ndarray, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Updates continuous states causally and produces graph-enriched edge features.
        
        Args:
            src_ips: List of source IP addresses
            dst_ips: List of destination IP addresses
            timestamps: Array of event timestamps (seconds)
            edge_features: Tensor of raw flow features (batch_size, input_dim)
            
        """
        # Detach states to prevent gradient leaking across different chunks (BPTT boundary)
        self.node_states = self.node_states.detach()
        
        # Maintain temporal causality by sorting execution by arrival timestamps
        sorted_indices = np.argsort(timestamps)
        unsorted_indices = np.argsort(sorted_indices)
        
        sorted_src = [src_ips[i] for i in sorted_indices]
        sorted_dst = [dst_ips[i] for i in sorted_indices]
        sorted_times = timestamps[sorted_indices]
        sorted_edges = edge_features[sorted_indices]
        
        src_node_idx = self._get_node_indices(sorted_src)
        dst_node_idx = self._get_node_indices(sorted_dst)
        
        updated_flow_features = []
        
        # Causally propagate continuous-time messages
        for i in range(len(sorted_indices)):
            u = src_node_idx[i]
            v = dst_node_idx[i]
            t = float(sorted_times[i])
            x_e = sorted_edges[i]
            
            # Retrieve node representations
            h_u = self.node_states[u]
            h_v = self.node_states[v]
            
            t_last_u = float(self.last_update_times[u])
            t_last_v = float(self.last_update_times[v])
            
            # Determine continuous time gap
            dt_u = max(t - t_last_u, 0.0)
            dt_v = max(t - t_last_v, 0.0)
            
            # Solve dynamic ODE updates
            h_u_new = self.ltc_cell(h_u, x_e, dt_u)
            h_v_new = self.ltc_cell(h_v, x_e, dt_v)
            
            # Write back node context
            self.node_states[u] = h_u_new
            self.node_states[v] = h_v_new
            self.last_update_times[u] = t
            self.last_update_times[v] = t
            
            # Fuse updated context with raw flow features
            z_e = self.edge_projector(torch.cat([h_u_new, h_v_new, x_e], dim=0))
            updated_flow_features.append(z_e)
            
        updated_flow_features = torch.stack(updated_flow_features, dim=0)
        # Restore original order of the mini-batch to correspond to caller's alignment
        return updated_flow_features[unsorted_indices]
