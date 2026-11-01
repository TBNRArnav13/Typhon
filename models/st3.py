import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy.stats import genpareto
from src.utils import logger


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer.
    
    Uses learnable B-spline basis functions on each input-output edge connection
    before summation, providing edge-centric non-linear expressivity that standard
    MLPs cannot achieve. Implemented with vectorized grid evaluation for speed.
    """
    def __init__(self, in_features: int, out_features: int, n_knots: int = 8, degree: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_knots = n_knots
        self.degree = degree
        self.num_basis = n_knots + degree

        # Base residual linear connection (SiLU-gated for stability)
        self.base_weight = nn.Linear(in_features, out_features)

        # Spline coefficient tensor: (out_features, in_features, num_basis)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, self.num_basis) * 0.01)

        # Uniform knot vector
        grid = torch.linspace(-2.0, 2.0, n_knots + 2 * degree + 1)
        self.register_buffer("grid", grid)

    def _compute_bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized B-spline basis via Cox-de Boor recursion."""
        x_c = torch.clamp(x, self.grid[self.degree], self.grid[-(self.degree + 1)])
        x_u = x_c.unsqueeze(-1)  # (B, F, 1)

        # Degree-0 basis
        bases = ((x_u >= self.grid[:-1]) & (x_u < self.grid[1:])).to(x.dtype)

        for d in range(1, self.degree + 1):
            n_basis = bases.size(-1) - 1
            left = self.grid[:n_basis]
            right = self.grid[d:d + n_basis]
            denom_l = (right - left).clamp(min=1e-8)
            t1 = ((x_c.unsqueeze(-1) - left) / denom_l) * bases[:, :, :n_basis]

            left2 = self.grid[1:n_basis + 1]
            right2 = self.grid[d + 1:d + 1 + n_basis]
            denom_r = (right2 - left2).clamp(min=1e-8)
            t2 = ((right2 - x_c.unsqueeze(-1)) / denom_r) * bases[:, :, 1:n_basis + 1]

            bases = t1 + t2

        return bases  # (B, F, num_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_weight(x)
        bases = self._compute_bspline_basis(x)
        spline_out = torch.einsum("bfk,ofk->bo", bases, self.spline_weight)
        return base_out + spline_out


class LTCTransformerEncoder(nn.Module):
    """
    LTC-Transformer with multi-scale dilational TCN and Rotary-inspired encoding.
    Uses GroupNorm (channel-first compatible) for the TCN branch.
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Multi-scale dilational TCN
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1, groups=1),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=1),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation="gelu"
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        # TCN branch (channel-first)
        res = x
        x_tcn = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = res + x_tcn

        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MPSLayer(nn.Module):
    """
    Matrix Product State (MPS) Tensor Network Contraction Layer.

    To maintain numerical stability, the input is first compressed via a linear
    bottleneck to a small number of MPS sites (default 8). Each site produces a
    bond_dim x bond_dim matrix. Frobenius-norm renormalization at every contraction
    step prevents exponential blowup or decay.
    """
    def __init__(self, in_features: int, n_sites: int = 8, bond_dim: int = 8):
        super().__init__()
        self.n_sites = n_sites
        self.bond_dim = bond_dim

        # Bottleneck: compress d_model features into n_sites
        self.bottleneck = nn.Linear(in_features, n_sites)

        # Per-site matrix projectors
        self.site_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, bond_dim * bond_dim),
                nn.Tanh()
            ) for _ in range(n_sites)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        sites = self.bottleneck(x)  # (B, n_sites)

        m = torch.eye(self.bond_dim, device=x.device).unsqueeze(0).expand(B, -1, -1).clone()

        for i in range(self.n_sites):
            feat = sites[:, i:i+1]  # (B, 1)
            mat = self.site_projectors[i](feat).view(B, self.bond_dim, self.bond_dim)
            m = torch.bmm(m, mat)
            # Frobenius renormalization at every step
            norm = torch.norm(m.view(B, -1), dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
            m = m / norm

        return m.view(B, -1)  # (B, bond_dim^2)


class ST3_Autoencoder(nn.Module):
    """
    Spatio-Temporal Tensor-Transformer (ST3) Autoencoder.

    Architecture:
        Input (B, L, 33)
        -> KAN spatial mapping -> (B, L, latent_dim)
        -> Linear projection -> (B, L, d_model)
        -> LTC-Transformer temporal -> (B, L, d_model)
        -> Last-step extraction -> (B, d_model)
        -> MPS tensor contraction -> (B, bond_dim^2)
        -> Decoder reconstruction -> (B, 33)
    """
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = torch.device(device)

        latent_dim = config["kan"]["latent_dim"]
        d_model = config["transformer"]["d_model"]
        bond_dim = config["qtn"]["bond_dim"]
        n_sites = config["qtn"].get("n_sites", 8)

        # 1. KAN spatial encoder
        self.kan = KANLayer(33, latent_dim, n_knots=config["kan"]["n_knots"])

        # 2. Projection to transformer dimension
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
        )

        # 3. Temporal transformer
        self.transformer = LTCTransformerEncoder(
            d_model=d_model,
            nhead=config["transformer"]["nhead"],
            num_layers=config["transformer"]["num_layers"],
            dim_feedforward=config["transformer"]["dim_feedforward"],
            dropout=config["transformer"]["dropout"],
        )

        # 4. MPS tensor contraction (with bottleneck)
        self.mps = MPSLayer(d_model, n_sites=n_sites, bond_dim=bond_dim)

        # 5. Decoder
        mps_out = bond_dim ** 2
        self.decoder = nn.Sequential(
            nn.Linear(mps_out, 64),
            nn.GELU(),
            nn.Linear(64, 33),
        )

    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        return super().to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, F = x.size()

        # KAN per-step
        x_flat = x.reshape(B * L, F)
        kan_out = self.kan(x_flat).view(B, L, -1)

        # Project + Transform
        proj = self.projector(kan_out)
        temporal = self.transformer(proj)

        # Last-step summary
        summary = temporal[:, -1, :]

        # MPS contraction
        mps_out = self.mps(summary)

        # Decode
        recon = self.decoder(mps_out)
        target = x[:, -1, :]
        return recon, target

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, target = self.forward(x)
        return torch.mean((recon - target) ** 2, dim=-1)


class EVTDetector:
    """
    Adaptive Extreme Value Theory (EVT) threshold engine.

    Fits a Generalized Pareto Distribution (GPD) on reconstruction errors above
    the configurable tail percentile to establish mathematically sound detection
    boundaries for zero-day anomalies.
    """
    def __init__(self, tail_percentile: float = 90.0, sensitivity_sigma: float = 3.0):
        self.tail_percentile = tail_percentile
        self.sensitivity_sigma = sensitivity_sigma
        self.threshold = 1.0
        self.gpd_params: Optional[Tuple[float, float, float]] = None

    def fit(self, errors: np.ndarray):
        try:
            self.u = float(np.percentile(errors, self.tail_percentile))
            tail = errors[errors > self.u] - self.u

            if len(tail) > 10:
                c, loc, scale = genpareto.fit(tail, floc=0.0)
                self.gpd_params = (c, loc, scale)

                prob_tail = 1.0 - (self.tail_percentile / 100.0)
                from scipy.special import erf
                target_prob = 1.0 - erf(self.sensitivity_sigma / np.sqrt(2.0))

                if abs(c) > 1e-4:
                    self.threshold = self.u + (scale / c) * ((prob_tail / target_prob) ** c - 1.0)
                else:
                    self.threshold = self.u + scale * np.log(prob_tail / target_prob)

                logger.info(f"EVT calibrated: u={self.u:.5f}, shape={c:.4f}, scale={scale:.4f}, threshold={self.threshold:.5f}")
            else:
                self.threshold = float(np.mean(errors) + self.sensitivity_sigma * np.std(errors))
                logger.warning(f"Insufficient tail samples. Sigma-fallback threshold: {self.threshold:.5f}")
        except Exception as e:
            self.threshold = float(np.mean(errors) + self.sensitivity_sigma * np.std(errors))
            logger.warning(f"GPD fit failed: {e}. Sigma-fallback threshold: {self.threshold:.5f}")

    def predict_anomaly(self, error: float) -> Tuple[bool, float]:
        is_anomaly = error > self.threshold
        return is_anomaly, error / max(self.threshold, 1e-9)
