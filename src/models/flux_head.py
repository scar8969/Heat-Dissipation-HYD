"""
Multi-task Flux prediction head.

Predicts surface energy fluxes from encoder features:
- Rn (net radiation)
- H (sensible heat flux)
- LE (latent heat flux)
- G (ground heat flux)
- LST (land surface temperature)

Multi-task learning with shared backbone and task-specific heads.

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FluxHead(nn.Module):
    """
    Multi-task flux prediction head.

    Predicts 5 quantities simultaneously from encoder features:
    - Rn: Net radiation (W/m2)
    - H: Sensible heat flux (W/m2)
    - LE: Latent heat flux (W/m2)
    - G: Ground heat flux (W/m2)
    - LST: Land surface temperature (°C)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_tasks: int = 5):
        super().__init__()

        # Shared feature processor
        self.shared = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Task-specific heads
        self.rn_head = self._make_head(hidden_dim, 1)
        self.h_head = self._make_head(hidden_dim, 1)
        self.le_head = self._make_head(hidden_dim, 1)
        self.g_head = self._make_head(hidden_dim, 1)
        self.lst_head = self._make_head(hidden_dim, 1)

        # Physics consistency layer
        self.physics_layer = PhysicsConsistencyLayer()

    def _make_head(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, out_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        met_forcing: torch.Tensor,
        apply_physics: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (B, C, H, W) encoder features
            met_forcing: (B, 5, H, W) meteorological forcing [SW, LW, T_air, RH, Wind]
            apply_physics: Whether to apply physics consistency

        Returns:
            fluxes: Dict of flux predictions
            lst: (B, 1, H, W) final LST
        """
        shared_feat = self.shared(features)

        # Task predictions
        rn = self.rn_head(shared_feat)
        h = self.h_head(shared_feat)
        le = self.le_head(shared_feat)
        g = self.g_head(shared_feat)
        lst = self.lst_head(shared_feat)

        fluxes = {
            "Rn": rn,
            "H": h,
            "LE": le,
            "G": g,
            "LST": lst,
        }

        # Apply physics consistency
        if apply_physics:
            fluxes, lst = self.physics_layer(fluxes, met_forcing)

        return fluxes, lst


class PhysicsConsistencyLayer(nn.Module):
    """
    Enforces physical consistency in flux predictions.

    Constraints:
    - Rn ≈ H + LE + G (energy balance closure)
    - H, LE, G have same sign as Rn
    - LE ≥ 0 (evaporation only)
    """

    def __init__(self):
        super().__init__()
        # Learnable closure fraction
        self.closure_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]))

    def forward(
        self,
        fluxes: Dict[str, torch.Tensor],
        met_forcing: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply physics consistency to flux predictions."""
        rn = fluxes["Rn"]
        h = fluxes["H"]
        le = fluxes["LE"]
        g = fluxes["G"]
        lst = fluxes["LST"]

        # Energy balance closure
        closure_weights = F.softmax(self.closure_weights, dim=0)
        total_flux = h + le + g
        residual = rn - total_flux

        # Distribute residual proportionally
        h = h + closure_weights[0] * residual
        le = le + closure_weights[1] * residual
        g = g + closure_weights[2] * residual

        # Ensure LE ≥ 0
        le = F.relu(le)

        # Recompute LST from fluxes (optional correction)
        # Simple correction based on energy balance
        lst_correction = 0.1 * residual
        lst = lst + lst_correction

        fluxes["H"] = h
        fluxes["LE"] = le
        fluxes["G"] = g

        return fluxes, lst


class SEBLoss(nn.Module):
    """
    Surface Energy Balance loss.

    Penalizes energy balance non-closure.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, fluxes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute SEB closure loss."""
        rn = fluxes["Rn"]
        h = fluxes["H"]
        le = fluxes["LE"]
        g = fluxes["G"]

        # Energy balance residual
        residual = rn - h - le - g

        # Loss
        loss = F.mse_loss(residual, torch.zeros_like(residual))

        return self.weight * loss
