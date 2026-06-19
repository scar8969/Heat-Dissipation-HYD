"""
Differentiable Surface Energy Balance (SEB) Head.

Based on SEB-TwinTFT (EPFL 2025):
- Predicts effective SEB parameters from features
- Enables physics-constrained LST derivation
- Learnable gate for blending data-driven and physics-based predictions

SEB Parameters:
    α: albedo
    r_a: aerodynamic resistance
    β: evaporative efficiency
    γ: longwave radiation fraction
    G_frac: ground heat flux fraction

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DifferentiableSEBHead(nn.Module):
    """
    Differentiable SEB parameter prediction head.

    Predicts effective SEB parameters from encoder features,
    then computes LST from energy balance equations.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        # Parameter prediction networks
        self.albedo_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Albedo in [0, 1]
        )

        self.ra_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Aerodynamic resistance > 0
        )

        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Evaporative efficiency in [0, 1]
        )

        self.gamma_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # LW fraction in [0, 1]
        )

        self.gfrac_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # G fraction in [0, 1]
        )

        # Physical constants
        self.register_buffer("sigma", torch.tensor(5.67e-8))  # Stefan-Boltzmann
        self.register_buffer("rho", torch.tensor(1.2))  # Air density (kg/m3)
        self.register_buffer("cp", torch.tensor(1005.0))  # Specific heat (J/kg/K)
        self.register_buffer("Lv", torch.tensor(2.45e6))  # Latent heat (J/kg)
        self.register_buffer("kappa", torch.tensor(0.41))  # Von Karman constant

    def forward(
        self,
        features: torch.Tensor,
        met_forcing: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict SEB parameters and compute LST.

        Args:
            features: (B, C, H, W) encoder features
            met_forcing: (B, 5, H, W) meteorological forcing
                [SW_down, LW_down, T_air, RH, Wind]

        Returns:
            lst: (B, 1, H, W) predicted LST from SEB
            params: Dict of SEB parameters
        """
        B, C, H, W = features.shape

        # Flatten spatial dims for MLP
        feat_flat = features.mean(dim=(2, 3))  # (B, C) global average

        # Predict parameters
        alpha = self.albedo_net(feat_flat)  # (B, 1)
        ra = self.ra_net(feat_flat)  # (B, 1)
        beta = self.beta_net(feat_flat)  # (B, 1)
        gamma = self.gamma_net(feat_flat)  # (B, 1)
        g_frac = self.gfrac_net(feat_flat)  # (B, 1)

        # Expand to spatial dims
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        ra = ra.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        g_frac = g_frac.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Extract met forcing
        if met_forcing is not None:
            sw_down = met_forcing[:, 0:1]  # (B, 1, H, W)
            lw_down = met_forcing[:, 1:2]
            t_air = met_forcing[:, 2:3]
            rh = met_forcing[:, 3:4]
            wind = met_forcing[:, 4:5]
        else:
            sw_down = torch.full((B, 1, H, W), 500.0, device=features.device)
            lw_down = torch.full((B, 1, H, W), 350.0, device=features.device)
            t_air = torch.full((B, 1, H, W), 30.0, device=features.device)
            rh = torch.full((B, 1, H, W), 0.5, device=features.device)
            wind = torch.full((B, 1, H, W), 2.0, device=features.device)

        # Compute LST from SEB
        lst = self._compute_lst_from_seb(alpha, ra, beta, gamma, g_frac,
                                          sw_down, lw_down, t_air, rh, wind)

        params = {
            "alpha": alpha,
            "ra": ra,
            "beta": beta,
            "gamma": gamma,
            "g_frac": g_frac,
        }

        return lst, params

    def _compute_lst_from_seb(
        self,
        alpha: torch.Tensor,
        ra: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        g_frac: torch.Tensor,
        sw_down: torch.Tensor,
        lw_down: torch.Tensor,
        t_air: torch.Tensor,
        rh: torch.Tensor,
        wind: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LST from SEB equations.

        SEB: Rn = H + LE + G

        Rn = SW↓(1-α) + (1-γ)(LW↓ - σTₛ⁴)
        H = ρcₚ(Tₛ - Tₐ)/rₐ
        LE = βρLᵥ(q_sat(Tₛ) - qₐ)/(rₐ + rₛ)
        G = G_frac·Rn
        """
        # Convert to Kelvin
        t_air_k = t_air + 273.15

        # Saturation specific humidity (Buck equation)
        es = 611.2 * torch.exp(17.67 * (t_air_k - 273.15) / (t_air_k - 273.15 + 243.5))
        q_sat = 0.622 * es / (101325 - 0.378 * es)
        q_a = (rh / 100) * q_sat

        # Surface resistance (empirical for urban)
        r_s = torch.where(
            beta > 0.1,
            100.0 / (beta + 1e-6),
            torch.tensor(1e6, device=beta.device),
        )

        # Initial guess for T_s
        t_s = t_air_k.clone()

        # Newton-Raphson iterations for energy balance closure
        for _ in range(5):
            # Net radiation
            rn = sw_down * (1 - alpha) + (1 - gamma) * (lw_down - self.sigma * t_s**4)

            # Sensible heat flux
            h = self.rho * self.cp * (t_s - t_air_k) / (ra + 1e-6)

            # Latent heat flux
            es_s = 611.2 * torch.exp(17.67 * (t_s - 273.15) / (t_s - 273.15 + 243.5))
            q_sat_s = 0.622 * es_s / (101325 - 0.378 * es_s)
            le = beta * self.rho * self.Lv * (q_sat_s - q_a) / (ra + r_s + 1e-6)

            # Ground heat flux
            g = g_frac * rn

            # Residual
            residual = rn - h - le - g

            # Derivative of residual w.r.t. T_s
            drn_dts = -4 * (1 - gamma) * self.sigma * t_s**3
            dh_dts = self.rho * self.cp / (ra + 1e-6)
            des_dts = es_s * 17.67 * 243.5 / (t_s - 273.15 + 243.5)**2
            dqs_dts = 0.622 * des_dts / (101325 - 0.378 * es_s)
            dle_dts = beta * self.rho * self.Lv * dqs_dts / (ra + r_s + 1e-6)
            dg_dts = g_frac * drn_dts

            dresidual_dts = drn_dts - dh_dts - dle_dts - dg_dts

            # Update
            t_s = t_s - residual / (dresidual_dts + 1e-6)

        # Convert to Celsius before returning
        return t_s - 273.15
