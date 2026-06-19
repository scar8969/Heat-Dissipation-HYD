"""
SegFormer-PINN: Physics-Informed Neural Network for Urban Heat.

Main hybrid architecture combining:
- SegFormer MiT-B3 encoder (spatial features)
- CuboidAttention temporal fusion
- Differentiable SEB head (physics)
- Flux prediction head (multi-task)
- UrbanGraph layer (spatial adjacency)
- Learnable gate for physics-data blending

Architecture:
    Input (B, 26, H, W) → MiT-B3 → [F1, F2, F3, F4]
    → CuboidAttention → F_fused
    → SEB Head → LST_physics
    → Flux Head → LST_data, Rn, H, LE, G
    → UrbanGraph → spatial refinement
    → Gate → LST_final = α·LST_physics + (1-α)·LST_data

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .mix_transformer import MixVisionTransformer
from .cuboid_attention import CuboidAttentionBlock
from .seb_head import DifferentiableSEBHead
from .flux_head import FluxHead
from .graph_layer import UrbanGraphLayer


class SegFormerPINN(nn.Module):
    """
    Physics-Informed SegFormer for urban heat mapping.

    Combines data-driven learning with physics constraints
    through a differentiable SEB head and learnable gate.
    """

    def __init__(
        self,
        in_channels: int = 26,
        variant: str = "MiT-B3",
        pretrained: bool = False,
        use_graph: bool = True,
        use_temporal: bool = True,
        cuboid_size: Tuple[int, int, int] = (4, 4, 4),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.use_graph = use_graph
        self.use_temporal = use_temporal

        # 1. SegFormer Encoder
        self.encoder = MixVisionTransformer(
            in_channels=in_channels,
            variant=variant,
            pretrained=pretrained,
            dropout=dropout,
        )

        # Get encoder output dimensions
        embed_dims = self.encoder.embed_dims  # [64, 128, 320, 512]
        last_dim = embed_dims[-1]  # 512

        # 2. CuboidAttention Temporal Fusion
        if use_temporal:
            self.temporal_fusion = CuboidAttentionBlock(
                dim=last_dim,
                cuboid_size=cuboid_size,
                num_heads=8,
                num_global_vectors=8,
                num_layers=2,
                dropout=dropout,
            )

        # 3. Differentiable SEB Head
        self.seb_head = DifferentiableSEBHead(
            in_dim=last_dim,
            hidden_dim=256,
        )

        # 4. Flux Prediction Head
        self.flux_head = FluxHead(
            in_dim=last_dim,
            hidden_dim=256,
        )

        # 5. UrbanGraph Layer (optional)
        if use_graph:
            self.graph_layer = UrbanGraphLayer(
                in_dim=last_dim,
                hidden_dim=128,
                num_heads=4,
            )

        # 6. Learnable Gate (physics-data blending)
        self.gate = nn.Sequential(
            nn.Linear(last_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 7. Final prediction head
        self.final_head = nn.Sequential(
            nn.Conv2d(last_dim, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        met_forcing: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input features (26 channels)
            met_forcing: (B, 5, H, W) meteorological forcing
            return_features: Whether to return intermediate features

        Returns:
            Dict with keys:
                - LST: (B, 1, H, W) final LST prediction
                - Rn: (B, 1, H, W) net radiation
                - H: (B, 1, H, W) sensible heat flux
                - LE: (B, 1, H, W) latent heat flux
                - G: (B, 1, H, W) ground heat flux
                - gate: (B, 1, H, W) gate values
                - features: (optional) encoder features
        """
        B, C, H, W = x.shape

        # 1. Encode spatial features
        features = self.encoder(x)  # List of 4 feature maps

        # Use last stage features
        f4 = features[-1]  # (B, 512, H/32, W/32)

        # 2. Temporal fusion (if using temporal data)
        if self.use_temporal and met_forcing is not None:
            # Expand features for temporal dimension
            f4_expanded = f4.unsqueeze(1)  # (B, 1, 512, H', W')
            f4_fused = self.temporal_fusion(f4, f4_expanded)
        else:
            f4_fused = f4

        # 3. UrbanGraph refinement (if enabled)
        if self.use_graph:
            f4_refined = self.graph_layer(f4_fused)
        else:
            f4_refined = f4_fused

        # 4. Upsample f4_refined to full resolution for SEB/Flux heads
        f4_upsampled = F.interpolate(f4_refined, size=(H, W), mode="bilinear", align_corners=False)

        # 5. Physics-based LST
        lst_physics, seb_params = self.seb_head(f4_upsampled, met_forcing)

        # 6. Data-driven flux predictions
        fluxes, lst_data = self.flux_head(f4_upsampled, met_forcing, apply_physics=False)

        # 7. Learnable gate
        gate_input = torch.cat([
            f4_upsampled.mean(dim=(2, 3)),  # Global average of features
            lst_physics.mean(dim=(2, 3)),  # Mean physics LST
        ], dim=1)

        gate_value = self.gate(gate_input)  # (B, 1)

        # Expand gate to spatial dims
        gate_map = gate_value.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # 8. Blend physics and data-driven predictions
        lst_final = gate_map * lst_physics + (1 - gate_map) * lst_data

        # 9. Final refinement
        lst_refined = self.final_head(f4_upsampled)

        # Residual connection
        lst_final = lst_final + lst_refined

        # Prepare output
        output = {
            "LST": lst_final,
            "lst_physics": lst_physics,
            "lst_data": lst_data,
            "gate": gate_map,
            "Rn": fluxes["Rn"],
            "H": fluxes["H"],
            "LE": fluxes["LE"],
            "G": fluxes["G"],
            **seb_params,
        }

        if return_features:
            output["features"] = features

        return output


class SegFormerPINNSmall(nn.Module):
    """
    Lightweight variant for faster training/inference.

    Uses MiT-B0 instead of MiT-B3.
    """

    def __init__(self, in_channels: int = 26, **kwargs):
        super().__init__()

        # Use smaller encoder
        self.encoder = MixVisionTransformer(
            in_channels=in_channels,
            variant="MiT-B0",
            pretrained=False,
        )

        embed_dims = self.encoder.embed_dims  # [32, 64, 160, 256]
        last_dim = embed_dims[-1]  # 256

        # Simplified heads
        self.seb_head = DifferentiableSEBHead(in_dim=last_dim, hidden_dim=128)
        self.flux_head = FluxHead(in_dim=last_dim, hidden_dim=128)

        # Final head
        self.final_head = nn.Sequential(
            nn.Conv2d(last_dim, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        met_forcing: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Simplified forward pass."""
        features = self.encoder(x)
        f4 = features[-1]

        lst_physics, seb_params = self.seb_head(f4, met_forcing)
        fluxes, lst_data = self.flux_head(f4, met_forcing, apply_physics=False)

        # Simple average of physics and data
        lst_final = (lst_physics + lst_data) / 2

        return {
            "LST": lst_final,
            "lst_physics": lst_physics,
            "lst_data": lst_data,
            **fluxes,
            **seb_params,
        }
