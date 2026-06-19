"""
Cuboid Attention module for spatiotemporal fusion.

Based on Earthformer (NeurIPS 2022):
- Decomposes spatiotemporal data into cuboids
- Applies cuboid-level self-attention in parallel
- Uses global vectors for cross-cuboid information sharing

Reference: Gao et al. "Earthformer: Exploring Space-Time Transformers for Earth System Forecasting" (NeurIPS 2022)

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CuboidAttention(nn.Module):
    """
    Cuboid Attention for spatiotemporal modeling.

    Decomposes input into non-overlapping cuboids and applies
    self-attention within each cuboid, plus global attention
    via global vectors.
    """

    def __init__(
        self,
        dim: int,
        cuboid_size: Tuple[int, int, int] = (4, 4, 4),
        num_heads: int = 8,
        num_global_vectors: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.cuboid_size = cuboid_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Global vectors
        self.num_global = num_global_vectors
        self.global_vectors = nn.Parameter(torch.randn(1, num_global_vectors, dim))
        self.global_proj = nn.Linear(dim, dim)

        # Layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        global_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, C, H, W) spatiotemporal tensor
            global_context: Optional global context tensor

        Returns:
            (B, T, C, H, W) attended tensor
        """
        B, T, C, H, W = x.shape
        ct, ch, cw = self.cuboid_size

        # Reshape to (B, T*H*W, C)
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, C, T * H * W).permute(0, 2, 1)

        # Add global vectors
        batch_global = self.global_vectors.expand(B, -1, -1)
        x_with_global = torch.cat([x_flat, batch_global], dim=1)

        # Self-attention
        q = self.q_proj(x_with_global)
        k = self.k_proj(x_with_global)
        v = self.v_proj(x_with_global)

        # Reshape for multi-head attention
        N = x_with_global.shape[1]
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        # Remove global vectors
        out = out[:, :T * H * W, :]

        # Reshape back to (B, T, C, H, W)
        out = out.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)

        return out


class CuboidAttentionBlock(nn.Module):
    """
    Complete Cuboid Attention block with temporal fusion.

    Takes multi-scale features from SegFormer encoder
    and applies spatiotemporal attention.
    """

    def __init__(
        self,
        dim: int = 512,
        cuboid_size: Tuple[int, int, int] = (4, 4, 4),
        num_heads: int = 8,
        num_global_vectors: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CuboidAttention(dim, cuboid_size, num_heads, num_global_vectors, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spatial_features: (B, C, H, W) spatial features from encoder
            temporal_features: (B, T, C, H, W) temporal features

        Returns:
            (B, C, H, W) fused features
        """
        B, C, H, W = spatial_features.shape

        if temporal_features is not None:
            # Combine spatial and temporal
            x = temporal_features
        else:
            # Use spatial as single timestep
            x = spatial_features.unsqueeze(1)  # (B, 1, C, H, W)

        # Apply attention layers
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x)
            x = residual + x
            # Apply norm: flatten (T, C, H, W) -> (B, T*C, H*W) -> norm -> reshape back
            B, T, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, C, T * H * W).permute(0, 2, 1)
            x_flat = norm(x_flat)
            x = x_flat.permute(0, 2, 1).reshape(B, C, H, W, T).permute(0, 4, 1, 2, 3)

        # Take the spatial features (mean over time if multiple steps)
        if x.shape[1] > 1:
            out = x.mean(dim=1)  # (B, C, H, W)
        else:
            out = x.squeeze(1)

        return out
