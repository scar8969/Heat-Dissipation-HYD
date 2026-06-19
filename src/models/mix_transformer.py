"""
MixVisionTransformer (MiT) encoder for SegFormer.

Implements the hierarchical Transformer encoder from SegFormer:
- Overlap Patch Merging
- Efficient Self-Attention with Sequence Reduction
- Multi-scale feature extraction (4 stages)

Reference: Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (2021)

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class OverlapPatchMerging(nn.Module):
    """Overlapping patch merging with convolution."""

    def __init__(self, in_channels: int, out_channels: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with sequence reduction."""

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Sequence reduction
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            # Sequence reduction
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with depthwise convolution."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with efficient attention and MixFFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class MixVisionTransformer(nn.Module):
    """
    MixVisionTransformer (MiT) encoder for SegFormer.

    Outputs multi-scale features at 4 stages:
        Stage 1: (B, embed_dims[0], H/4, W/4)
        Stage 2: (B, embed_dims[1], H/8, W/8)
        Stage 3: (B, embed_dims[2], H/16, W/16)
        Stage 4: (B, embed_dims[3], H/32, W/32)

    Variants:
        MiT-B0: embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2]
        MiT-B1: embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2]
        MiT-B2: embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3]
        MiT-B3: embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3]
        MiT-B5: embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3]
    """

    VARIANTS = {
        "MiT-B0": {"embed_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2], "num_heads": [1, 2, 5, 8], "sr_ratios": [8, 4, 2, 1]},
        "MiT-B1": {"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2], "num_heads": [1, 2, 5, 8], "sr_ratios": [8, 4, 2, 1]},
        "MiT-B2": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3], "num_heads": [1, 2, 5, 8], "sr_ratios": [8, 4, 2, 1]},
        "MiT-B3": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3], "num_heads": [1, 2, 5, 8], "sr_ratios": [8, 4, 2, 1]},
        "MiT-B5": {"embed_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3], "num_heads": [1, 2, 5, 8], "sr_ratios": [8, 4, 2, 1]},
    }

    def __init__(
        self,
        in_channels: int = 18,
        variant: str = "MiT-B3",
        pretrained: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        config = self.VARIANTS[variant]
        embed_dims = config["embed_dims"]
        depths = config["depths"]
        num_heads = config["num_heads"]
        sr_ratios = config["sr_ratios"]

        self.embed_dims = embed_dims
        self.depths = depths

        # Patch embedding stages
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Stage 1
        self.patch_embeds.append(OverlapPatchMerging(in_channels, embed_dims[0], patch_size=7, stride=4))
        for _ in range(depths[0]):
            self.blocks.append(TransformerBlock(embed_dims[0], num_heads[0], sr_ratios[0], dropout=dropout))
        self.norms.append(nn.LayerNorm(embed_dims[0]))

        # Stages 2-4
        for i in range(1, 4):
            self.patch_embeds.append(OverlapPatchMerging(embed_dims[i-1], embed_dims[i], patch_size=3, stride=2))
            for _ in range(depths[i]):
                self.blocks.append(TransformerBlock(embed_dims[i], num_heads[i], sr_ratios[i], dropout=dropout))
            self.norms.append(nn.LayerNorm(embed_dims[i]))

        # Number of blocks per stage
        self.block_idx = []
        idx = 0
        for d in depths:
            self.block_idx.append(list(range(idx, idx + d)))
            idx += d

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            List of 4 feature maps at different scales
        """
        features = []
        B = x.shape[0]

        for stage_idx in range(4):
            # Patch embedding (expects B, C, H, W)
            x = self.patch_embeds[stage_idx](x)

            # After patch embedding: x is (B, N, C) where N = H' * W'
            N = x.shape[1]
            C = x.shape[2]

            # Get spatial dimensions from total tokens
            H = int(math.sqrt(N))
            W = N // H

            # Transformer blocks (work on (B, N, C) format)
            for block_idx in self.block_idx[stage_idx]:
                x = self.blocks[block_idx](x, H, W)

            x = self.norms[stage_idx](x)

            # Reshape to (B, C, H, W) for feature output and next stage input
            x = x.transpose(1, 2).reshape(B, C, H, W)
            features.append(x)

        return features


def load_pretrained_mit(model: MixVisionTransformer, pretrained_path: str) -> MixVisionTransformer:
    """
    Load pretrained MiT weights.

    Args:
        model: MiT model instance
        pretrained_path: Path to pretrained checkpoint

    Returns:
        Model with loaded weights
    """
    if pretrained_path and Path(pretrained_path).exists():
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    return model
