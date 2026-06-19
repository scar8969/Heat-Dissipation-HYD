"""
UrbanGraph layer for spatial adjacency modeling.

Models spatial dependencies between pixels using graph neural networks.
Each pixel is a node, with edges connecting neighboring pixels.

Applications:
- Temperature propagation between neighboring pixels
- Urban form effects (building density, height)
- Green-blue infrastructure effects

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UrbanGraphLayer(nn.Module):
    """
    Graph neural network layer for urban spatial modeling.

    Constructs a graph from pixel features and applies
    message passing to capture spatial dependencies.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Node feature projection
        self.node_proj = nn.Linear(in_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(3, hidden_dim)  # dx, dy, distance

        # Message passing layers
        self.mp1 = GraphConv(hidden_dim, hidden_dim, num_heads)
        self.mp2 = GraphConv(hidden_dim, hidden_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, in_dim)

        # Normalization
        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        radius: int = 3,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input features
            radius: Neighborhood radius for graph edges

        Returns:
            (B, C, H, W) graph-processed features
        """
        B, C, H, W = x.shape

        # Project node features
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)

        # Construct graph edges
        edge_index, edge_attr = self._construct_graph(H, W, radius, x.device)

        # Message passing (process each batch element separately)
        h_list = []
        for b in range(B):
            h_b = x_flat[b]  # (N, C)
            h_b = self.node_proj(h_b)  # (N, hidden)
            h_b = self.mp1(h_b, edge_index, edge_attr)
            h_b = F.gelu(h_b)
            h_b = self.mp2(h_b, edge_index, edge_attr)
            h_list.append(h_b)

        h = torch.stack(h_list, dim=0)  # (B, N, hidden)

        # Project back
        out = self.out_proj(h)  # (B, N, C)

        # Residual connection
        out = out + x_flat
        out = self.norm(out)

        # Reshape back
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out

    def _construct_graph(self, H: int, W: int, radius: int, device: torch.device):
        """Construct graph edges for spatial grid."""
        # Create grid indices
        rows, cols = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        rows = rows.flatten()
        cols = cols.flatten()
        num_nodes = H * W

        edge_list = []
        edge_attr_list = []

        # Connect each node to neighbors within radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue

                # Compute target positions
                target_rows = rows + dr
                target_cols = cols + dc

                # Valid neighbors
                valid = (
                    (target_rows >= 0) & (target_rows < H) &
                    (target_cols >= 0) & (target_cols < W)
                )

                if valid.any():
                    src_idx = rows[valid] * W + cols[valid]
                    dst_idx = target_rows[valid] * W + target_cols[valid]

                    edge_list.append(torch.stack([src_idx, dst_idx], dim=0))

                    # Edge attributes: dx, dy, distance
                    dx = torch.full_like(src_idx, dc, dtype=torch.float32)
                    dy = torch.full_like(src_idx, dr, dtype=torch.float32)
                    dist = torch.sqrt(dx**2 + dy**2)
                    edge_attr_list.append(torch.stack([dx, dy, dist], dim=1))

        edge_index = torch.cat(edge_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        return edge_index, edge_attr


class GraphConv(nn.Module):
    """
    Graph convolution layer with multi-head attention.

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        num_heads: Number of attention heads
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        # Edge feature integration
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Graph convolution with edge features.

        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, 3) edge attributes [dx, dy, dist]

        Returns:
            (N, out_dim) updated node features
        """
        N = x.shape[0]
        src, dst = edge_index

        # Compute attention
        q = self.q_proj(x)[dst]  # (E, out)
        k = self.k_proj(x)[src]  # (E, out)
        v = self.v_proj(x)[src]  # (E, out)

        # Reshape for multi-head
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Edge features
        edge_emb = self.edge_mlp(edge_attr)  # (E, head_dim)
        edge_emb = edge_emb.unsqueeze(1).expand(-1, self.num_heads, -1)

        # Attention with edge bias
        attn = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = attn + (edge_emb * k).sum(dim=-1) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=0)

        # Aggregate
        out = attn.unsqueeze(-1) * (v + edge_emb)

        # Scatter to nodes
        out_flat = out.view(-1, self.num_heads * self.head_dim)
        node_out = torch.zeros(N, out_flat.shape[1], device=x.device)
        node_out.scatter_add_(0, dst.unsqueeze(1).expand_as(out_flat), out_flat)

        return self.out_proj(node_out)
