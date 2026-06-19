"""
Tests for model components.

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mix_transformer import MixVisionTransformer
from src.models.cuboid_attention import CuboidAttention, CuboidAttentionBlock
from src.models.seb_head import DifferentiableSEBHead
from src.models.flux_head import FluxHead
from src.models.graph_layer import UrbanGraphLayer
from src.models.physics_loss import PhysicsInformedLoss
from src.models.segformer_pinn import SegFormerPINN


class TestMixVisionTransformer:
    """Tests for MiT encoder."""

    def test_mit_b0_output_shape(self):
        model = MixVisionTransformer(in_channels=18, variant="MiT-B0")
        x = torch.randn(1, 18, 256, 256)
        features = model(x)

        assert len(features) == 4
        assert features[0].shape == (1, 32, 64, 64)
        assert features[1].shape == (1, 64, 32, 32)
        assert features[2].shape == (1, 160, 16, 16)
        assert features[3].shape == (1, 256, 8, 8)

    def test_mit_b3_output_shape(self):
        model = MixVisionTransformer(in_channels=18, variant="MiT-B3")
        x = torch.randn(1, 18, 256, 256)
        features = model(x)

        assert len(features) == 4
        assert features[0].shape == (1, 64, 64, 64)
        assert features[3].shape == (1, 512, 8, 8)

    def test_gradient_flow(self):
        model = MixVisionTransformer(in_channels=18, variant="MiT-B0")
        x = torch.randn(1, 18, 256, 256, requires_grad=True)
        features = model(x)
        loss = features[-1].sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCuboidAttention:
    """Tests for Cuboid Attention."""

    def test_output_shape(self):
        block = CuboidAttentionBlock(dim=64, cuboid_size=(4, 4, 4))
        x = torch.randn(1, 64, 16, 16)
        temporal = torch.randn(1, 5, 64, 16, 16)

        out = block(x, temporal)
        assert out.shape == (1, 64, 16, 16)


class TestSEBHead:
    """Tests for SEB head."""

    def test_output_shapes(self):
        head = DifferentiableSEBHead(in_dim=64, hidden_dim=128)
        features = torch.randn(1, 64, 16, 16)
        met = torch.randn(1, 5, 16, 16)

        lst, params = head(features, met)

        assert lst.shape == (1, 1, 16, 16)
        assert "alpha" in params
        assert "ra" in params
        assert "beta" in params

    def test_lst_range(self):
        head = DifferentiableSEBHead(in_dim=64, hidden_dim=128)
        features = torch.randn(1, 64, 16, 16)
        # Met forcing: SW(W/m2), LW(W/m2), T_air(Celsius), RH(%), Wind(m/s)
        met = torch.zeros(1, 5, 16, 16)
        met[:, 0] = 500  # SW_down W/m2
        met[:, 1] = 400  # LW_down W/m2
        met[:, 2] = 27   # T_air ~27C = 300K
        met[:, 3] = 60   # RH %
        met[:, 4] = 2    # Wind m/s

        lst, _ = head(features, met)

        # LST should be in Celsius (roughly 15-45C)
        assert lst.mean() > 0
        assert lst.mean() < 60


class TestFluxHead:
    """Tests for Flux head."""

    def test_output_shapes(self):
        head = FluxHead(in_dim=64, hidden_dim=128)
        features = torch.randn(1, 64, 16, 16)
        met = torch.randn(1, 5, 16, 16)

        fluxes, lst = head(features, met)

        assert "Rn" in fluxes
        assert "H" in fluxes
        assert "LE" in fluxes
        assert "G" in fluxes
        assert lst.shape == (1, 1, 16, 16)

    def test_energy_balance(self):
        head = FluxHead(in_dim=64, hidden_dim=128)
        features = torch.randn(1, 64, 16, 16)
        met = torch.randn(1, 5, 16, 16)

        fluxes, _ = head(features, met, apply_physics=True)

        # Energy balance should approximately close
        residual = fluxes["Rn"] - fluxes["H"] - fluxes["LE"] - fluxes["G"]
        assert residual.abs().mean() < 50  # Allow some closure error


class TestGraphLayer:
    """Tests for UrbanGraph layer."""

    def test_output_shape(self):
        layer = UrbanGraphLayer(in_dim=64, hidden_dim=32)
        x = torch.randn(1, 64, 32, 32)

        out = layer(x, radius=2)
        assert out.shape == x.shape


class TestPhysicsLoss:
    """Tests for Physics-informed loss."""

    def test_loss_computation(self):
        loss_fn = PhysicsInformedLoss()

        predictions = {
            "LST": torch.randn(1, 1, 32, 32) + 35,
            "Rn": torch.randn(1, 1, 32, 32) * 100 + 400,
            "H": torch.randn(1, 1, 32, 32) * 50 + 200,
            "LE": torch.randn(1, 1, 32, 32) * 30 + 100,
            "G": torch.randn(1, 1, 32, 32) * 20 + 50,
        }

        targets = {"LST": torch.randn(1, 1, 32, 32) + 35}
        met = torch.randn(1, 5, 32, 32)

        losses = loss_fn(predictions, targets, met)

        assert "total" in losses
        assert "data" in losses
        assert losses["total"] > 0


class TestSegFormerPINN:
    """Tests for main hybrid model."""

    def test_forward_pass(self):
        model = SegFormerPINN(
            in_channels=18,
            variant="MiT-B0",
            use_graph=False,
            use_temporal=False,
        )

        x = torch.randn(1, 18, 128, 128)
        met = torch.randn(1, 5, 128, 128)

        output = model(x, met)

        assert "LST" in output
        assert "Rn" in output
        assert "H" in output
        assert "LE" in output
        assert "G" in output
        assert output["LST"].shape == (1, 1, 128, 128)

    def test_gradient_flow(self):
        model = SegFormerPINN(
            in_channels=18,
            variant="MiT-B0",
            use_graph=False,
            use_temporal=False,
        )

        x = torch.randn(1, 18, 128, 128, requires_grad=True)
        met = torch.randn(1, 5, 128, 128)

        output = model(x, met)
        loss = output["LST"].sum()
        loss.backward()

        assert x.grad is not None

    def test_parameter_count(self):
        model = SegFormerPINN(
            in_channels=18,
            variant="MiT-B0",
            use_graph=False,
            use_temporal=False,
        )

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 100000  # At least 100K parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
