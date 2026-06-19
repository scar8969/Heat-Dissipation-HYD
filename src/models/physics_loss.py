"""
Physics-informed loss functions for SEB-constrained training.

Combines:
- Data-driven loss (L1, L2, SSIM)
- Physics loss (energy balance, gradient, monotonicity, temporal smoothness)
- Graph smoothness loss
- Learnable gate for adaptive weighting

Based on:
- SEB-TwinTFT (EPFL 2025): Physics as structural constraint
- UrbanHeatML: Multi-task loss design

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysicsInformedLoss(nn.Module):
    """
    Composite loss function for physics-informed training.

    L = L_data + λ1*L_energy + λ2*L_gradient + λ3*L_monotonicity
        + λ4*L_temporal + λ5*L_graph + λ6*L_SEB

    Includes learnable gate for adaptive loss weighting.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # Default config
        if config is None:
            config = {}

        self.config = config

        # Loss components
        self.data_loss = DataDrivenLoss(config.get("data_weight", 1.0))
        self.energy_loss = EnergyBalanceLoss(config.get("energy_weight", 0.5))
        self.gradient_loss = GradientConsistencyLoss(config.get("gradient_weight", 0.3))
        self.monotonicity_loss = MonotonicityLoss(config.get("monotonicity_weight", 0.2))
        self.temporal_loss = TemporalSmoothnessLoss(config.get("temporal_weight", 0.1))
        self.graph_loss = GraphSmoothnessLoss(config.get("graph_weight", 0.1))

        # Learnable gate (TwinTFT-style)
        self.num_losses = 6
        self.log_sigma = nn.Parameter(torch.zeros(self.num_losses))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        met_forcing: Optional[torch.Tensor] = None,
        prev_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Args:
            predictions: Dict of predicted fluxes/LST
            targets: Dict of target fluxes/LST
            met_forcing: Meteorological forcing for energy balance
            prev_predictions: Previous timestep predictions for temporal loss

        Returns:
            Dict of total loss and component losses
        """
        # Data-driven loss
        l_data = self.data_loss(predictions, targets)

        # Physics losses (only if met_forcing provided)
        l_energy = torch.tensor(0.0, device=l_data.device)
        l_gradient = torch.tensor(0.0, device=l_data.device)
        l_monotonicity = torch.tensor(0.0, device=l_data.device)

        if met_forcing is not None:
            l_energy = self.energy_loss(predictions, met_forcing)
            l_gradient = self.gradient_loss(predictions)
            l_monotonicity = self.monotonicity_loss(predictions, met_forcing)

        # Temporal loss
        l_temporal = torch.tensor(0.0, device=l_data.device)
        if prev_predictions is not None:
            l_temporal = self.temporal_loss(predictions, prev_predictions)

        # Graph smoothness
        l_graph = self.graph_loss(predictions)

        # Collect losses
        losses = torch.stack([l_data, l_energy, l_gradient, l_monotonicity, l_temporal, l_graph])

        # Adaptive weighting (learnable gate)
        weights = torch.exp(-self.log_sigma) / torch.exp(-self.log_sigma).sum()
        total_loss = (weights * losses).sum() + self.log_sigma.sum()

        return {
            "total": total_loss,
            "data": l_data,
            "energy": l_energy,
            "gradient": l_gradient,
            "monotonicity": l_monotonicity,
            "temporal": l_temporal,
            "graph": l_graph,
            "weights": weights.detach(),
        }


class DataDrivenLoss(nn.Module):
    """Hybrid L1 + SSIM loss for robust data fitting."""

    def __init__(self, weight: float = 1.0, alpha: float = 0.84):
        super().__init__()
        self.weight = weight
        self.alpha = alpha  # Balance between L1 and SSIM

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute data-driven loss."""
        pred_lst = predictions.get("LST", predictions.get("lst"))
        target_lst = targets.get("LST", targets.get("lst"))

        if pred_lst is None or target_lst is None:
            return torch.tensor(0.0, device=pred_lst.device if pred_lst is not None else "cpu")

        # L1 loss
        l1_loss = F.l1_loss(pred_lst, target_lst)

        # SSIM loss (structural similarity)
        ssim_loss = 1 - self._compute_ssim(pred_lst, target_lst)

        # Combined
        loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

        return self.weight * loss

    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Compute Structural Similarity Index."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Create Gaussian window
        channels = x.shape[1]
        kernel = self._create_gaussian_kernel(window_size, channels, x.device)

        mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=channels)
        mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=channels)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x ** 2, kernel, padding=window_size // 2, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, kernel, padding=window_size // 2, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=channels) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        return ssim_map.mean()

    def _create_gaussian_kernel(self, window_size: int, channels: int, device: torch.device) -> torch.Tensor:
        """Create Gaussian kernel for SSIM."""
        sigma = 1.5
        x = torch.arange(window_size, device=device).float() - window_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
        return kernel_2d


class EnergyBalanceLoss(nn.Module):
    """
    Penalize energy balance non-closure.

    SEB: Rn = H + LE + G
    Loss = ||Rn - H - LE - G||^2
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        met_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy balance loss."""
        rn = predictions.get("Rn", predictions.get("rn"))
        h = predictions.get("H", predictions.get("h"))
        le = predictions.get("LE", predictions.get("le"))
        g = predictions.get("G", predictions.get("g"))

        if rn is None or h is None or le is None or g is None:
            return torch.tensor(0.0, device=met_forcing.device)

        # Energy balance residual
        residual = rn - h - le - g

        # Loss
        loss = F.mse_loss(residual, torch.zeros_like(residual))

        return self.weight * loss


class GradientConsistencyLoss(nn.Module):
    """
    Penalize unrealistic temperature gradients.

    Temperature differences between neighboring pixels should
    be consistent with surface properties.
    """

    def __init__(self, weight: float = 1.0, threshold: float = 5.0):
        super().__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute gradient consistency loss."""
        lst = predictions.get("LST", predictions.get("lst"))

        if lst is None:
            return torch.tensor(0.0, device="cpu")

        # Compute gradients
        grad_y = lst[:, :, 1:, :] - lst[:, :, :-1, :]
        grad_x = lst[:, :, :, 1:] - lst[:, :, :, :-1]

        # Penalize gradients exceeding threshold
        loss_y = F.relu(grad_y.abs() - self.threshold).mean()
        loss_x = F.relu(grad_x.abs() - self.threshold).mean()

        return self.weight * (loss_x + loss_y) / 2


class MonotonicityLoss(nn.Module):
    """
    Enforce monotonic relationships between LST and drivers.

    Known monotonic relationships:
    - LST increases with NDBI (built-up index)
    - LST decreases with NDVI (vegetation index)
    - LST decreases with MNDWI (water index)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        met_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """Compute monotonicity loss."""
        lst = predictions.get("LST", predictions.get("lst"))

        if lst is None:
            return torch.tensor(0.0, device="cpu")

        # LST should increase with SW radiation (first channel of met_forcing)
        sw = met_forcing[:, 0:1]
        lst_diff = lst[:, :, 1:, :] - lst[:, :, :-1, :]
        sw_diff = sw[:, :, 1:, :] - sw[:, :, :-1, :]

        # Penalty when relationship is violated
        violation = F.relu(-(lst_diff * sw_diff))

        return self.weight * violation.mean()


class TemporalSmoothnessLoss(nn.Module):
    """
    Penalize unrealistic temporal changes.

    LST should be temporally smooth (small changes between timesteps).
    """

    def __init__(self, weight: float = 1.0, max_change: float = 2.0):
        super().__init__()
        self.weight = weight
        self.max_change = max_change

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        prev_predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute temporal smoothness loss."""
        lst = predictions.get("LST", predictions.get("lst"))
        prev_lst = prev_predictions.get("LST", prev_predictions.get("lst"))

        if lst is None or prev_lst is None:
            return torch.tensor(0.0, device=lst.device if lst is not None else "cpu")

        # Temporal difference
        temporal_diff = lst - prev_lst

        # Penalize changes exceeding threshold
        loss = F.relu(temporal_diff.abs() - self.max_change).mean()

        return self.weight * loss


class GraphSmoothnessLoss(nn.Module):
    """
    Penalize unrealistic spatial variations.

    LST should be spatially smooth (small variations between neighbors).
    """

    def __init__(self, weight: float = 1.0, threshold: float = 3.0):
        super().__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute graph smoothness loss."""
        lst = predictions.get("LST", predictions.get("lst"))

        if lst is None:
            return torch.tensor(0.0, device="cpu")

        # Horizontal and vertical differences
        diff_h = lst[:, :, :, 1:] - lst[:, :, :, :-1]
        diff_v = lst[:, :, 1:, :] - lst[:, :, :-1, :]

        # Penalize large differences
        loss_h = F.relu(diff_h.abs() - self.threshold).mean()
        loss_v = F.relu(diff_v.abs() - self.threshold).mean()

        return self.weight * (loss_h + loss_v) / 2
