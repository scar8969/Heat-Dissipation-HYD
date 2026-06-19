"""
Driver analysis for urban heat.

Quantifies the contribution of each driver to LST:
- SHAP-based analysis
- Permutation importance
- Sensitivity indices
- Partial dependence

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DriverAnalyzer:
    """
    Analyze drivers of urban heat.

    Quantifies contribution of each feature to LST predictions.
    """

    FEATURE_NAMES = [
        "LST", "NDVI", "NDBI", "MNDWI", "NDBaI", "ALBEDO", "EMISSIVITY",
        "FVC", "LAI", "ELEVATION", "SLOPE", "ASPECT", "SVF", "Z0",
        "BUILDING_DENSITY", "BUILDING_HEIGHT", "ROAD_DENSITY", "LCZ",
    ]

    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def permutation_importance(
        self,
        dataloader,
        n_repeats: int = 10,
    ) -> Dict[str, float]:
        """
        Compute permutation importance for each feature.

        Args:
            dataloader: DataLoader for evaluation
            n_repeats: Number of permutation repeats

        Returns:
            Dict of feature importance scores
        """
        self.model.eval()

        # Baseline performance
        baseline_loss = self._evaluate(dataloader)

        importance = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            losses = []
            for _ in range(n_repeats):
                loss = self._evaluate_permuted(dataloader, feature_idx=i)
                losses.append(loss)

            importance[name] = float(np.mean(losses) - baseline_loss)

        # Normalize to sum to 1
        total = sum(importance.values()) + 1e-6
        importance = {k: v / total for k, v in importance.items()}

        return importance

    def _evaluate(self, dataloader) -> float:
        """Evaluate model on dataset."""
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            lst = batch["LST"].to(self.device)

            predictions = self.model(features)
            loss = torch.nn.functional.mse_loss(predictions["LST"], lst)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _evaluate_permuted(self, dataloader, feature_idx: int) -> float:
        """Evaluate model with permuted feature."""
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            features = batch["features"].clone()
            lst = batch["LST"].to(self.device)

            # Permute feature
            perm = torch.randperm(features.shape[2])
            features[:, feature_idx] = features[perm, feature_idx]
            features = features.to(self.device)

            predictions = self.model(features)
            loss = torch.nn.functional.mse_loss(predictions["LST"], lst)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def gradient_sensitivity(
        self,
        dataloader,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compute gradient-based sensitivity.

        Measures average gradient magnitude of LST w.r.t. each feature.

        Args:
            dataloader: DataLoader
            n_samples: Number of samples to use

        Returns:
            Dict of sensitivity scores
        """
        self.model.eval()
        sensitivities = {name: [] for name in self.FEATURE_NAMES}

        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break

            features = batch["features"].to(self.device).requires_grad_(True)

            predictions = self.model(features)
            lst = predictions["LST"]

            # Compute gradients
            lst.sum().backward(retain_graph=True)

            if features.grad is not None:
                for j, name in enumerate(self.FEATURE_NAMES):
                    grad = features.grad[:, j].abs().mean().item()
                    sensitivities[name].append(grad)

        # Average across samples
        sensitivities = {
            name: float(np.mean(vals)) if vals else 0.0
            for name, vals in sensitivities.items()
        }

        # Normalize
        total = sum(sensitivities.values()) + 1e-6
        sensitivities = {k: v / total for k, v in sensitivities.items()}

        return sensitivities

    def analyze_hotspot_drivers(
        self,
        lst: np.ndarray,
        features: np.ndarray,
        hotspot_mask: np.ndarray,
        background_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compare feature distributions in hotspots vs background.

        Args:
            lst: LST map
            features: (C, H, W) feature stack
            hotspot_mask: Binary hotspot mask
            background_mask: Optional background mask

        Returns:
            Dict of driver importance scores
        """
        if background_mask is None:
            background_mask = ~hotspot_mask

        n_features = features.shape[0]
        scores = {}

        for i, name in enumerate(self.FEATURE_NAMES):
            if i >= n_features:
                break

            hot_vals = features[i][hotspot_mask]
            bg_vals = features[i][background_mask]

            if len(hot_vals) > 0 and len(bg_vals) > 0:
                # KS statistic
                from scipy.stats import ks_2samp
                ks_stat, _ = ks_2samp(hot_vals, bg_vals)
                scores[name] = float(ks_stat)

        # Normalize
        total = sum(scores.values()) + 1e-6
        scores = {k: v / total for k, v in scores.items()}

        return scores


class InteractionAnalyzer:
    """
    Analyze interactions between drivers.
    """

    def __init__(self):
        pass

    def compute_interactions(
        self,
        features: np.ndarray,
        lst: np.ndarray,
        feature_names: List[str] = None,
        max_pairs: int = 20,
    ) -> List[Dict]:
        """
        Compute pairwise feature interactions with LST.

        Args:
            features: (C, H, W) feature stack
            lst: (H, W) LST map
            feature_names: List of feature names
            max_pairs: Maximum number of pairs to analyze

        Returns:
            List of interaction results
        """
        if feature_names is None:
            feature_names = DriverAnalyzer.FEATURE_NAMES

        n_features = min(features.shape[0], len(feature_names))
        interactions = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                f1 = features[i].ravel()
                f2 = features[j].ravel()
                lst_flat = lst.ravel()

                # Compute interaction strength
                interaction = self._compute_interaction(f1, f2, lst_flat)

                interactions.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "interaction_strength": float(interaction),
                })

        # Sort by strength
        interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)

        return interactions[:max_pairs]

    def _compute_interaction(self, f1: np.ndarray, f2: np.ndarray, y: np.ndarray) -> float:
        """
        Compute interaction strength between two features.
        Uses correlation of product with target.
        """
        # Correlation of product with target
        product = f1 * f2
        corr = np.corrcoef(product, y)[0, 1]

        # Correlation of individual features
        corr1 = np.corrcoef(f1, y)[0, 1]
        corr2 = np.corrcoef(f2, y)[0, 1]

        # Interaction = combined - sum of individual
        interaction = abs(corr) - (abs(corr1) + abs(corr2)) / 2

        return max(0, interaction)
