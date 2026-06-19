"""
Sensitivity analysis for urban heat drivers.

Computes sensitivity indices:
- Morris screening (elementary effects)
- Sobol indices (variance-based)
- Regional sensitivity analysis

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Sensitivity analysis for urban heat drivers.

    Quantifies how variations in input features affect LST predictions.
    """

    FEATURE_NAMES = [
        "NDVI", "NDBI", "MNDWI", "NDBaI", "ALBEDO", "EMISSIVITY",
        "FVC", "LAI", "ELEVATION", "SVF", "Z0", "BUILDING_DENSITY",
        "BUILDING_HEIGHT", "ROAD_DENSITY",
    ]

    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def morris_screening(
        self,
        baseline_features: np.ndarray,
        n_trajectories: int = 10,
        delta: float = 0.05,
    ) -> Dict[str, float]:
        """
        Morris screening for elementary effects.

        Args:
            baseline_features: (1, C, H, W) baseline feature values
            n_trajectories: Number of Morris trajectories
            delta: Perturbation size (fraction of range)

        Returns:
            Dict of mean absolute elementary effects per feature
        """
        self.model.eval()
        n_features = baseline_features.shape[1]

        elementary_effects = {name: [] for name in self.FEATURE_NAMES[:n_features]}

        for _ in range(n_trajectories):
            # Random permutation of feature order
            perm = np.random.permutation(n_features)

            # Start from random point
            x = baseline_features.copy()

            # Evaluate at baseline
            base_lst = self._predict(x)

            for idx in perm:
                # Perturb feature
                x_perturbed = x.copy()
                feature_range = np.max(x[:, idx]) - np.min(x[:, idx])
                x_perturbed[:, idx] += delta * feature_range

                # Evaluate
                pert_lst = self._predict(x_perturbed)

                # Elementary effect
                ee = (pert_lst - base_lst) / (delta * feature_range + 1e-10)

                name = self.FEATURE_NAMES[idx]
                elementary_effects[name].append(abs(float(ee)))

                # Update for next step
                x = x_perturbed
                base_lst = pert_lst

        # Compute mean absolute elementary effect
        mu_star = {
            name: float(np.mean(vals)) if vals else 0.0
            for name, vals in elementary_effects.items()
        }

        # Normalize
        total = sum(mu_star.values()) + 1e-6
        mu_star = {k: v / total for k, v in mu_star.items()}

        return mu_star

    def sobol_indices(
        self,
        baseline_features: np.ndarray,
        n_samples: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Sobol indices (first-order and total).

        Uses Saltelli sampling scheme.

        Args:
            baseline_features: (1, C, H, W) baseline values
            n_samples: Number of samples

        Returns:
            Dict with first-order (S1) and total (ST) indices
        """
        self.model.eval()
        n_features = baseline_features.shape[1]

        # Generate Saltelli samples
        A, B = self._saltelli_sampling(baseline_features, n_samples)

        # Evaluate
        f_A = self._predict_batch(A)
        f_B = self._predict_batch(B)

        # First-order indices
        S1 = {}
        ST = {}

        for i, name in enumerate(self.FEATURE_NAMES[:n_features]):
            # Create ABi (A with column i from B)
            ABi = A.copy()
            ABi[:, i] = B[:, i]

            f_ABi = self._predict_batch(ABi)

            # First-order
            var_f = np.var(f_A)
            S1[name] = float(np.mean(f_B * (f_ABi - f_A)) / (var_f + 1e-10))

            # Total effect
            ST[name] = float(0.5 * np.mean((f_A - f_ABi) ** 2) / (var_f + 1e-10))

        return {"S1": S1, "ST": ST}

    def _saltelli_sampling(
        self,
        baseline: np.ndarray,
        n_samples: int,
    ) -> tuple:
        """Generate Saltelli sampling matrices."""
        n_features = baseline.shape[1]

        # Base samples
        A = baseline.repeat(n_samples, axis=0)
        B = baseline.repeat(n_samples, axis=0)

        # Add random perturbations
        for i in range(n_features):
            feature_range = np.max(baseline[:, i]) - np.min(baseline[:, i])
            A[:, i] += np.random.uniform(-0.5, 0.5, n_samples) * feature_range
            B[:, i] += np.random.uniform(-0.5, 0.5, n_samples) * feature_range

        return A, B

    def _predict(self, features: np.ndarray) -> float:
        """Predict LST for single sample."""
        x = torch.from_numpy(features).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        return float(pred["LST"].mean().cpu())

    def _predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict LST for batch of samples."""
        x = torch.from_numpy(features).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        return pred["LST"].cpu().numpy().ravel()


class RegionalSensitivity:
    """
    Regional sensitivity analysis.

    Divides the feature space into regions and analyzes
    sensitivity within each region.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        features: np.ndarray,
        lst: np.ndarray,
        feature_names: List[str] = None,
        n_regions: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        Regional sensitivity analysis.

        Args:
            features: (C, H, W) feature stack
            lst: (H, W) LST map
            feature_names: Feature names
            n_regions: Number of regions per feature

        Returns:
            Regional sensitivity results
        """
        if feature_names is None:
            feature_names = SensitivityAnalyzer.FEATURE_NAMES

        results = {}
        n_features = min(features.shape[0], len(feature_names))

        for i in range(n_features):
            feat = features[i].ravel()
            feat_name = feature_names[i]

            # Create regions (quintiles)
            percentiles = np.linspace(0, 100, n_regions + 1)
            thresholds = np.percentile(feat, percentiles)

            region_stats = []
            for r in range(n_regions):
                mask = (feat >= thresholds[r]) & (feat < thresholds[r + 1])
                if np.sum(mask) > 0:
                    region_lst = lst.ravel()[mask]
                    region_stats.append({
                        "region": r + 1,
                        "feature_range": [float(thresholds[r]), float(thresholds[r + 1])],
                        "mean_lst": float(np.mean(region_lst)),
                        "std_lst": float(np.std(region_lst)),
                        "count": int(np.sum(mask)),
                    })

            results[feat_name] = region_stats

        return results
