"""
Surface Energy Balance (SEB) diagnostics.

Validates SEB predictions against:
- Energy balance closure
- Physical consistency
- Literature values
- Station observations

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SEBDiagnostics:
    """
    Surface Energy Balance diagnostics.

    Validates that predicted fluxes satisfy physical constraints.
    """

    # Literature values for Hyderabad (W/m2)
    LITERATURE = {
        "Rn_mean": {"value": 450, "range": [350, 550]},
        "H_mean": {"value": 280, "range": [200, 380]},
        "LE_mean": {"value": 120, "range": [80, 180]},
        "G_mean": {"value": 50, "range": [20, 80]},
        "closure_error_max": {"value": 20, "unit": "W/m2"},
    }

    def __init__(self, pixel_size: float = 30.0):
        self.pixel_size = pixel_size
        self.pixel_area = pixel_size ** 2

    def diagnose(
        self,
        fluxes: Dict[str, np.ndarray],
        lst: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run full SEB diagnostics.

        Args:
            fluxes: Dict of flux maps {Rn, H, LE, G}
            lst: LST map
            mask: Optional valid mask

        Returns:
            Diagnostic results
        """
        if mask is None:
            mask = np.ones_like(lst, dtype=bool)

        results = {}

        # 1. Energy balance closure
        results["closure"] = self._check_closure(fluxes, mask)

        # 2. Physical consistency
        results["consistency"] = self._check_consistency(fluxes, lst, mask)

        # 3. Literature comparison
        results["literature"] = self._compare_literature(fluxes, mask)

        # 4. Spatial patterns
        results["spatial"] = self._check_spatial_patterns(fluxes, lst, mask)

        # 5. Overall assessment
        results["assessment"] = self._overall_assessment(results)

        return results

    def _check_closure(self, fluxes: Dict[str, np.ndarray], mask: np.ndarray) -> Dict:
        """Check energy balance closure."""
        rn = fluxes["Rn"][mask]
        h = fluxes["H"][mask]
        le = fluxes["LE"][mask]
        g = fluxes["G"][mask]

        residual = rn - h - le - g

        return {
            "mean_residual": float(np.mean(residual)),
            "std_residual": float(np.std(residual)),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "fraction_within_20wm2": float(np.mean(np.abs(residual) < 20)),
            "max_abs_residual": float(np.max(np.abs(residual))),
        }

    def _check_consistency(
        self,
        fluxes: Dict[str, np.ndarray],
        lst: np.ndarray,
        mask: np.ndarray,
    ) -> Dict:
        """Check physical consistency."""
        rn = fluxes["Rn"][mask]
        h = fluxes["H"][mask]
        le = fluxes["LE"][mask]
        g = fluxes["G"][mask]
        lst_vals = lst[mask]

        checks = {}

        # H should be positive during daytime (sensible heat from surface)
        checks["h_positive_fraction"] = float(np.mean(h > 0))

        # LE should be positive (evaporation)
        checks["le_positive_fraction"] = float(np.mean(le > 0))

        # G should be positive during daytime (heat into surface)
        checks["g_positive_fraction"] = float(np.mean(g > 0))

        # H/Rn ratio (Bowen ratio related)
        bowen = h / (le + 1e-6)
        checks["bowen_ratio_mean"] = float(np.mean(bowen))
        checks["bowen_ratio_range"] = [float(np.percentile(bowen, 5)), float(np.percentile(bowen, 95))]

        # LE/Rn ratio (evaporative fraction)
        ef = le / (rn + 1e-6)
        checks["evaporative_fraction_mean"] = float(np.mean(ef))

        return checks

    def _compare_literature(self, fluxes: Dict[str, np.ndarray], mask: np.ndarray) -> Dict:
        """Compare with literature values."""
        results = {}

        for flux_name, lit_values in self.LITERATURE.items():
            if flux_name in ["closure_error_max"]:
                continue

            base_flux = flux_name.replace("_mean", "")
            if base_flux in fluxes:
                values = fluxes[base_flux][mask]
                mean_val = float(np.mean(values))

                in_range = lit_values["range"][0] <= mean_val <= lit_values["range"][1]

                results[flux_name] = {
                    "mean": mean_val,
                    "literature_mean": lit_values["value"],
                    "literature_range": lit_values["range"],
                    "within_range": in_range,
                    "deviation": mean_val - lit_values["value"],
                }

        return results

    def _check_spatial_patterns(
        self,
        fluxes: Dict[str, np.ndarray],
        lst: np.ndarray,
        mask: np.ndarray,
    ) -> Dict:
        """Check spatial pattern consistency."""
        rn = fluxes["Rn"]
        h = fluxes["H"]
        le = fluxes["LE"]
        g = fluxes["G"]

        # Correlations between fluxes and LST
        rn_lst_corr = float(np.corrcoef(rn[mask], lst[mask])[0, 1])
        h_lst_corr = float(np.corrcoef(h[mask], lst[mask])[0, 1])
        le_lst_corr = float(np.corrcoef(le[mask], lst[mask])[0, 1])

        return {
            "rn_lst_correlation": rn_lst_corr,
            "h_lst_correlation": h_lst_corr,
            "le_lst_correlation": le_lst_corr,
            "expected_h_lst_positive": h_lst_corr > 0,
            "expected_le_lst_negative": le_lst_corr < 0,
        }

    def _overall_assessment(self, results: Dict) -> Dict:
        """Provide overall assessment."""
        issues = []
        warnings = []

        # Check closure
        if results["closure"]["fraction_within_20wm2"] < 0.8:
            issues.append("Energy balance closure error > 20 W/m2 for > 20% of pixels")

        # Check consistency
        if results["consistency"]["h_positive_fraction"] < 0.5:
            warnings.append("Less than 50% of pixels have positive H")

        if results["consistency"]["le_positive_fraction"] < 0.5:
            warnings.append("Less than 50% of pixels have positive LE")

        # Check literature
        for flux_name, comparison in results["literature"].items():
            if not comparison["within_range"]:
                warnings.append(f"{flux_name} outside literature range")

        # Check spatial patterns
        if results["spatial"]["h_lst_correlation"] < 0:
            warnings.append("H-LST correlation is negative (unexpected)")

        if results["spatial"]["le_lst_correlation"] > 0:
            warnings.append("LE-LST correlation is positive (unexpected)")

        return {
            "n_issues": len(issues),
            "n_warnings": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "quality_score": max(0, 1 - len(issues) * 0.2 - len(warnings) * 0.1),
        }


class FluxValidator:
    """
    Validate flux predictions against station observations.
    """

    def __init__(self):
        pass

    def validate(
        self,
        predicted: Dict[str, np.ndarray],
        observed: Dict[str, np.ndarray],
        timestamps: List[str],
    ) -> Dict:
        """
        Validate predictions against observations.

        Args:
            predicted: Dict of predicted flux maps
            observed: Dict of observed flux values (time series)
            timestamps: List of observation timestamps

        Returns:
            Validation metrics
        """
        results = {}

        for flux_name in ["Rn", "H", "LE", "G"]:
            if flux_name not in predicted or flux_name not in observed:
                continue

            pred_values = []
            obs_values = []

            for t in timestamps:
                if t in observed[flux_name]:
                    # Get spatial mean of prediction
                    pred_mean = np.mean(predicted[flux_name])
                    obs_val = observed[flux_name][t]

                    pred_values.append(pred_mean)
                    obs_values.append(obs_val)

            if len(pred_values) > 0:
                pred_values = np.array(pred_values)
                obs_values = np.array(obs_values)

                rmse = np.sqrt(np.mean((pred_values - obs_values) ** 2))
                mae = np.mean(np.abs(pred_values - obs_values))
                r2 = 1 - np.sum((pred_values - obs_values) ** 2) / \
                     np.sum((obs_values - obs_values.mean()) ** 2)

                results[flux_name] = {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "n_samples": len(pred_values),
                }

        return results
