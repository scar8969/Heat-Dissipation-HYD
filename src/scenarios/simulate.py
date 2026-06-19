"""
Scenario simulation for urban heat interventions.

Simulates the effect of interventions on LST:
- Pixel-level simulation using SEB model
- Zone-level aggregation
- Uncertainty quantification
- Multi-scenario comparison

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

from .interventions import InterventionType, INTERVENTIONS, InterventionPlacer

logger = logging.getLogger(__name__)


class ScenarioSimulator:
    """
    Simulate intervention effects on LST.

    Uses the trained model to predict LST under different
    intervention scenarios.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
        pixel_size: float = 30.0,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self.pixel_size = pixel_size
        self.placer = InterventionPlacer()

    @torch.no_grad()
    def simulate_single(
        self,
        features: np.ndarray,
        intervention_key: str,
        coverage_fraction: float = 1.0,
        met_forcing: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate effect of a single intervention.

        Args:
            features: (C, H, W) current features
            intervention_key: Key in INTERVENTIONS dict
            coverage_fraction: Fraction of applicable area to cover
            met_forcing: (5, H, W) meteorological forcing

        Returns:
            Dict with predicted LST change and other outputs
        """
        intervention = INTERVENTIONS[intervention_key]

        # Get applicability
        applicability = self.placer.compute_applicability(
            lulc=features[17].astype(int),  # LULC channel
            building_density=features[14],
            existing_albedo=features[5],
            pixel_size=self.pixel_size,
        )

        # Apply intervention
        features_modified = self.placer.apply_intervention(
            features, intervention, applicability[intervention_key], coverage_fraction
        )

        # Predict baseline LST
        baseline = self._predict(features, met_forcing)

        # Predict modified LST
        modified = self._predict(features_modified, met_forcing)

        # Compute difference
        lst_change = modified["LST"] - baseline["LST"]

        return {
            "baseline_lst": baseline["LST"],
            "modified_lst": modified["LST"],
            "lst_change": lst_change,
            "mean_lst_change": float(np.mean(lst_change)),
            "max_lst_change": float(np.min(lst_change)),  # Negative = cooling
            "applicability": applicability[intervention_key],
            "coverage_area_ha": float(np.sum(applicability[intervention_key]) * self.pixel_size**2 / 10000),
        }

    def simulate_combined(
        self,
        features: np.ndarray,
        intervention_plan: List[Dict],
        met_forcing: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate combined intervention plan.

        Args:
            features: (C, H, W) current features
            intervention_plan: List of {intervention_key, coverage_fraction}
            met_forcing: Meteorological forcing

        Returns:
            Combined simulation results
        """
        features_modified = features.copy()

        total_cost = 0.0
        applied_interventions = []

        for item in intervention_plan:
            key = item["intervention_key"]
            coverage = item.get("coverage_fraction", 1.0)

            intervention = INTERVENTIONS[key]

            # Get applicability
            applicability = self.placer.compute_applicability(
                lulc=features_modified[17].astype(int),
                building_density=features_modified[14],
                existing_albedo=features_modified[5],
                pixel_size=self.pixel_size,
            )

            # Apply intervention
            features_modified = self.placer.apply_intervention(
                features_modified, intervention, applicability[key], coverage
            )

            # Compute cost
            area_m2 = np.sum(applicability[key] > 0.5) * self.pixel_size**2
            cost = area_m2 * coverage * intervention.cost_per_m2
            total_cost += cost

            applied_interventions.append({
                "intervention": key,
                "coverage_fraction": coverage,
                "area_ha": float(area_m2 * coverage / 10000),
                "cost_inr": float(cost),
            })

        # Predict baseline and modified
        baseline = self._predict(features, met_forcing)
        modified = self._predict(features_modified, met_forcing)

        lst_change = modified["LST"] - baseline["LST"]

        return {
            "baseline_lst": baseline["LST"],
            "modified_lst": modified["LST"],
            "lst_change": lst_change,
            "mean_lst_change": float(np.mean(lst_change)),
            "total_cost_inr": total_cost,
            "applied_interventions": applied_interventions,
        }

    def simulate_budget_scenarios(
        self,
        features: np.ndarray,
        budget_inr: float,
        met_forcing: Optional[np.ndarray] = None,
        n_scenarios: int = 100,
    ) -> List[Dict]:
        """
        Generate and simulate random budget-constrained scenarios.

        Args:
            features: (C, H, W) current features
            budget_inr: Total budget in INR
            met_forcing: Meteorological forcing
            n_scenarios: Number of random scenarios

        Returns:
            List of scenario results
        """
        scenarios = []

        for i in range(n_scenarios):
            # Random intervention mix
            plan = self._generate_random_plan(budget_inr)

            # Simulate
            result = self.simulate_combined(features, plan, met_forcing)
            result["scenario_id"] = i
            result["plan"] = plan

            scenarios.append(result)

        return scenarios

    def _generate_random_plan(self, budget_inr: float) -> List[Dict]:
        """Generate a random intervention plan within budget."""
        plan = []
        remaining_budget = budget_inr

        # Randomly select interventions
        intervention_keys = list(INTERVENTIONS.keys())
        np.random.shuffle(intervention_keys)

        for key in intervention_keys:
            if remaining_budget <= 0:
                break

            intervention = INTERVENTIONS[key]

            # Random coverage (10-100%)
            coverage = np.random.uniform(0.1, 1.0)

            # Estimate cost (rough)
            estimated_area_m2 = 100000  # Assume 1 ha
            estimated_cost = estimated_area_m2 * coverage * intervention.cost_per_m2

            if estimated_cost <= remaining_budget:
                plan.append({
                    "intervention_key": key,
                    "coverage_fraction": coverage,
                })
                remaining_budget -= estimated_cost

        return plan

    def _predict(
        self,
        features: np.ndarray,
        met_forcing: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run model prediction."""
        x = torch.from_numpy(features).unsqueeze(0).float().to(self.device)

        if met_forcing is not None:
            m = torch.from_numpy(met_forcing).unsqueeze(0).float().to(self.device)
        else:
            m = None

        pred = self.model(x, m)

        return {k: v.cpu().numpy().squeeze() for k, v in pred.items() if isinstance(v, torch.Tensor)}


class UncertaintyQuantifier:
    """
    Quantify uncertainty in intervention simulations.

    Uses Monte Carlo dropout for uncertainty estimation.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")

    def enable_dropout(self):
        """Enable dropout for MC sampling."""
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    @torch.no_grad()
    def mc_dropout_predict(
        self,
        features: np.ndarray,
        met_forcing: Optional[np.ndarray] = None,
        n_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Monte Carlo dropout prediction.

        Args:
            features: (C, H, W) input features
            met_forcing: Meteorological forcing
            n_samples: Number of MC samples

        Returns:
            Dict with mean and std of predictions
        """
        self.model.eval()
        self.enable_dropout()

        x = torch.from_numpy(features).unsqueeze(0).float().to(self.device)
        if met_forcing is not None:
            m = torch.from_numpy(met_forcing).unsqueeze(0).float().to(self.device)
        else:
            m = None

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(x, m)
            predictions.append(pred["LST"].cpu().numpy().squeeze())

        predictions = np.stack(predictions)

        return {
            "mean": np.mean(predictions, axis=0),
            "std": np.std(predictions, axis=0),
            "p5": np.percentile(predictions, 5, axis=0),
            "p95": np.percentile(predictions, 95, axis=0),
        }
