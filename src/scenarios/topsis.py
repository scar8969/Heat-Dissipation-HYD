"""
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

Multi-criteria decision making for ranking intervention scenarios.

Steps:
1. Normalize decision matrix
2. Apply weights
3. Determine ideal and anti-ideal solutions
4. Compute separation measures
5. Compute relative closeness
6. Rank scenarios

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TOPSISRanker:
    """
    TOPSIS multi-criteria ranking of intervention scenarios.

    Ranks scenarios based on multiple objectives:
    - LST reduction
    - Cost
    - Coverage
    - Equity
    - Co-benefits
    """

    DEFAULT_WEIGHTS = {
        "lst_reduction": 0.35,
        "cost": 0.25,
        "coverage": 0.20,
        "equity": 0.10,
        "co_benefits": 0.10,
    }

    DEFAULT_BENEFITS = {
        "lst_reduction": True,  # Maximize
        "cost": False,  # Minimize
        "coverage": True,  # Maximize
        "equity": True,  # Maximize
        "co_benefits": True,  # Maximize
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        benefits: Optional[Dict[str, bool]] = None,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.benefits = benefits or self.DEFAULT_BENEFITS

    def rank(
        self,
        scenarios: List[Dict],
        criteria: List[str] = None,
    ) -> List[Dict]:
        """
        Rank scenarios using TOPSIS.

        Args:
            scenarios: List of scenario dicts with criteria values
            criteria: List of criteria names to use

        Returns:
            Ranked scenarios with TOPSIS scores
        """
        if criteria is None:
            criteria = list(self.weights.keys())

        n_scenarios = len(scenarios)
        n_criteria = len(criteria)

        # Build decision matrix
        matrix = np.zeros((n_scenarios, n_criteria))
        for i, scenario in enumerate(scenarios):
            for j, criterion in enumerate(criteria):
                matrix[i, j] = scenario.get(criterion, 0)

        # Normalize
        norm_matrix = self._normalize(matrix)

        # Apply weights
        weight_vector = np.array([self.weights.get(c, 1.0) for c in criteria])
        weight_vector = weight_vector / weight_vector.sum()
        weighted_matrix = norm_matrix * weight_vector

        # Determine ideal solutions
        ideal_best = np.zeros(n_criteria)
        ideal_worst = np.zeros(n_criteria)

        for j in range(n_criteria):
            if self.benefits.get(criteria[j], True):
                ideal_best[j] = np.max(weighted_matrix[:, j])
                ideal_worst[j] = np.min(weighted_matrix[:, j])
            else:
                ideal_best[j] = np.min(weighted_matrix[:, j])
                ideal_worst[j] = np.max(weighted_matrix[:, j])

        # Compute separation measures
        dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

        # Compute relative closeness
        closeness = dist_worst / (dist_best + dist_worst + 1e-10)

        # Rank
        rank_indices = np.argsort(-closeness)

        ranked_scenarios = []
        for rank, idx in enumerate(rank_indices, 1):
            scenario = scenarios[idx].copy()
            scenario["topsis_score"] = float(closeness[idx])
            scenario["topsis_rank"] = rank
            scenario["distance_ideal"] = float(dist_best[idx])
            scenario["distance_anti_ideal"] = float(dist_worst[idx])
            ranked_scenarios.append(scenario)

        return ranked_scenarios

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Vector normalization."""
        norm = np.sqrt(np.sum(matrix ** 2, axis=0))
        return matrix / (norm + 1e-10)

    def sensitivity_analysis(
        self,
        scenarios: List[Dict],
        criteria: List[str] = None,
        n_iterations: int = 100,
    ) -> Dict:
        """
        Sensitivity analysis on weights.

        Args:
            scenarios: List of scenarios
            criteria: Criteria names
            n_iterations: Number of weight variations

        Returns:
            Sensitivity results
        """
        if criteria is None:
            criteria = list(self.weights.keys())

        n_criteria = len(criteria)
        base_ranking = self.rank(scenarios, criteria)
        base_scores = {s["scenario_id"]: s["topsis_score"] for s in base_ranking}

        score_variations = {s["scenario_id"]: [] for s in scenarios}

        for _ in range(n_iterations):
            # Random weight perturbation
            random_weights = np.random.dirichlet(np.ones(n_criteria))
            perturbed_weights = {
                criteria[j]: float(random_weights[j])
                for j in range(n_criteria)
            }

            # Rank with perturbed weights
            ranker = TOPSISRanker(weights=perturbed_weights, benefits=self.benefits)
            perturbed_ranking = ranker.rank(scenarios, criteria)

            for s in perturbed_ranking:
                score_variations[s["scenario_id"]].append(s["topsis_score"])

        # Compute statistics
        stability = {}
        for scenario_id, scores in score_variations.items():
            stability[scenario_id] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "cv": float(np.std(scores) / (np.mean(scores) + 1e-10)),
                "rank_stability": self._compute_rank_stability(
                    score_variations, scenario_id
                ),
            }

        return {
            "base_scores": base_scores,
            "stability": stability,
        }

    def _compute_rank_stability(
        self,
        all_variations: Dict[str, List[float]],
        scenario_id: str,
    ) -> float:
        """Compute rank stability for a scenario."""
        scores = all_variations[scenario_id]
        ranks = []

        for score in scores:
            rank = 1
            for other_id, other_scores in all_variations.items():
                if other_id != scenario_id:
                    for other_score in other_scores:
                        if other_score > score:
                            rank += 1
                            break
            ranks.append(rank)

        return float(1.0 - (np.std(ranks) / (np.mean(ranks) + 1e-10)))


class WeightElicitation:
    """
    Elicitate criteria weights from stakeholders.
    """

    def __init__(self):
        pass

    def pairwise_comparison(self, criteria: List[str]) -> Dict[str, float]:
        """
        Simple pairwise comparison for weight elicitation.

        Args:
            criteria: List of criteria names

        Returns:
            Elicited weights
        """
        n = len(criteria)
        comparison_matrix = np.ones((n, n))

        # Placeholder - in practice, would ask stakeholder
        # For now, return equal weights
        weights = {c: 1.0 / n for c in criteria}

        return weights

    def rank_order_centroid(self, criteria: List[str], ranking: List[str]) -> Dict[str, float]:
        """
        Rank Order Centroid (ROC) weight generation.

        Args:
            criteria: List of criteria names
            ranking: Criteria ranked from most to least important

        Returns:
            Generated weights
        """
        n = len(criteria)
        weights = {}

        for i, criterion in enumerate(ranking):
            weight = (1.0 / n) * sum(1.0 / (j + 1) for j in range(i, n))
            weights[criterion] = weight

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights
