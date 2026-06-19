"""
Multi-objective optimization for intervention placement.

Uses NSGA-III to optimize:
- Minimize: Mean LST, Cost
- Maximize: Coverage, Equity

Constraints:
- Budget: ₹10/50/100 Cr
- Spatial: Zone aggregation (500 zones)
- Temporal: Phased implementation

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    budget_inr: float = 1e9  # 100 Cr
    n_zones: int = 500
    n_objectives: int = 4
    n_constraints: int = 3
    population_size: int = 100
    n_generations: int = 200
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8


class NSGA3Optimizer:
    """
    NSGA-III multi-objective optimizer for intervention placement.

    Optimizes intervention allocation across zones
    to minimize LST while respecting budget and equity constraints.
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        zone_stats: List[Dict],
        interventions: List[str],
        intervention_costs: Dict[str, float],
        intervention_effects: Dict[str, Dict[str, float]],
    ) -> Dict:
        """
        Run NSGA-III optimization.

        Args:
            zone_stats: List of zone statistics
            interventions: List of intervention keys
            intervention_costs: Cost per m2 for each intervention
            intervention_effects: Expected LST change for each intervention

        Returns:
            Optimization results with Pareto front
        """
        n_zones = len(zone_stats)
        n_interventions = len(interventions)

        # Initialize population
        population = self._initialize_population(n_zones, n_interventions)

        # Evaluate initial population
        fitness = self._evaluate_population(
            population, zone_stats, interventions, intervention_costs, intervention_effects
        )

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Selection
            parents = self._tournament_selection(population, fitness)

            # Crossover and mutation
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring, n_interventions)

            # Evaluate offspring
            offspring_fitness = self._evaluate_population(
                offspring, zone_stats, interventions, intervention_costs, intervention_effects
            )

            # Environmental selection
            population, fitness = self._environmental_selection(
                population, fitness, offspring, offspring_fitness
            )

            if gen % 50 == 0:
                logger.info(f"Generation {gen}: Best fitness = {np.min(fitness[:, 0]):.4f}")

        # Extract Pareto front
        pareto_front = self._extract_pareto_front(population, fitness)

        return {
            "pareto_solutions": pareto_front,
            "population": population,
            "fitness": fitness,
            "n_generations": self.config.n_generations,
        }

    def _initialize_population(self, n_zones: int, n_interventions: int) -> np.ndarray:
        """Initialize random population."""
        population = np.zeros((self.config.population_size, n_zones, n_interventions))

        for i in range(self.config.population_size):
            for z in range(n_zones):
                # Random intervention assignment (0 = no intervention)
                intervention_idx = np.random.randint(0, n_interventions + 1)
                if intervention_idx > 0:
                    population[i, z, intervention_idx - 1] = 1.0

        return population

    def _evaluate_population(
        self,
        population: np.ndarray,
        zone_stats: List[Dict],
        interventions: List[str],
        intervention_costs: Dict[str, float],
        intervention_effects: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """Evaluate fitness for all individuals."""
        n_individuals = population.shape[0]
        fitness = np.zeros((n_individuals, self.config.n_objectives))

        for i in range(n_individuals):
            fitness[i] = self._evaluate_individual(
                population[i], zone_stats, interventions, intervention_costs, intervention_effects
            )

        return fitness

    def _evaluate_individual(
        self,
        solution: np.ndarray,
        zone_stats: List[Dict],
        interventions: List[str],
        intervention_costs: Dict[str, float],
        intervention_effects: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """Evaluate fitness for a single individual."""
        n_zones = len(zone_stats)
        n_interventions = len(interventions)

        total_cost = 0.0
        total_lst_change = 0.0
        total_area = 0.0
        zones_treated = 0

        for z in range(n_zones):
            zone = zone_stats[z]
            zone_area_m2 = zone["area_ha"] * 10000

            for j in range(n_interventions):
                if solution[z, j] > 0.5:
                    intervention_key = interventions[j]
                    cost = zone_area_m2 * intervention_costs[intervention_key]
                    lst_change = intervention_effects[intervention_key].get("lst_change", -1.0)

                    total_cost += cost
                    total_lst_change += lst_change * zone_area_m2
                    total_area += zone_area_m2
                    zones_treated += 1

        # Objectives (to minimize)
        mean_lst_change = total_lst_change / (total_area + 1e-6)
        normalized_cost = total_cost / self.config.budget_inr
        coverage = total_area / (sum(z["area_ha"] for z in zone_stats) * 10000 + 1e-6)
        equity = zones_treated / (n_zones + 1e-6)

        return np.array([
            -mean_lst_change,  # Maximize cooling (minimize negative)
            normalized_cost,  # Minimize cost
            -coverage,  # Maximize coverage
            -equity,  # Maximize equity
        ])

    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        tournament_size: int = 3,
    ) -> np.ndarray:
        """Tournament selection."""
        n_individuals = population.shape[0]
        selected = np.zeros_like(population)

        for i in range(n_individuals):
            candidates = np.random.choice(n_individuals, tournament_size, replace=False)
            candidate_fitness = fitness[candidates]

            # Non-dominated sorting
            ranks = self._non_dominated_sort(candidate_fitness)
            best = candidates[np.argmin(ranks)]

            selected[i] = population[best]

        return selected

    def _non_dominated_sort(self, fitness: np.ndarray) -> np.ndarray:
        """Simple non-dominated sorting."""
        n = fitness.shape[0]
        ranks = np.zeros(n, dtype=int)

        for i in range(n):
            dominated_count = 0
            for j in range(n):
                if i != j:
                    if np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                        dominated_count += 1
            ranks[i] = dominated_count

        return ranks

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        n_individuals = parents.shape[0]
        offspring = np.zeros_like(parents)

        for i in range(0, n_individuals, 2):
            if i + 1 >= n_individuals:
                offspring[i] = parents[i]
                break

            if np.random.random() < self.config.crossover_rate:
                point = np.random.randint(0, parents.shape[1])
                offspring[i, :point] = parents[i, :point]
                offspring[i, point:] = parents[i + 1, point:]
                offspring[i + 1, :point] = parents[i + 1, :point]
                offspring[i + 1, point:] = parents[i, point:]
            else:
                offspring[i] = parents[i]
                offspring[i + 1] = parents[i + 1]

        return offspring

    def _mutate(self, population: np.ndarray, n_interventions: int) -> np.ndarray:
        """Random mutation."""
        mutated = population.copy()

        for i in range(population.shape[0]):
            for z in range(population.shape[1]):
                if np.random.random() < self.config.mutation_rate:
                    # Randomly change intervention
                    mutated[i, z] = 0
                    intervention_idx = np.random.randint(0, n_interventions + 1)
                    if intervention_idx > 0:
                        mutated[i, z, intervention_idx - 1] = 1.0

        return mutated

    def _environmental_selection(
        self,
        pop1: np.ndarray,
        fit1: np.ndarray,
        pop2: np.ndarray,
        fit2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Environmental selection (mu + lambda)."""
        combined_pop = np.concatenate([pop1, pop2], axis=0)
        combined_fit = np.concatenate([fit1, fit2], axis=0)

        # Non-dominated sorting
        ranks = self._non_dominated_sort(combined_fit)

        # Select best
        selected_idx = np.argsort(ranks)[:pop1.shape[0]]

        return combined_pop[selected_idx], combined_fit[selected_idx]

    def _extract_pareto_front(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> List[Dict]:
        """Extract Pareto-optimal solutions."""
        ranks = self._non_dominated_sort(fitness)
        pareto_idx = np.where(ranks == 0)[0]

        solutions = []
        for idx in pareto_idx:
            solutions.append({
                "solution": population[idx],
                "fitness": fitness[idx],
            })

        return solutions
