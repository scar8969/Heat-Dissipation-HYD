"""
Urban heat mitigation interventions.

Defines intervention types with physical parameters:
- Cool roofs (paint, membrane)
- Green roofs (extensive, intensive)
- Street trees
- Urban parks
- Cool pavements
- Permeable pavements
- Water bodies

Each intervention has:
- Albedo change (Δα)
- Evaporative efficiency change (Δβ)
- Sky view factor change (ΔSVF)
- Implementation cost (INR/m2)
- Lifetime (years)
- Maintenance cost (% of initial)

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class InterventionType:
    """Definition of an intervention type."""
    name: str
    description: str
    delta_albedo: float  # Change in albedo
    delta_beta: float  # Change in evaporative efficiency
    delta_svf: float  # Change in sky view factor
    cost_per_m2: float  # INR/m2
    lifetime_years: int  # Lifetime in years
    maintenance_fraction: float  # Annual maintenance as fraction of initial cost
    spatial_resolution: float  # Minimum spatial scale (m)
    applicable_lulc: List[int] = field(default_factory=list)  # Applicable LULC codes

    def total_cost_40yr(self) -> float:
        """Compute total cost over 40 years."""
        return self.cost_per_m2 + self.maintenance_fraction * self.cost_per_m2 * 40

    def annual_cost(self) -> float:
        """Compute annualized cost."""
        return self.cost_per_m2 / self.lifetime_years + self.maintenance_fraction * self.cost_per_m2


# Define intervention types
INTERVENTIONS = {
    "cool_roof_paint": InterventionType(
        name="Cool Roof Paint",
        description="Reflective coating applied to existing roofs",
        delta_albedo=0.30,
        delta_beta=0.0,
        delta_svf=0.0,
        cost_per_m2=300,
        lifetime_years=5,
        maintenance_fraction=0.05,
        spatial_resolution=1.0,
        applicable_lulc=[2, 3],  # Built-up areas
    ),
    "cool_roof_membrane": InterventionType(
        name="Cool Roof Membrane",
        description="Single-ply reflective membrane",
        delta_albedo=0.40,
        delta_beta=0.0,
        delta_svf=0.0,
        cost_per_m2=1200,
        lifetime_years=15,
        maintenance_fraction=0.02,
        spatial_resolution=1.0,
        applicable_lulc=[2, 3],
    ),
    "green_roof_extensive": InterventionType(
        name="Extensive Green Roof",
        description="Lightweight vegetation on shallow substrate (6-15cm)",
        delta_albedo=0.10,
        delta_beta=0.40,
        delta_svf=-0.05,
        cost_per_m2=4500,
        lifetime_years=40,
        maintenance_fraction=0.01,
        spatial_resolution=100.0,
        applicable_lulc=[2, 3],
    ),
    "green_roof_intensive": InterventionType(
        name="Intensive Green Roof",
        description="Deep substrate with diverse vegetation (>15cm)",
        delta_albedo=0.15,
        delta_beta=0.60,
        delta_svf=-0.10,
        cost_per_m2=9000,
        lifetime_years=40,
        maintenance_fraction=0.02,
        spatial_resolution=100.0,
        applicable_lulc=[2, 3],
    ),
    "street_trees": InterventionType(
        name="Street Trees",
        description="Tree planting along roads and sidewalks",
        delta_albedo=0.05,
        delta_beta=0.30,
        delta_svf=-0.15,
        cost_per_m2=500,
        lifetime_years=30,
        maintenance_fraction=0.03,
        spatial_resolution=30.0,
        applicable_lulc=[2, 4, 5],  # Built-up, roads, open space
    ),
    "urban_park": InterventionType(
        name="Urban Park",
        description="Conversion of built-up area to parkland",
        delta_albedo=0.10,
        delta_beta=0.50,
        delta_svf=0.10,
        cost_per_m2=2000,
        lifetime_years=50,
        maintenance_fraction=0.02,
        spatial_resolution=1000.0,
        applicable_lulc=[2, 3],
    ),
    "cool_pavement": InterventionType(
        name="Cool Pavement",
        description="Reflective or permeable pavement coating",
        delta_albedo=0.25,
        delta_beta=0.10,
        delta_svf=0.0,
        cost_per_m2=800,
        lifetime_years=10,
        maintenance_fraction=0.03,
        spatial_resolution=1.0,
        applicable_lulc=[4],  # Roads
    ),
    "permeable_pavement": InterventionType(
        name="Permeable Pavement",
        description="Porous pavement allowing water infiltration",
        delta_albedo=0.10,
        delta_beta=0.20,
        delta_svf=0.0,
        cost_per_m2=1500,
        lifetime_years=15,
        maintenance_fraction=0.02,
        spatial_resolution=1.0,
        applicable_lulc=[4],
    ),
    "water_body": InterventionType(
        name="Water Body",
        description="Artificial lake or pond",
        delta_albedo=-0.10,
        delta_beta=0.80,
        delta_svf=0.20,
        cost_per_m2=5000,
        lifetime_years=50,
        maintenance_fraction=0.01,
        spatial_resolution=10000.0,
        applicable_lulc=[5, 6],  # Open land, vacant
    ),
}


class InterventionPlacer:
    """
    Place interventions on the landscape.

    Determines where each intervention can be applied
    based on:
    - LULC type
    - Existing features
    - Budget constraints
    - Spatial constraints
    """

    def __init__(self, interventions: Dict[str, InterventionType] = None):
        self.interventions = interventions or INTERVENTIONS

    def compute_applicability(
        self,
        lulc: np.ndarray,
        building_density: np.ndarray,
        existing_albedo: np.ndarray,
        pixel_size: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute applicability maps for each intervention.

        Args:
            lulc: (H, W) land use/land cover
            building_density: (H, W) building density [0, 1]
            existing_albedo: (H, W) current albedo
            pixel_size: Pixel size in meters

        Returns:
            Dict of applicability maps (0 or 1)
        """
        applicability = {}

        for key, intervention in self.interventions.items():
            mask = np.zeros_like(lulc, dtype=np.float32)

            # Check LULC applicability
            for lulc_code in intervention.applicable_lulc:
                mask[lulc == lulc_code] = 1.0

            # Additional constraints
            if "roof" in key.lower():
                # Only on buildings
                mask *= building_density

            if "pavement" in key.lower() or "road" in key.lower():
                # Only on roads (LULC=4)
                mask *= (lulc == 4).astype(np.float32)

            if "tree" in key.lower():
                # Only where there's space (not on buildings)
                mask *= (1 - building_density)

            # Minimum spatial resolution
            min_pixels = max(1, int(intervention.spatial_resolution / pixel_size))
            if min_pixels > 1:
                # Erode to ensure minimum size
                from scipy.ndimage import binary_erosion
                binary_mask = mask > 0.5
                eroded = binary_erosion(binary_mask, iterations=min_pixels // 2)
                mask = eroded.astype(np.float32)

            applicability[key] = mask

        return applicability

    def apply_intervention(
        self,
        features: np.ndarray,
        intervention: InterventionType,
        applicability: np.ndarray,
        coverage_fraction: float = 1.0,
    ) -> np.ndarray:
        """
        Apply intervention to feature stack.

        Args:
            features: (C, H, W) feature stack
            intervention: Intervention type
            applicability: (H, W) applicability mask
            coverage_fraction: Fraction of applicable area to cover

        Returns:
            Modified feature stack
        """
        modified = features.copy()

        # Apply to applicable pixels
        mask = applicability > 0.5

        # Random sampling for coverage fraction
        if coverage_fraction < 1.0:
            random_mask = np.random.random(mask.shape) < coverage_fraction
            mask = mask & random_mask

        # Modify albedo (channel 5)
        if intervention.delta_albedo != 0:
            modified[5][mask] = np.clip(
                modified[5][mask] + intervention.delta_albedo, 0, 1
            )

        # Modify emissivity (channel 6) - small adjustment
        if intervention.delta_beta != 0:
            modified[6][mask] = np.clip(
                modified[6][mask] + intervention.delta_beta * 0.05, 0.9, 1.0
            )

        # Modify SVF (channel 12)
        if intervention.delta_svf != 0:
            modified[12][mask] = np.clip(
                modified[12][mask] + intervention.delta_svf, 0, 1
            )

        return modified


def load_interventions(config_path: str) -> Dict[str, InterventionType]:
    """Load custom interventions from JSON config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    interventions = {}
    for key, params in config.items():
        interventions[key] = InterventionType(**params)

    return interventions
