"""
Heat hotspot detection and classification.

Identifies and classifies urban heat hotspots:
- Pixel-level hotspot detection
- Zone aggregation for management
- Hotspot classification by intensity and drivers
- Trend analysis over time

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import ndimage
from sklearn.cluster import KMeans
import json

logger = logging.getLogger(__name__)


class HotspotDetector:
    """
    Detect and classify urban heat hotspots.

    Hotspots are defined as areas with LST significantly above
    the urban mean temperature.
    """

    def __init__(
        self,
        lst_mean: float = None,
        lst_std: float = None,
        hotspot_threshold: float = 1.5,
        extreme_threshold: float = 2.5,
        min_area_ha: float = 1.0,
    ):
        """
        Initialize hotspot detector.

        Args:
            lst_mean: Mean LST for the city (computed if None)
            lst_std: Standard deviation of LST (computed if None)
            hotspot_threshold: Number of std above mean for hotspot
            extreme_threshold: Number of std above mean for extreme hotspot
            min_area_ha: Minimum hotspot area in hectares
        """
        self.lst_mean = lst_mean
        self.lst_std = lst_std
        self.hotspot_threshold = hotspot_threshold
        self.extreme_threshold = extreme_threshold
        self.min_area_ha = min_area_ha

    def detect(
        self,
        lst: np.ndarray,
        pixel_size: float = 30.0,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Detect heat hotspots from LST map.

        Args:
            lst: (H, W) land surface temperature
            pixel_size: Pixel size in meters
            mask: Optional mask (True = valid)

        Returns:
            Dict with hotspot maps and statistics
        """
        # Compute statistics if not provided
        if self.lst_mean is None:
            valid = mask if mask is not None else np.ones_like(lst, dtype=bool)
            self.lst_mean = np.mean(lst[valid])
            self.lst_std = np.std(lst[valid])

        # Compute anomaly
        anomaly = lst - self.lst_mean

        # Classify intensity
        intensity = np.zeros_like(lst, dtype=np.int8)
        intensity[anomaly >= self.hotspot_threshold * self.lst_std] = 1  # Moderate
        intensity[anomaly >= self.extreme_threshold * self.lst_std] = 2  # Extreme

        # Apply mask
        if mask is not None:
            intensity[~mask] = 0

        # Label connected components
        labeled, num_features = ndimage.label(intensity > 0)

        # Filter by minimum area
        pixel_area_ha = (pixel_size ** 2) / 10000
        min_pixels = self.min_area_ha / pixel_area_ha

        valid_labels = []
        for i in range(1, num_features + 1):
            area = np.sum(labeled == i)
            if area >= min_pixels:
                valid_labels.append(i)

        # Create filtered hotspot map
        hotspot_map = np.isin(labeled, valid_labels).astype(np.int8)

        # Compute statistics
        stats = self._compute_statistics(hotspot_map, lst, labeled, valid_labels, pixel_area_ha)

        return {
            "hotspot_map": hotspot_map,
            "intensity_map": intensity,
            "labeled_map": labeled,
            "anomaly_map": anomaly,
            "statistics": stats,
        }

    def _compute_statistics(
        self,
        hotspot_map: np.ndarray,
        lst: np.ndarray,
        labeled: np.ndarray,
        valid_labels: List[int],
        pixel_area_ha: float,
    ) -> Dict:
        """Compute hotspot statistics."""
        total_area_ha = np.sum(hotspot_map) * pixel_area_ha

        hotspots = []
        for i, label in enumerate(valid_labels, 1):
            mask = labeled == label
            area_ha = np.sum(mask) * pixel_area_ha
            mean_lst = np.mean(lst[mask])
            max_lst = np.max(lst[mask])

            hotspots.append({
                "id": i,
                "area_ha": float(area_ha),
                "mean_lst": float(mean_lst),
                "max_lst": float(max_lst),
                "centroid": self._compute_centroid(mask),
            })

        return {
            "num_hotspots": len(valid_labels),
            "total_area_ha": float(total_area_ha),
            "fraction_hotspot": float(total_area_ha / (hotspot_map.size * pixel_area_ha)),
            "hotspots": hotspots,
        }

    def _compute_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of a hotspot."""
        ys, xs = np.where(mask)
        return float(np.mean(xs)), float(np.mean(ys))


class ZoneAggregator:
    """
    Aggregate pixels into management zones.

    Uses K-means clustering based on:
- Spatial location (x, y)
- LST values
- Land use/land cover type
- Building density

This reduces 694K pixels to ~500 zones for optimization.
    """

    def __init__(self, n_zones: int = 500, random_state: int = 42):
        self.n_zones = n_zones
        self.random_state = random_state
        self.kmeans = None
        self.zone_stats = None

    def fit(
        self,
        lst: np.ndarray,
        lulc: np.ndarray,
        building_density: np.ndarray,
        pixel_size: float = 30.0,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fit zone aggregation.

        Args:
            lst: (H, W) LST map
            lulc: (H, W) land use/land cover
            building_density: (H, W) building density [0, 1]
            pixel_size: Pixel size in meters
            mask: Optional mask (True = valid)

        Returns:
            Zone assignment map and statistics
        """
        H, W = lst.shape

        # Create feature matrix
        y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        features = np.stack([
            lst.ravel(),
            lulc.ravel(),
            building_density.ravel(),
            x_grid.ravel() * pixel_size,
            y_grid.ravel() * pixel_size,
        ], axis=1)

        # Apply mask
        if mask is not None:
            valid = mask.ravel()
        else:
            valid = np.ones(features.shape[0], dtype=bool)

        features_valid = features[valid]

        # K-means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_zones,
            random_state=self.random_state,
            n_init=10,
        )
        labels_valid = self.kmeans.fit_predict(features_valid)

        # Create full label map
        labels = np.full(H * W, -1, dtype=np.int32)
        labels[valid] = labels_valid
        labels = labels.reshape(H, W)

        # Compute zone statistics
        self.zone_stats = self._compute_zone_stats(labels, lst, lulc, building_density, pixel_area_ha=(pixel_size**2)/10000)

        return {
            "zone_map": labels,
            "zone_stats": self.zone_stats,
        }

    def _compute_zone_stats(
        self,
        labels: np.ndarray,
        lst: np.ndarray,
        lulc: np.ndarray,
        building_density: np.ndarray,
        pixel_area_ha: float,
    ) -> List[Dict]:
        """Compute statistics for each zone."""
        stats = []
        for zone_id in range(self.n_zones):
            mask = labels == zone_id
            if not np.any(mask):
                continue

            area_ha = np.sum(mask) * pixel_area_ha
            mean_lst = float(np.mean(lst[mask]))
            std_lst = float(np.std(lst[mask]))
            mean_bd = float(np.mean(building_density[mask]))

            # Dominant LULC
            lulc_vals = lulc[mask]
            lulc_counts = np.bincount(lulc_vals.astype(int), minlength=10)
            dominant_lulc = int(np.argmax(lulc_counts))

            stats.append({
                "zone_id": zone_id,
                "area_ha": area_ha,
                "mean_lst": mean_lst,
                "std_lst": std_lst,
                "mean_building_density": mean_bd,
                "dominant_lulc": dominant_lulc,
                "pixel_count": int(np.sum(mask)),
            })

        return stats


class HotspotClassifier:
    """
    Classify hotspots by their primary drivers.

    Uses feature importance analysis to determine whether
    each hotspot is primarily driven by:
    - Built-up area (high NDBI)
    - Vegetation deficit (low NDVI)
    - Water deficit (low MNDWI)
    - Canyon geometry (low SVF)
    - Anthropogenic heat
    """

    DRIVER_NAMES = {
        0: "built_up",
        1: "vegetation_deficit",
        2: "water_deficit",
        3: "canyon_geometry",
        4: "anthropogenic",
        5: "mixed",
    }

    def __init__(self):
        pass

    def classify(
        self,
        lst: np.ndarray,
        ndvi: np.ndarray,
        ndbi: np.ndarray,
        mndwi: np.ndarray,
        svf: np.ndarray,
        building_density: np.ndarray,
        labeled_hotspots: np.ndarray,
        valid_labels: List[int],
    ) -> List[Dict]:
        """
        Classify hotspots by driver.

        Args:
            lst: LST map
            ndvi: NDVI map
            ndbi: NDBI map
            mndwi: MNDWI map
            svf: Sky View Factor map
            building_density: Building density map
            labeled_hotspots: Labeled hotspot map
            valid_labels: Valid hotspot labels

        Returns:
            List of classified hotspots
        """
        classifications = []

        for label in valid_labels:
            mask = labeled_hotspots == label

            # Compute driver metrics within hotspot
            metrics = {
                "ndvi_mean": float(np.mean(ndvi[mask])),
                "ndbi_mean": float(np.mean(ndbi[mask])),
                "mndwi_mean": float(np.mean(mndwi[mask])),
                "svf_mean": float(np.mean(svf[mask])),
                "building_density_mean": float(np.mean(building_density[mask])),
            }

            # Classify primary driver
            driver_scores = self._compute_driver_scores(metrics)
            primary_driver = max(driver_scores, key=driver_scores.get)

            classifications.append({
                "zone_id": label,
                "primary_driver": primary_driver,
                "driver_scores": driver_scores,
                "metrics": metrics,
            })

        return classifications

    def _compute_driver_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute driver scores based on feature values."""
        scores = {}

        # Built-up driver: high NDBI, high building density
        scores["built_up"] = max(0, metrics["ndbi_mean"]) * metrics["building_density_mean"]

        # Vegetation deficit: low NDVI
        scores["vegetation_deficit"] = max(0, 0.3 - metrics["ndvi_mean"])

        # Water deficit: low MNDWI
        scores["water_deficit"] = max(0, 0.1 - metrics["mndwi_mean"])

        # Canyon geometry: low SVF
        scores["canyon_geometry"] = max(0, 0.5 - metrics["svf_mean"])

        # Anthropogenic: moderate building density, high LST
        scores["anthropogenic"] = metrics["building_density_mean"] * 0.5

        # Normalize
        total = sum(scores.values()) + 1e-6
        scores = {k: v / total for k, v in scores.items()}

        return scores


class TrendAnalyzer:
    """
    Analyze temporal trends in heat hotspots.

    Identifies:
- Hotspots that are growing
- Hotspots that are shrinking
- New emerging hotspots
- Persistent hotspots
    """

    def __init__(self):
        pass

    def analyze(
        self,
        hotspot_maps: List[np.ndarray],
        timestamps: List[str],
        pixel_size: float = 30.0,
    ) -> Dict:
        """
        Analyze temporal trends.

        Args:
            hotspot_maps: List of hotspot maps (binary)
            timestamps: List of timestamps
            pixel_size: Pixel size in meters

        Returns:
            Trend analysis results
        """
        n_timesteps = len(hotspot_maps)
        pixel_area_ha = (pixel_size ** 2) / 10000

        # Compute area over time
        areas = [np.sum(m) * pixel_area_ha for m in hotspot_maps]

        # Identify persistent hotspots (present in >80% of timesteps)
        stacked = np.stack(hotspot_maps)
        persistence = np.mean(stacked, axis=0)
        persistent = persistence > 0.8

        # Identify growing/shrinking hotspots
        if n_timesteps >= 2:
            first = hotspot_maps[0]
            last = hotspot_maps[-1]

            growing = (last > 0) & (first == 0)
            shrinking = (last == 0) & (first > 0)
            stable = (last > 0) & (first > 0)
        else:
            growing = np.zeros_like(hotspot_maps[0], dtype=bool)
            shrinking = np.zeros_like(hotspot_maps[0], dtype=bool)
            stable = hotspot_maps[0].astype(bool)

        return {
            "timestamps": timestamps,
            "areas_ha": areas,
            "persistence_map": persistence,
            "persistent_area_ha": float(np.sum(persistent) * pixel_area_ha),
            "growing_area_ha": float(np.sum(growing) * pixel_area_ha),
            "shrinking_area_ha": float(np.sum(shrinking) * pixel_area_ha),
            "stable_area_ha": float(np.sum(stable) * pixel_area_ha),
        }
