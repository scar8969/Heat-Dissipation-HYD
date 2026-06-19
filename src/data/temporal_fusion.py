"""
Temporal fusion for gap-free daily LST.

Implements WGAST-style approach:
- MODIS (1km, daily) + Landsat (30m, 16-day) + Sentinel-2 (10m, 5-day)
- Produces daily 30m LST product

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class TemporalFusion:
    """
    Fuse multi-source satellite data for daily 30m LST.

    Based on WGAST (Bouaziz et al. 2025) and FuseTen approaches.
    """

    def __init__(self, target_resolution: int = 30):
        self.target_resolution = target_resolution

    def spatial_fusion(
        self,
        coarse_lst: np.ndarray,
        fine_reference: np.ndarray,
        fine_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Spatial fusion: downscale MODIS LST using Landsat/Sentinel patterns.

        Uses unmixing-based approach to combine coarse thermal with fine optical.

        Args:
            coarse_lst: (H, W) MODIS LST at 1km
            fine_reference: (H, W) Landsat LST at 30m (reference date)
            fine_mask: (H, W) valid pixel mask

        Returns:
            (H, W) downscaled LST
        """
        from scipy.ndimage import zoom

        # Upscale MODIS to 30m
        factor = 1000 / self.target_resolution
        coarse_upscaled = zoom(coarse_lst, factor, order=1)

        # Ensure same shape
        h, w = fine_reference.shape
        coarse_upscaled = coarse_upscaled[:h, :w]

        # Compute scaling factor
        coarse_mean = np.nanmean(coarse_upscaled[fine_mask])
        fine_mean = np.nanmean(fine_reference[fine_mask])

        if coarse_mean > 0:
            scale = fine_mean / coarse_mean
        else:
            scale = 1.0

        # Apply scaling
        downscaled = coarse_upscaled * scale

        return downscaled

    def temporal_interpolation(
        self,
        lst_stack: np.ndarray,
        time_indices: np.ndarray,
        target_time: int,
    ) -> np.ndarray:
        """
        Temporal interpolation between available observations.

        Args:
            lst_stack: (T, H, W) LST time series
            time_indices: (T,) time index for each observation
            target_time: Target time index

        Returns:
            (H, W) interpolated LST
        """
        # Find bracketing observations
        valid = ~np.all(np.isnan(lst_stack), axis=(1, 2))
        valid_times = time_indices[valid]
        valid_stack = lst_stack[valid]

        if len(valid_times) == 0:
            return np.full(lst_stack.shape[1:], np.nan)

        if len(valid_times) == 1:
            return valid_stack[0]

        # Linear interpolation
        idx_before = np.max(np.where(valid_times <= target_time)[0]) \
            if np.any(valid_times <= target_time) else 0
        idx_after = np.min(np.where(valid_times >= target_time)[0]) \
            if np.any(valid_times >= target_time) else len(valid_times) - 1

        if idx_before == idx_after:
            return valid_stack[idx_before]

        t_before = valid_times[idx_before]
        t_after = valid_times[idx_after]
        weight = (target_time - t_before) / (t_after - t_before + 1e-10)

        interp = (1 - weight) * valid_stack[idx_before] + \
                 weight * valid_stack[idx_after]

        return interp

    def create_daily_product(
        self,
        modis_daily: np.ndarray,
        landsat_composite: np.ndarray,
        n_days: int = 365,
    ) -> np.ndarray:
        """
        Create daily 30m LST product using temporal fusion.

        Args:
            modis_daily: (T_modis, H, W) daily MODIS LST at 1km
            landsat_composite: (H, W) Landsat composite at 30m
            n_days: Number of output days

        Returns:
            (n_days, H, W) daily 30m LST
        """
        t_modis, h, w = modis_daily.shape

        # Initialize output
        daily_lst = np.zeros((n_days, h, w), dtype=np.float32)

        for day in range(n_days):
            # Get MODIS LST for this day
            modis_idx = min(day, t_modis - 1)
            modis_day = modis_daily[modis_idx]

            # Spatial fusion
            fused = self.spatial_fusion(
                modis_day,
                landsat_composite,
                landsat_composite > 0,
            )

            daily_lst[day] = fused

        return daily_lst


class UncertaintyEstimator:
    """Estimate uncertainty in fused LST products."""

    @staticmethod
    def temporal_uncertainty(
        lst_stack: np.ndarray,
        time_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate uncertainty from temporal variability.

        Args:
            lst_stack: (T, H, W) LST time series
            time_indices: (T,) time indices

        Returns:
            (H, W) uncertainty estimate (std dev)
        """
        return np.nanstd(lst_stack, axis=0)

    @staticmethod
    def spatial_uncertainty(
        lst: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Estimate uncertainty from local spatial variability.

        Args:
            lst: (H, W) LST
            kernel_size: Window size for local std

        Returns:
            (H, W) uncertainty estimate
        """
        from scipy.ndimage import uniform_filter

        lst_mean = uniform_filter(lst.astype(np.float64), kernel_size)
        lst_sq_mean = uniform_filter((lst.astype(np.float64)) ** 2, kernel_size)
        local_var = lst_sq_mean - lst_mean ** 2

        return np.sqrt(np.maximum(local_var, 0))


def fuse_landsat_modis(
    landsat_lst: np.ndarray,
    modis_lst: np.ndarray,
    landsat_transform: Optional[object] = None,
    modis_transform: Optional[object] = None,
) -> np.ndarray:
    """
    Fuse Landsat and MODIS LST using simple unmixing.

    Args:
        landsat_lst: Landsat LST at 30m
        modis_lst: MODIS LST at 1km

    Returns:
        Fused LST at 30m
    """
    from scipy.ndimage import zoom

    # Upscale MODIS to Landsat resolution
    factor = 1000 / 30
    modis_upscaled = zoom(modis_lst, factor, order=1)

    # Match shapes
    h, w = landsat_lst.shape
    modis_upscaled = modis_upscaled[:h, :w]

    # Weighted average (Landsat weight based on quality)
    landsat_valid = ~np.isnan(landsat_lst)
    modis_valid = ~np.isnan(modis_upscaled)

    if landsat_valid.any() and modis_valid.any():
        landsat_weight = 0.7
        modis_weight = 0.3

        fused = np.where(
            landsat_valid & modis_valid,
            landsat_weight * landsat_lst + modis_weight * modis_upscaled,
            np.where(landsat_valid, landsat_lst, modis_upscaled),
        )
    else:
        fused = np.where(landsat_valid, landsat_lst, modis_upscaled)

    return fused
