"""
Feature engineering for urban heat mitigation.

Computes:
- NDVI, NDBI, MNDWI, NDBaI
- Albedo (Liang 2001)
- Emissivity (Sobrino 2008)
- FVC, LAI
- Sky View Factor (SVF)
- Aerodynamic roughness length (Z0)

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
from typing import Tuple


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    return np.where(
        (nir + red) > 0,
        (nir.astype(np.float64) - red.astype(np.float64)) /
        (nir.astype(np.float64) + red.astype(np.float64)),
        0,
    )


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index."""
    return np.where(
        (swir1 + nir) > 0,
        (swir1.astype(np.float64) - nir.astype(np.float64)) /
        (swir1.astype(np.float64) + nir.astype(np.float64)),
        0,
    )


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified Normalized Difference Water Index."""
    return np.where(
        (green + swir1) > 0,
        (green.astype(np.float64) - swir1.astype(np.float64)) /
        (green.astype(np.float64) + swir1.astype(np.float64)),
        0,
    )


def ndbai(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Difference Bareness Index."""
    return np.where(
        (swir1 + swir2) > 0,
        (swir1.astype(np.float64) - swir2.astype(np.float64)) /
        (swir1.astype(np.float64) + swir2.astype(np.float64)),
        0,
    )


def albedo(
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b5: np.ndarray,
    b6: np.ndarray,
    b7: np.ndarray,
) -> np.ndarray:
    """
    Surface albedo using Liang (2001) coefficients.

    α = 0.130*B2 + 0.217*B3 + 0.262*B4 + 0.233*B5 + 0.102*B6 + 0.056*B7
    """
    return (
        0.130 * b2.astype(np.float64) +
        0.217 * b3.astype(np.float64) +
        0.262 * b4.astype(np.float64) +
        0.233 * b5.astype(np.float64) +
        0.102 * b6.astype(np.float64) +
        0.056 * b7.astype(np.float64)
    )


def fvc(ndvi_arr: np.ndarray) -> np.ndarray:
    """
    Fractional Vegetation Cover.

    FVC = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min))^2
    Assumes NDVI_min=0.05, NDVI_max=0.7
    """
    ndvi_min, ndvi_max = 0.05, 0.7
    fvc_arr = ((ndvi_arr - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
    return np.clip(fvc_arr, 0, 1)


def emissivity(ndvi_arr: np.ndarray, fvc_arr: np.ndarray) -> np.ndarray:
    """
    Surface emissivity (Sobrino 2008).

    ε = 0.985 + 0.012 * (1 - FVC)
    """
    return 0.985 + 0.012 * (1 - fvc_arr)


def lai(ndvi_arr: np.ndarray) -> np.ndarray:
    """
    Leaf Area Index from NDVI.

    LAI = 0.57 * exp(2.33 * NDVI) for NDVI > 0.1
    """
    return np.where(ndvi_arr > 0.1, 0.57 * np.exp(2.33 * ndvi_arr), 0)


def sky_view_factor(
    dem: np.ndarray,
    radius_m: int = 100,
    cell_size: int = 30,
) -> np.ndarray:
    """
    Approximate Sky View Factor from DEM.

    Uses focal statistics to estimate SVF.

    Args:
        dem: Digital Elevation Model
        radius_m: Search radius in meters
        cell_size: Pixel size in meters

    Returns:
        SVF array (0-1)
    """
    from scipy.ndimage import uniform_filter, maximum_filter

    radius_pixels = max(radius_m // cell_size, 1)

    dem_mean = uniform_filter(dem.astype(np.float64), radius_pixels * 2 + 1)
    dem_max = maximum_filter(dem.astype(np.float64), radius_pixels * 2 + 1)

    svf = np.where(
        (dem_max - dem) > 0,
        (dem_mean - dem) / (dem_max - dem),
        1,
    )
    return np.clip(svf, 0, 1)


def roughness_length(
    bldg_height: np.ndarray,
    bldg_density: np.ndarray,
    fvc_arr: np.ndarray,
) -> np.ndarray:
    """
    Aerodynamic roughness length (Z0).

    Grimmond & Oke (1999):
        Z0 = 0.1 * h_bldg * rho_bldg + 0.01 * rho_veg

    Args:
        bldg_height: Building height in meters
        bldg_density: Building density (0-1)
        fvc_arr: Fractional vegetation cover (0-1)

    Returns:
        Z0 in meters
    """
    return 0.1 * bldg_height * bldg_density + 0.01 * fvc_arr


def compute_all_features(
    landsat_bands: dict,
    dem: np.ndarray,
    bldg_height: np.ndarray,
    bldg_density: np.ndarray,
    road_density: np.ndarray,
    lcz: np.ndarray,
) -> dict:
    """
    Compute all features from input data.

    Args:
        landsat_bands: Dict with keys 'B2'-'B7', 'NDVI' (if pre-computed)
        dem: Digital Elevation Model
        bldg_height: Building height
        bldg_density: Building density
        road_density: Road density
        lcz: Local Climate Zone classification

    Returns:
        Dict with all computed features
    """
    nir = landsat_bands.get("B5", landsat_bands.get("NIR"))
    red = landsat_bands.get("B4", landsat_bands.get("Red"))
    green = landsat_bands.get("B3", landsat_bands.get("Green"))
    swir1 = landsat_bands.get("B6", landsat_bands.get("SWIR1"))
    swir2 = landsat_bands.get("B7", landsat_bands.get("SWIR2"))
    b2 = landsat_bands.get("B2")
    b3 = landsat_bands.get("B3")
    b4 = landsat_bands.get("B4")
    b5 = landsat_bands.get("B5")

    ndvi_arr = ndvi(nir, red)
    fvc_arr = fvc(ndvi_arr)

    features = {
        "NDVI": ndvi_arr,
        "NDBI": ndbi(swir1, nir),
        "MNDWI": mndwi(green, swir1),
        "NDBaI": ndbai(swir1, swir2),
        "ALBEDO": albedo(b2, b3, b4, b5, swir1, swir2),
        "EMISSIVITY": emissivity(ndvi_arr, fvc_arr),
        "FVC": fvc_arr,
        "LAI": lai(ndvi_arr),
        "SVF": sky_view_factor(dem),
        "Z0": roughness_length(bldg_height, bldg_density, fvc_arr),
        "ELEVATION": dem,
        "BUILDING_DENSITY": bldg_density,
        "BUILDING_HEIGHT": bldg_height,
        "ROAD_DENSITY": road_density,
        "LCZ": lcz,
    }

    return features
