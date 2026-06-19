"""
Preprocessing pipeline for urban heat mitigation data.

Handles:
- Cloud masking (QA_PIXEL bitwise)
- LST retrieval and scaling
- Spectral index computation
- ERA5 downscaling (bilinear + lapse rate)
- Data resampling to 30m
- Gap filling and mosaicking
- Feature stacking

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from typing import Tuple, Optional, List
import warnings


class CloudMasker:
    """Cloud masking using Landsat 8 QA_PIXEL band."""

    @staticmethod
    def mask_landsat8(qa_pixel: np.ndarray) -> np.ndarray:
        """
        Apply cloud mask to Landsat 8 QA_PIXEL.

        Bit flags:
            Bit 0: Fill
            Bit 1: Dilated Cloud
            Bit 2: Cirrus (high confidence)
            Bit 3: Cloud (high confidence)
            Bit 4: Cloud Shadow (high confidence)
            Bit 5: Snow

        Args:
            qa_pixel: QA_PIXEL band values

        Returns:
            Boolean mask (True = clear)
        """
        clear = (
            (qa_pixel & 1) == 0  # Not fill
            & ((qa_pixel >> 1) & 1) == 0  # Not dilated cloud
            & ((qa_pixel >> 2) & 1) == 0  # Not cirrus
            & ((qa_pixel >> 3) & 1) == 0  # Not cloud
            & ((qa_pixel >> 4) & 1) == 0  # Not cloud shadow
            & ((qa_pixel >> 5) & 1) == 0  # Not snow
        )
        return clear

    @staticmethod
    def mask_sentinel2(scl: np.ndarray) -> np.ndarray:
        """
        Apply cloud mask to Sentinel-2 SCL band.

        SCL classes to mask:
            3: Cloud shadow
            8: Cloud medium probability
            9: Cloud high probability
            10: Thin cirrus

        Args:
            scl: SCL band values

        Returns:
            Boolean mask (True = clear)
        """
        clear = ~np.isin(scl, [3, 8, 9, 10])
        return clear


class LSTRetriever:
    """Land Surface Temperature retrieval from Landsat 8."""

    @staticmethod
    def retrieve_from_landsat8(
        thermal_band: np.ndarray,
        qa_pixel: np.ndarray,
        scale_factor: float = 0.00341802,
        offset: float = 149.0,
        to_celsius: bool = True,
    ) -> np.ndarray:
        """
        Retrieve LST from Landsat 8 Collection 2 Level 2.

        Formula: LST = ST_B10 * scale_factor + offset

        Args:
            thermal_band: ST_B10 band values (DN)
            qa_pixel: QA_PIXEL band for cloud masking
            scale_factor: Scale factor (default: 0.00341802)
            offset: Offset (default: 149.0)
            to_celsius: Convert to Celsius (default: True)

        Returns:
            LST array in °C
        """
        # Apply scale factor and offset
        lst = thermal_band.astype(np.float32) * scale_factor + offset

        # Convert from Kelvin to Celsius
        if to_celsius:
            lst = lst - 273.15

        # Apply cloud mask
        mask = CloudMasker.mask_landsat8(qa_pixel)
        lst[~mask] = np.nan

        return lst

    @staticmethod
    def retrieve_from_ecostress(
        lst_band: np.ndarray,
        scale_factor: float = 0.02,
        to_celsius: bool = True,
    ) -> np.ndarray:
        """
        Retrieve LST from ECOSTRESS.

        Args:
            lst_band: ECOSTRESS LST band values
            scale_factor: Scale factor (default: 0.02)
            to_celsius: Convert to Celsius

        Returns:
            LST array in °C
        """
        lst = lst_band.astype(np.float32) * scale_factor

        if to_celsius:
            lst = lst - 273.15

        return lst


class SpectralIndices:
    """Compute spectral indices from Landsat 8 bands."""

    @staticmethod
    def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index."""
        return (nir.astype(float) - red.astype(float)) / \
               (nir.astype(float) + red.astype(float) + 1e-10)

    @staticmethod
    def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Normalized Difference Built-up Index."""
        return (swir1.astype(float) - nir.astype(float)) / \
               (swir1.astype(float) + nir.astype(float) + 1e-10)

    @staticmethod
    def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """Modified Normalized Difference Water Index."""
        return (green.astype(float) - swir1.astype(float)) / \
               (green.astype(float) + swir1.astype(float) + 1e-10)

    @staticmethod
    def ndbai(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """Normalized Difference Bareness Index."""
        return (swir1.astype(float) - swir2.astype(float)) / \
               (swir1.astype(float) + swir2.astype(float) + 1e-10)

    @staticmethod
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

        Formula:
            α = 0.130*B2 + 0.217*B3 + 0.262*B4 + 0.233*B5 + 0.102*B6 + 0.056*B7
        """
        return (
            0.130 * b2.astype(float) +
            0.217 * b3.astype(float) +
            0.262 * b4.astype(float) +
            0.233 * b5.astype(float) +
            0.102 * b6.astype(float) +
            0.056 * b7.astype(float)
        )

    @staticmethod
    def emissivity(
        ndvi: np.ndarray,
        fvc: np.ndarray,
    ) -> np.ndarray:
        """
        Surface emissivity from NDVI and FVC.

        Formula (Sobrino 2008):
            ε = 0.985 + 0.012 * (1 - FVC)

        Args:
            ndvi: Normalized Difference Vegetation Index
            fvc: Fractional Vegetation Cover

        Returns:
            Emissivity array (0.9-1.0)
        """
        return 0.985 + 0.012 * (1 - fvc)

    @staticmethod
    def fvc(ndvi: np.ndarray) -> np.ndarray:
        """
        Fractional Vegetation Cover.

        Formula:
            FVC = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min))^2

        Assumes NDVI_min = 0.05, NDVI_max = 0.7
        """
        ndvi_min, ndvi_max = 0.05, 0.7
        fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
        return np.clip(fvc, 0, 1)

    @staticmethod
    def lai(ndvi: np.ndarray) -> np.ndarray:
        """
        Leaf Area Index from NDVI.

        Empirical formula:
            LAI = 0.57 * exp(2.33 * NDVI) for NDVI > 0.1
        """
        lai = np.where(ndvi > 0.1, 0.57 * np.exp(2.33 * ndvi), 0)
        return np.clip(lai, 0, 8)


class ERA5Downscaler:
    """Downscale ERA5-Land data to 30m using bilinear + lapse rate."""

    @staticmethod
    def downscale_temperature(
        temp_era5: np.ndarray,
        elevation: np.ndarray,
        era5_resolution: float = 0.1,
        target_resolution: float = 30,
        lapse_rate: float = -6.5,
    ) -> np.ndarray:
        """
        Downscale ERA5 2m temperature using elevation lapse rate.

        Formula:
            T_30m = T_era5 + lapse_rate * (elevation - elevation_era5) / 1000

        Args:
            temp_era5: ERA5 temperature at 0.1° resolution
            elevation: SRTM elevation at 30m resolution
            era5_resolution: ERA5 grid resolution in degrees
            target_resolution: Target resolution in meters
            lapse_rate: Temperature lapse rate (°C/km, default: -6.5)

        Returns:
            Downscaled temperature at 30m
        """
        # Upscale ERA5 to 30m using bilinear interpolation
        from scipy.ndimage import zoom

        factor = era5_resolution * 111000 / target_resolution
        temp_30m = zoom(temp_era5, factor, order=1)

        # Elevation difference
        elev_era5 = zoom(elevation, 1 / factor, order=1)
        elev_era5 = zoom(elev_era5, factor, order=1)

        # Apply lapse rate correction
        delta_elev = elevation - elev_era5
        correction = lapse_rate * delta_elev / 1000

        return temp_30m + correction

    @staticmethod
    def downscale_generic(
        data_era5: np.ndarray,
        era5_resolution: float = 0.1,
        target_resolution: float = 30,
    ) -> np.ndarray:
        """
        Generic bilinear downscaling for non-temperature variables.

        Args:
            data_era5: ERA5 data at 0.1° resolution
            era5_resolution: ERA5 grid resolution in degrees
            target_resolution: Target resolution in meters

        Returns:
            Downscaled data at 30m
        """
        from scipy.ndimage import zoom

        factor = era5_resolution * 111000 / target_resolution
        return zoom(data_era5, factor, order=1)


class RasterProcessor:
    """Process raster files for resampling and stacking."""

    @staticmethod
    def resample_to_grid(
        src_path: str,
        dst_path: str,
        target_shape: Tuple[int, int],
        target_transform: rasterio.Affin,
        target_crs: str = "EPSG:32643",
        resampling: Resampling = Resampling.bilinear,
    ) -> None:
        """
        Resample raster to target grid.

        Args:
            src_path: Source raster path
            dst_path: Output raster path
            target_shape: (height, width) of target grid
            target_transform: Target affine transform
            target_crs: Target CRS
            resampling: Resampling method
        """
        with rasterio.open(src_path) as src:
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": target_crs,
                "transform": target_transform,
                "width": target_shape[1],
                "height": target_shape[0],
            })

            with rasterio.open(dst_path, "w", **kwargs) as dst:
                for band in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                    )

    @staticmethod
    def stack_bands(
        file_list: List[str],
        output_path: str,
        band_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Stack multiple single-band rasters into multi-band GeoTIFF.

        Args:
            file_list: List of raster file paths
            output_path: Output file path
            band_names: Optional band names

        Returns:
            Stacked array (bands, height, width)
        """
        if not file_list:
            raise ValueError("No files provided")

        with rasterio.open(file_list[0]) as src:
            meta = src.meta.copy()
            height, width = src.shape

        n_bands = len(file_list)
        stack = np.zeros((n_bands, height, width), dtype=np.float32)

        for i, fpath in enumerate(file_list):
            with rasterio.open(fpath) as src:
                stack[i] = src.read(1)

        meta.update({
            "count": n_bands,
            "dtype": "float32",
        })

        if band_names:
            meta["descriptions"] = band_names

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(stack)

        return stack

    @staticmethod
    def gap_fill_nearest(
        image: np.ndarray,
        max_distance: int = 5,
    ) -> np.ndarray:
        """
        Fill gaps using nearest-neighbor interpolation.

        Args:
            image: Input array with NaN values
            max_distance: Maximum fill distance in pixels

        Returns:
            Gap-filled array
        """
        from scipy.ndimage import distance_transform_edt

        mask = np.isnan(image)
        if not mask.any():
            return image

        # Distance transform
        dist, indices = distance_transform_edt(mask, return_indices=True)

        # Fill from nearest valid pixel
        filled = image.copy()
        fill_mask = mask & (dist <= max_distance)
        filled[fill_mask] = image[tuple(indices[:, fill_mask])]

        return filled


class FeatureStacker:
    """Create final feature stack from processed data."""

    def __init__(self, resolution: int = 30):
        self.resolution = resolution

    def create_stack(
        self,
        lst: np.ndarray,
        ndvi: np.ndarray,
        ndbi: np.ndarray,
        mndwi: np.ndarray,
        ndbai: np.ndarray,
        albedo: np.ndarray,
        emissivity: np.ndarray,
        fvc: np.ndarray,
        lai: np.ndarray,
        elevation: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        svf: np.ndarray,
        z0: np.ndarray,
        building_density: np.ndarray,
        building_height: np.ndarray,
        road_density: np.ndarray,
        lcz: np.ndarray,
        output_path: str,
    ) -> np.ndarray:
        """
        Create 26-band feature stack.

        Band layout:
            0: LST (target)
            1: NDVI
            2: NDBI
            3: MNDWI
            4: NDBaI
            5: ALBEDO
            6: EMISSIVITY
            7: FVC
            8: LAI
            9: ELEVATION
            10: SLOPE
            11: ASPECT
            12: SVF
            13: Z0
            14: BUILDING_DENSITY
            15: BUILDING_HEIGHT
            16: ROAD_DENSITY
            17: LCZ

        Args:
            ...: All input arrays
            output_path: Output GeoTIFF path

        Returns:
            Stacked array (18, height, width)
        """
        stack = np.stack([
            lst, ndvi, ndbi, mndwi, ndbai, albedo,
            emissivity, fvc, lai, elevation, slope, aspect,
            svf, z0, building_density, building_height,
            road_density, lcz,
        ], axis=0)

        # Save to GeoTIFF
        height, width = lst.shape
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 18,
            "dtype": "float32",
            "crs": "EPSG:32643",
            "transform": from_bounds(
                78.25, 17.24, 78.72, 17.68, width, height
            ),
        }

        band_names = [
            "LST", "NDVI", "NDBI", "MNDWI", "NDBaI", "ALBEDO",
            "EMISSIVITY", "FVC", "LAI", "ELEVATION", "SLOPE", "ASPECT",
            "SVF", "Z0", "BUILDING_DENSITY", "BUILDING_HEIGHT",
            "ROAD_DENSITY", "LCZ",
        ]

        with rasterio.open(output_path, "w", **meta) as dst:
            for i, name in enumerate(band_names):
                dst.write(stack[i], i + 1)
                dst.set_band_description(i + 1, name)

        print(f"Feature stack saved to {output_path}")
        print(f"  Shape: {stack.shape}")
        print(f"  Bands: {band_names}")

        return stack
