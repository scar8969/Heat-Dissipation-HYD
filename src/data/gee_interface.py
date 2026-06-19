"""
Google Earth Engine interface for urban heat mitigation data acquisition.

Handles:
- Landsat 8/9 LST, spectral indices
- ECOSTRESS LST
- Sentinel-2 LULC classification
- MODIS daily LST
- ERA5 meteorological variables
- GHSL built fraction/height
- SRTM elevation
- WUDAPT LCZ maps

Author: Urban Heat Mitigation Project
Date: June 2026
"""

import ee
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path


class GEEInterface:
    """
    Google Earth Engine interface for data acquisition.

    Usage:
        gee = GEEInterface()
        gee.authenticate()
        region = gee.get_hyderabad_boundary()
        lst = gee.get_landsat_composite(region)
    """

    # Hyderabad GHMC bounds
    HYDERABAD_BOUNDS = [78.25, 17.24, 78.72, 17.68]  # W, S, E, N

    # Landsat 8/9 Path/Row for Hyderabad
    PATH_ROW = [(144, 48), (144, 49)]

    # Seasonal windows
    SEASON_MONTHS = {
        "pre_monsoon": [3, 4, 5, 6],
        "post_monsoon": [10, 11, 12],
    }

    def __init__(self):
        self._authenticated = False

    def authenticate(self) -> None:
        """Initialize GEE session."""
        try:
            ee.Initialize(project="your-project-id")
            self._authenticated = True
            print("GEE authenticated successfully")
        except Exception:
            print("Run 'earthengine authenticate' first")
            raise

    def _check_auth(self):
        if not self._authenticated:
            raise RuntimeError("Call authenticate() first")

    def get_hyderabad_boundary(self) -> ee.FeatureCollection:
        """Get GHMC boundary from GAUL dataset."""
        self._check_auth()
        boundary = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2") \
            .filter(ee.Filter.eq("ADM2_NAME", "Hyderabad"))
        return boundary

    def get_region(self, buffer_m: float = 0) -> ee.Geometry:
        """Get rectangular region around Hyderabad."""
        self._check_auth()
        region = ee.Geometry.Rectangle(self.HYDERABAD_BOUNDS)
        if buffer_m > 0:
            region = region.buffer(buffer_m)
        return region

    def _apply_l8_cloud_mask(self, image: ee.Image) -> ee.Image:
        """Apply cloud mask to Landsat 8 imagery using QA_PIXEL."""
        qa = image.select("QA_PIXEL")
        mask = (
            qa.bitwiseAnd(1).eq(0)  # clear
            .And(qa.bitwiseAnd(2).eq(0))  # not dilated cloud
            .And(qa.bitwiseAnd(4).eq(0))  # not cloud shadow
            .And(qa.bitwiseAnd(8).eq(0))  # not snow
        )
        return image.updateMask(mask)

    def _compute_l8_indices(self, image: ee.Image) -> ee.Image:
        """Compute spectral indices from Landsat 8 bands."""
        ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
        mndwi = image.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI")
        ndbai = image.normalizedDifference(["SR_B6", "SR_B7"]).rename("NDBaI")

        # Albedo (Liang 2001 coefficients)
        albedo = (
            image.select("SR_B2").multiply(0.130)
            .add(image.select("SR_B3").multiply(0.217))
            .add(image.select("SR_B4").multiply(0.262))
            .add(image.select("SR_B5").multiply(0.233))
            .add(image.select("SR_B6").multiply(0.102))
            .add(image.select("SR_B7").multiply(0.056))
            .rename("ALBEDO")
        )

        # LST (scale factor for Landsat 8 Collection 2)
        lst = image.select("ST_B10") \
            .multiply(0.00341802).add(149.0) \
            .subtract(273.15) \
            .rename("LST")

        return image.addBands([ndvi, ndbi, mndwi, ndbai, albedo, lst])

    def get_landsat_composite(
        self,
        region: ee.Geometry,
        start_date: str = "2018-06-01",
        end_date: str = "2024-09-30",
        months: Optional[List[int]] = None,
        cloud_threshold: float = 10.0,
    ) -> ee.Image:
        """
        Create Landsat 8/9 median composite.

        Args:
            region: Study area geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            months: Filter by months (default: pre/post monsoon)
            cloud_threshold: Maximum cloud cover percentage

        Returns:
            ee.Image with LST, NDVI, NDBI, MNDWI, NDBaI, ALBEDO
        """
        self._check_auth()

        if months is None:
            months = self.SEASON_MONTHS["pre_monsoon"] + \
                     self.SEASON_MONTHS["post_monsoon"]

        # Landsat 8 Collection 2 Level 2
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.calendarRange(months[0], months[-1], "month"))
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
            .map(self._apply_l8_cloud_mask)
            .map(self._compute_l8_indices)
        )

        # Also add Landsat 9 if available
        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.calendarRange(months[0], months[-1], "month"))
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_threshold))
            .map(self._apply_l8_cloud_mask)
            .map(self._compute_l8_indices)
        )

        # Merge and composite
        composite = l8.merge(l9).median().clip(region)
        return composite

    def get_ecostress_lst(
        self,
        region: ee.Geometry,
        start_date: str = "2018-06-01",
        end_date: str = "2024-09-30",
    ) -> ee.Image:
        """
        Get ECOSTRESS LST composite (70m, resampled to 30m).

        Args:
            region: Study area geometry
            start_date: Start date
            end_date: End date

        Returns:
            ee.Image with ECOSTRESS LST
        """
        self._check_auth()

        eco = (
            ee.ImageCollection("ECOSTRESS/ECO2LSTE/001")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select("LST")
            .map(lambda img: img.multiply(0.02).rename("LST_ECO"))
        )

        composite = eco.mean().clip(region)
        return composite

    def get_sentinel2_composite(
        self,
        region: ee.Geometry,
        start_date: str = "2022-01-01",
        end_date: str = "2024-06-01",
        cloud_threshold: float = 20.0,
    ) -> ee.Image:
        """
        Get Sentinel-2 composite for LULC classification.

        Args:
            region: Study area geometry
            start_date: Start date
            end_date: End date
            cloud_threshold: Maximum cloudy pixel percentage

        Returns:
            ee.Image with Sentinel-2 bands
        """
        self._check_auth()

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
            .map(lambda img: img.divide(10000))  # Scale to [0, 1]
            .median()
            .clip(region)
        )

        return s2

    def get_modis_lst(
        self,
        region: ee.Geometry,
        start_date: str = "2018-06-01",
        end_date: str = "2024-09-30",
    ) -> ee.ImageCollection:
        """
        Get MODIS daily LST collection for temporal fusion.

        Args:
            region: Study area geometry
            start_date: Start date
            end_date: End date

        Returns:
            ee.ImageCollection with daily MODIS LST
        """
        self._check_auth()

        modis = (
            ee.ImageCollection("MODIS/061/MYD11A1")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select("LST_Day_1km", "LST_Night_1km")
            .map(lambda img: img.multiply(0.02).subtract(273.15))
        )

        return modis

    def get_era5_land(
        self,
        region: ee.Geometry,
        start_date: str = "2018-06-01",
        end_date: str = "2024-09-30",
    ) -> ee.ImageCollection:
        """
        Get ERA5-Land hourly meteorological data.

        Args:
            region: Study area geometry
            start_date: Start date
            end_date: End date

        Returns:
            ee.ImageCollection with downscaled ERA5 variables
        """
        self._check_auth()

        era5 = (
            ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select([
                "temperature_2m",
                "dewpoint_temperature_2m",
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
                "surface_pressure",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards",
                "total_precipitation",
            ])
        )

        # Daily mean
        daily = era5.filterDate(start_date, end_date) \
            .mean() \
            .clip(region)

        return daily

    def get_srtm_elevation(self, region: ee.Geometry) -> ee.Image:
        """
        Get SRTM 30m elevation data.

        Args:
            region: Study area geometry

        Returns:
            ee.Image with elevation, slope, aspect
        """
        self._check_auth()

        srtm = ee.Image("USGS/SRTMGL1_003").clip(region)

        # Compute slope and aspect
        slope = ee.Terrain.slope(srtm).rename("SLOPE")
        aspect = ee.Terrain.aspect(srtm).rename("ASPECT")

        return srtm.rename("ELEVATION").addBands([slope, aspect])

    def get_ghsl_built(self, region: ee.Geometry) -> ee.Image:
        """
        Get Global Human Settlement Layer built fraction and height.

        Args:
            region: Study area geometry

        Returns:
            ee.Image with GHSL built metrics
        """
        self._check_auth()

        ghsl = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020") \
            .select("built_surface") \
            .clip(region)

        return ghsl.rename("GHSL_BUILT")

    def compute_sky_view_factor(
        self,
        dem: ee.Image,
        region: ee.Geometry,
        radius: int = 100,
    ) -> ee.Image:
        """
        Compute Sky View Factor from DEM using hemispherical approach.

        Args:
            dem: Digital Elevation Model
            region: Study area geometry
            radius: Search radius in meters

        Returns:
            ee.Image with SVF (0-1)
        """
        self._check_auth()

        # Simplified SVF using focal statistics
        mean_dem = dem.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.circle(radius, "meters"),
        )
        max_dem = dem.reduceNeighborhood(
            reducer=ee.Reducer.max(),
            kernel=ee.Kernel.circle(radius, "meters"),
        )

        # SVF approximation
        svf = mean_dem.subtract(dem).divide(max_dem.subtract(dem).add(0.01))
        svf = svf.clamp(0, 1).rename("SVF")

        return svf

    def compute_roughness_length(
        self,
        bldg_height: ee.Image,
        bldg_density: ee.Image,
        fvc: ee.Image,
    ) -> ee.Image:
        """
        Compute aerodynamic roughness length (Z0).

        Uses Grimmond & Oke (1999) empirical formula:
        Z0 = 0.1 * h_bldg * rho_bldg + 0.01 * rho_veg

        Args:
            bldg_height: Building height (m)
            bldg_density: Building density (0-1)
            fvc: Fractional vegetation cover (0-1)

        Returns:
            ee.Image with Z0 (m)
        """
        z0 = bldg_height.multiply(0.1).multiply(bldg_density) \
            .add(fvc.multiply(0.01)) \
            .rename("Z0")

        return z0

    def export_to_drive(
        self,
        image: ee.Image,
        filename: str,
        region: ee.Geometry,
        scale: int = 30,
        crs: str = "EPSG:32643",
        max_pixels: int = 1e13,
    ) -> ee.Task:
        """
        Export image to Google Drive.

        Args:
            image: Image to export
            filename: Output filename
            region: Export region
            scale: Spatial resolution in meters
            crs: Coordinate reference system
            max_pixels: Maximum number of pixels

        Returns:
            ee.Task object
        """
        self._check_auth()

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            folder="urban_heat_mitigation",
            region=region,
            scale=scale,
            crs=crs,
            maxPixels=max_pixels,
            fileFormat="GeoTIFF",
        )
        task.start()
        print(f"Export started: {filename}")
        return task

    def export_composite(
        self,
        filename: str = "hyderabad_composite",
        scale: int = 30,
    ) -> ee.Task:
        """
        Export full composite stack to Google Drive.

        This creates a multi-band GeoTIFF with all features.
        """
        region = self.get_region()
        composite = self.get_landsat_composite(region)
        eco = self.get_ecostress_lst(region)
        srtm = self.get_srtm_elevation(region)
        ghsl = self.get_ghsl_built(region)

        # Stack all bands
        full_stack = composite \
            .addBands(eco) \
            .addBands(srtm) \
            .addBands(ghsl)

        return self.export_to_drive(full_stack, filename, region, scale)


def get_gee_collections() -> Dict[str, str]:
    """Return dictionary of GEE collection IDs."""
    return {
        "landsat8_l2": "LANDSAT/LC08/C02/T1_L2",
        "landsat9_l2": "LANDSAT/LC09/C02/T1_L2",
        "ecostress_lst": "ECOSTRESS/ECO2LSTE/001",
        "sentinel2_sr": "COPERNICUS/S2_SR_HARMONIZED",
        "modis_lst": "MODIS/061/MYD11A1",
        "era5_land": "ECMWF/ERA5_LAND/HOURLY",
        "srtm": "USGS/SRTMGL1_003",
        "ghsl_built": "JRC/GHSL/P2023A/GHS_BUILT_S/2020",
        "gaul_boundary": "FAO/GAUL_SIMPLIFIED_500m/2015/level2",
    }
