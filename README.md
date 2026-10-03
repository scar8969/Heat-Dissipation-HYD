# 🏙️ Urban Heat Mitigation — Hyderabad

**Data-driven platform for mapping, modeling, and mitigating urban heat in Hyderabad.**

A Streamlit dashboard backed by a physics-informed deep-learning model (SegFormer-PINN) that fuses satellite, meteorological, and urban data to map land surface temperature (LST), detect heat hotspots, and optimize cooling interventions.

---

## ✦ What It Does

Hyderabad faces severe summer heat. This platform combines **remote sensing**, **ML**, and **scenario optimization** to:

- **Map land surface temperature** from Landsat, ECOSTRESS, MODIS, and Sentinel-2 at 30 m resolution.
- **Detect & classify heat hotspots** across the GHMC boundary.
- **Diagnose the surface energy balance (SEB)** — sensible heat, latent heat, ground flux, net radiation.
- **Rank the drivers** of urban heat (NDVI, NDBI, built-up density, albedo, …).
- **Simulate cooling interventions** — cool roofs, green roofs, street trees, parks, cool/permeable pavement, water bodies.
- **Optimize intervention placement** with NSGA-III multi-objective optimization and TOPSIS ranking across 500 zones and multiple budgets.

## ✦ Features

- **12-page interactive Streamlit dashboard** (dark theme) — Executive Dashboard, LST Explorer, Hotspot Detection, SEB Diagnostics, Driver Analysis, Intervention Simulator, NSGA-III Optimization, and more.
- **Physics-informed ML** — SegFormer-PINN with MiT-B5 encoder, CuboidAttention temporal fusion, SEB + flux heads, MC dropout uncertainty, and graph-based spatial context.
- **Multi-sensor data fusion** — Landsat 8/9, ECOSTRESS, MODIS, Sentinel-2, ERA5, GHSL, OSM, CPCB air quality, WUDAPT LCZ.
- **Spatial cross-validation** (5-fold) and uncertainty quantification.
- **Git LFS** for large raster and model artifacts.

## ✦ Tech Stack

| Layer        | Tech                                                        |
| ------------ | ----------------------------------------------------------- |
| Dashboard    | Streamlit + Plotly + Folium + geemap                         |
| Geospatial   | GeoPandas, Rasterio, Xarray, Riokarray, PyProj, Shapely, OSMnx |
| ML / DL      | PyTorch, Transformers, TIMM, Captum                          |
| Optimization | PyMOO (NSGA-III), Optuna, TOPSIS                             |
| Data         | Google Earth Engine, CDS API, NumPy, Pandas, SciPy           |
| Config       | Hydra + OmegaConf + YAML                                     |

## ✦ Project Structure

```
urban-heat-mitigation/
├── streamlit_app.py          # 12-page Streamlit dashboard
├── config/
│   └── config.yaml           # study area, data sources, model, training, optimization
├── src/
│   ├── data/                 # GEE interface, preprocessing, feature engineering, temporal fusion
│   ├── models/               # SegFormer-PINN, physics loss, SEB/flux heads, training
│   ├── analysis/             # hotspot detection, SEB diagnostics, driver & sensitivity analysis
│   ├── scenarios/            # interventions, simulation, NSGA-III optimization, TOPSIS
│   └── viz/                  # dashboard helpers & dark theme
├── notebooks/                # 01 data acquisition, 02 preprocessing, 04 training
├── data/
│   ├── raw/                  # GeoTIFFs, satellite & urban data (Git LFS)
│   ├── processed/            # NPZ patches, suitability masks (Git LFS)
│   └── splits/               # train/val/test index splits
├── outputs/
│   ├── models/               # best_model.pth, metrics.json, model_info.json (Git LFS)
│   └── maps/ rasters/ reports/ dashboard/ optimization/
├── tests/
├── requirements.txt
└── environment.yml
```

## ✦ Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local Streamlit URL (default `http://localhost:8501`).

> **Note:** Large data and model files (GeoTIFFs, NPZ patches, `.pth` checkpoints) are tracked with **Git LFS**. Clone with `git lfs pull` after `git clone` to fetch them.

### Environment

A conda environment is also provided:

```bash
conda env create -f environment.yml
conda activate urban-heat
streamlit run streamlit_app.py
```

## ✦ Data & Model Notes

- **Study area:** Hyderabad (GHMC), EPSG:32643, 30 m resolution, 2018–2024 summer months.
- **Model:** SegFormer-PINN (MiT-B5), 26 input channels, 256×256 patches, 99 patches / 69 train / 14 val / 16 test.
- **Target:** Land Surface Temperature (LST).
- **Optimization:** NSGA-III across 500 zones, budgets ₹10 / ₹50 / ₹100 Cr, TOPSIS ranking.
- **Evaluation targets:** RMSE ≤ 1.5 °C, MAE ≤ 1.2 °C, R² ≥ 0.85.

## ✦ License

[MIT](./LICENSE)