"""
Data loading and analysis helpers for the Streamlit dashboard.

Loads real data from GeoTIFFs and NPZ patches,
runs analysis pipelines, and caches results.
"""

import os
import sys
import yaml
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.hotspots import HotspotDetector, ZoneAggregator, HotspotClassifier
from src.analysis.seb_diagnostics import SEBDiagnostics
from src.analysis.drivers import DriverAnalyzer
from src.analysis.sensitivity import SensitivityAnalyzer
from src.scenarios.interventions import INTERVENTIONS, InterventionPlacer
from src.scenarios.simulate import ScenarioSimulator
from src.scenarios.optimize import NSGA3Optimizer, OptimizationConfig
from src.scenarios.topsis import TOPSISRanker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PATCHES = PROJECT_ROOT / "data" / "processed" / "patches"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


# ── Theme constants ────────────────────────────────────────────────────
DARK_THEME = {
    "bg": "#031427",
    "surface": "#0A1E3D",
    "surface2": "#112A4A",
    "border": "#1E3456",
    "accent": "#FF6B35",
    "accent2": "#FF8C5A",
    "text": "#E8EDF5",
    "text_muted": "#8892A4",
    "success": "#00C853",
    "warning": "#FFB300",
    "error": "#FF5252",
    "info": "#448AFF",
}


@st.cache_resource
def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_raster(path: str) -> Tuple[np.ndarray, Dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Raster not found: {path}")
    with rasterio.open(path) as src:
        data = src.read()
        meta = src.meta
    if data.shape[0] == 1:
        data = data[0]
    return data, meta


@st.cache_resource
def load_all_data() -> Dict[str, Any]:
    data = {}
    data["lst"], data["lst_meta"] = load_raster(str(DATA_RAW / "lst_hyderabad.tif"))
    data["features"], data["features_meta"] = load_raster(str(DATA_RAW / "features_hyderabad.tif"))
    data["met_forcing"], data["met_meta"] = load_raster(str(DATA_RAW / "met_forcing_hyderabad.tif"))
    data["flux_rn"], data["flux_rn_meta"] = load_raster(str(DATA_RAW / "flux_rn.tif"))
    data["flux_h"], data["flux_h_meta"] = load_raster(str(DATA_RAW / "flux_h.tif"))
    data["flux_le"], data["flux_le_meta"] = load_raster(str(DATA_RAW / "flux_le.tif"))
    data["flux_g"], data["flux_g_meta"] = load_raster(str(DATA_RAW / "flux_g.tif"))
    data["hotspots_gt"], _ = load_raster(str(DATA_RAW / "hotspots_groundtruth.tif"))
    return data


def get_feature_names() -> List[str]:
    return [
        "LST", "NDVI", "NDBI", "MNDWI", "NDBaI", "ALBEDO", "EMISSIVITY",
        "FVC", "LAI", "ELEVATION", "SLOPE", "ASPECT", "SVF", "Z0",
        "BUILDING_DENSITY", "BUILDING_HEIGHT", "ROAD_DENSITY", "LCZ",
    ]


def get_met_names() -> List[str]:
    return ["SW_down", "LW_down", "T_air", "RH", "Wind"]


def get_intervention_names() -> Dict[str, str]:
    return {k: v.name for k, v in INTERVENTIONS.items()}


@st.cache_resource
def run_hotspot_detection(lst: np.ndarray) -> Dict:
    detector = HotspotDetector()
    result = detector.detect(lst)
    return result


@st.cache_resource
def run_hotspot_classification(
    lst: np.ndarray,
    features: np.ndarray,
    hotspot_result: Dict,
) -> List[Dict]:
    labeled = hotspot_result["labeled_map"]
    stats = hotspot_result["statistics"]
    valid_labels = [h["id"] for h in stats["hotspots"]]
    classifier = HotspotClassifier()
    ndvi = features[1] if features.shape[0] > 1 else np.zeros_like(lst)
    ndbi = features[2] if features.shape[0] > 2 else np.zeros_like(lst)
    mndwi = features[3] if features.shape[0] > 3 else np.zeros_like(lst)
    svf = features[12] if features.shape[0] > 12 else np.zeros_like(lst)
    bd = features[14] if features.shape[0] > 14 else np.zeros_like(lst)
    return classifier.classify(lst, ndvi, ndbi, mndwi, svf, bd, labeled, valid_labels)


@st.cache_resource
def run_seb_diagnostics(
    fluxes: Dict[str, np.ndarray],
    lst: np.ndarray,
) -> Dict:
    diagnostics = SEBDiagnostics()
    return diagnostics.diagnose(fluxes, lst)


@st.cache_resource
def run_driver_importance(
    features: np.ndarray,
    lst: np.ndarray,
    hotspot_mask: np.ndarray,
) -> Dict[str, float]:
    device = torch.device("cpu")
    dummy_model = DummyModel()
    analyzer = DriverAnalyzer(dummy_model, device)
    return analyzer.analyze_hotspot_drivers(lst, features, hotspot_mask)


@st.cache_resource
def run_zone_aggregation(
    lst: np.ndarray,
    features: np.ndarray,
    n_zones: int = 500,
) -> Dict:
    lulc = features[17].astype(int) if features.shape[0] > 17 else np.zeros_like(lst, dtype=int)
    bd = features[14] if features.shape[0] > 14 else np.zeros_like(lst)
    aggregator = ZoneAggregator(n_zones=n_zones)
    return aggregator.fit(lst, lulc, bd)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, met_forcing=None):
        return {"LST": x.mean(dim=1, keepdim=True)}


@st.cache_resource
def compute_topsis_ranking(scenarios: List[Dict]) -> List[Dict]:
    ranker = TOPSISRanker()
    return ranker.rank(scenarios)


# ── Plotting utilities ────────────────────────────────────────────────

def make_dark_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#0A1E3D",
        plot_bgcolor="#0A1E3D",
        font=dict(color="#E8EDF5", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#1E3456", zerolinecolor="#1E3456"),
        yaxis=dict(gridcolor="#1E3456", zerolinecolor="#1E3456"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Styled metric card using Streamlit columns."""
    st.markdown(
        f"""
        <div style="background:#0A1E3D; border:1px solid #1E3456; border-radius:8px; padding:16px; margin-bottom:8px;">
            <div style="color:#8892A4; font-family:Inter; font-size:12px; text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
            <div style="color:#E8EDF5; font-family:JetBrains Mono; font-size:28px; font-weight:600; margin-top:4px;">{value}</div>
            {f'<div style="color:#00C853; font-family:Inter; font-size:13px; margin-top:2px;">▲ {delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = None):
    st.markdown(
        f"""
        <div style="margin-bottom:24px;">
            <h2 style="font-family:Hanken Grotesk; font-weight:600; color:#E8EDF5; margin-bottom:4px;">{title}</h2>
            {f'<p style="color:#8892A4; font-family:Inter; font-size:14px;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown(
        "<hr style='border-color:#1E3456; margin:24px 0;'>",
        unsafe_allow_html=True,
    )
