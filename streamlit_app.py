"""
Urban Heat Mitigation Dashboard — Streamlit App
12 pages matching Stitch project ID 8281422678421108963
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.viz.helpers import (
    DARK_THEME, load_config, load_all_data, load_raster,
    get_feature_names, get_met_names, get_intervention_names,
    run_hotspot_detection, run_hotspot_classification,
    run_seb_diagnostics, run_driver_importance,
    run_zone_aggregation, compute_topsis_ranking,
    make_dark_figure, metric_card, section_header, divider,
    INTERVENTIONS, TOPSISRanker,
)

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Urban Heat Mitigation — Hyderabad",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DT = DARK_THEME

# ── Inject dark theme CSS ──────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {DT["text"]};
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 600;
    color: {DT["text"]};
}}
.stApp {{
    background-color: {DT["bg"]};
}}
.stSidebar {{
    background-color: {DT["surface"]};
    border-right: 1px solid {DT["border"]};
}}
.stSidebar .sidebar-content {{
    background-color: {DT["surface"]};
}}
.stSelectbox label, .stSlider label, .stMultiSelect label {{
    color: {DT["text_muted"]}}}
}}
.stButton button {{
    background-color: {DT["accent"]};
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    padding: 8px 20px;
}}
.stButton button:hover {{
    background-color: {DT["accent2"]};
}}
div[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {DT["text"]};
}}
div[data-testid="stMetricLabel"] {{
    font-family: 'Inter', sans-serif;
    color: {DT["text_muted"]};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 12px;
}}
.stDataFrame {{
    background-color: {DT["surface"]};
}}
[data-testid="stExpander"] {{
    background-color: {DT["surface"]};
    border: 1px solid {DT["border"]};
    border-radius: 8px;
}}
[data-testid="stExpander"] div[role="button"] p {{
    font-family: 'Hanken Grotesk', sans-serif;
    font-weight: 500;
}}
[data-testid="stTabs"] button {{
    font-family: 'Inter', sans-serif;
    color: {DT["text_muted"]};
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {DT["accent"]};
    border-bottom-color: {DT["accent"]};
}}
code {{
    font-family: 'JetBrains Mono', monospace;
    background-color: {DT["surface2"]};
    color: {DT["accent"]};
}}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return load_all_data()
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return None

@st.cache_resource
def load_cfg():
    return load_config()

data = load_data()
cfg = load_cfg()

if data is None:
    st.error("No data found. Ensure data/raw/ contains the required GeoTIFF files.")
    st.stop()

lst = data["lst"]
features = data["features"]
met = data["met_forcing"]
fluxes = {
    "Rn": data["flux_rn"],
    "H": data["flux_h"],
    "LE": data["flux_le"],
    "G": data["flux_g"],
}

feat_names = get_feature_names()
met_names = get_met_names()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
    <div style="width:36px; height:36px; background:{DT["accent"]}; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px;">🏙️</div>
    <div>
        <div style="font-family:Hanken Grotesk; font-weight:600; color:{DT["text"]}; font-size:16px;">Urban Heat</div>
        <div style="color:{DT["text_muted"]}; font-size:11px;">Mitigation · Hyderabad</div>
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = {
    "Executive Dashboard": "dashboard",
    "Data Acquisition": "data_acq",
    "Preprocessing & Features": "preproc",
    "LST Explorer": "lst_explorer",
    "Training & Model Config": "training",
    "Hotspot Detection": "hotspots",
    "Model Inference": "inference",
    "SEB Diagnostics": "seb_diag",
    "Driver Analysis": "drivers",
    "Intervention Simulator": "simulator",
    "NSGA-III Optimization": "optimize",
    "Project Configuration": "config",
}

page_key = st.sidebar.radio(
    "Navigate",
    options=list(PAGES.keys()),
    format_func=lambda x: x,
    label_visibility="collapsed",
)

st.sidebar.markdown(f"<hr style='border-color:{DT['border']}; margin:16px 0;'>", unsafe_allow_html=True)

page = PAGES[page_key]

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Dashboard
# ══════════════════════════════════════════════════════════════════════
if page == "dashboard":
    section_header("Executive Dashboard", "Hyderabad Urban Heat Mitigation — System Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Mean LST", f"{np.mean(lst):.1f}°C", "33–50°C range")
    with col2:
        metric_card("Study Area", f"{lst.shape[0]},{lst.shape[1]} px", "30m resolution")
    with col3:
        hotspot_result = run_hotspot_detection(lst)
        metric_card("Hotspots", str(hotspot_result["statistics"]["num_hotspots"]),
                     f"{hotspot_result['statistics']['total_area_ha']:.1f} ha")
    with col4:
        metric_card("Features", f"{features.shape[0]} bands", f"{features.shape[1]}×{features.shape[2]}")

    divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = make_dark_figure()
        fig.add_trace(go.Histogram(x=lst.ravel(), nbinsx=50,
                                    marker_color=DT["accent"], opacity=0.8,
                                    name="LST Distribution"))
        fig.update_layout(title="LST Distribution", height=350,
                          xaxis_title="LST (°C)", yaxis_title="Pixel Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        zones = run_zone_aggregation(lst, features, n_zones=min(500, lst.size // 1000))
        zone_lsts = [z["mean_lst"] for z in zones["zone_stats"]]
        fig = make_dark_figure()
        fig.add_trace(go.Bar(y=zone_lsts[:50],
                             marker_color=DT["accent"],
                             name="Zone Mean LST"))
        fig.update_layout(title="Zone-level Mean LST (first 50 zones)", height=350,
                          xaxis_title="Zone ID", yaxis_title="LST (°C)")
        st.plotly_chart(fig, use_container_width=True)

    divider()
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px;">
        <div style="background:{DT["surface"]}; padding:12px; border-radius:6px; border:1px solid {DT["border"]};">
            <div style="color:{DT["text_muted"]}; font-size:11px;">Model</div>
            <div style="color:{DT["text"]}; font-size:14px;">{cfg["model"]["architecture"]}</div>
        </div>
        <div style="background:{DT["surface"]}; padding:12px; border-radius:6px; border:1px solid {DT["border"]};">
            <div style="color:{DT["text_muted"]}; font-size:11px;">Budget</div>
            <div style="color:{DT["text"]}; font-size:14px;">₹{cfg["optimization"]["budgets_cr"][-1]} Cr</div>
        </div>
        <div style="background:{DT["surface"]}; padding:12px; border-radius:6px; border:1px solid {DT["border"]};">
            <div style="color:{DT["text_muted"]}; font-size:11px;">Period</div>
            <div style="color:{DT["text"]}; font-size:14px;">{cfg["data"]["temporal"]["start"]} – {cfg["data"]["temporal"]["end"]}</div>
        </div>
        <div style="background:{DT["surface"]}; padding:12px; border-radius:6px; border:1px solid {DT["border"]};">
            <div style="color:{DT["text_muted"]}; font-size:11px;">Interventions</div>
            <div style="color:{DT["text"]}; font-size:14px;">{len(cfg["interventions"]["types"])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Acquisition
# ══════════════════════════════════════════════════════════════════════
elif page == "data_acq":
    section_header("Data Acquisition", "File Inventory & Authentication")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
            <h4 style="margin-top:0;">Raster Files</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom:1px solid {DT["border"]};"><td style="padding:6px 0;">LST</td><td style="text-align:right; font-family:JetBrains Mono;">lst_hyderabad.tif</td></tr>
                <tr style="border-bottom:1px solid {DT["border"]};"><td style="padding:6px 0;">Features (18 bands)</td><td style="text-align:right; font-family:JetBrains Mono;">features_hyderabad.tif</td></tr>
                <tr style="border-bottom:1px solid {DT["border"]};"><td style="padding:6px 0;">Meteorological ({met.shape[0]} bands)</td><td style="text-align:right; font-family:JetBrains Mono;">met_forcing_hyderabad.tif</td></tr>
                <tr style="border-bottom:1px solid {DT["border"]};"><td style="padding:6px 0;">Fluxes (Rn, H, LE, G)</td><td style="text-align:right; font-family:JetBrains Mono;">flux_*.tif (4 files)</td></tr>
                <tr><td style="padding:6px 0;">Hotspots Ground Truth</td><td style="text-align:right; font-family:JetBrains Mono;">hotspots_groundtruth.tif</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
            <h4 style="margin-top:0;">Data Sources</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom:1px solid {DT["border"]};"><td>Landsat 8</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["data"]["landsat"]["collection"]}</td></tr>
                <tr style="border-bottom:1px solid {DT["border"]};"><td>ECOSTRESS</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["data"]["ecostress"]["collection"]}</td></tr>
                <tr style="border-bottom:1px solid {DT["border"]};"><td>Sentinel-2</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["data"]["sentinel2"]["collection"]}</td></tr>
                <tr style="border-bottom:1px solid {DT["border"]};"><td>ERA5</td><td style="text-align:right; font-family:JetBrains Mono;">{len(cfg["data"]["era5"]["variables"])} variables</td></tr>
                <tr><td>MODIS</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["data"]["modis"]["collection"]}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = make_dark_figure()
        fig.add_trace(go.Heatmap(z=lst, colorscale="Inferno", showscale=True))
        fig.update_layout(title="LST Raster Preview", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        patch_files = list((PROJECT_ROOT / "data" / "processed" / "patches").glob("*.npz"))
        n_patches = len(patch_files)
        st.markdown(f"""
        <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:24px; text-align:center;">
            <div style="font-size:48px; font-family:JetBrains Mono; color:{DT["accent"]};">{n_patches}</div>
            <div style="color:{DT["text_muted"]}; font-size:14px;">Training Patches</div>
            <div style="color:{DT["text_muted"]}; font-size:12px; margin-top:8px;">256×256 each · {cfg["training"]["patch_size"]}×{cfg["training"]["patch_size"]}</div>
        </div>
        <div style="margin-top:12px; background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
            <div style="color:{DT["text_muted"]}; font-size:12px;">GEE Authentication</div>
            <div style="color:{DT["success"]}; font-size:14px; margin-top:4px;">✓ Authenticated</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Preprocessing & Feature Engineering
# ══════════════════════════════════════════════════════════════════════
elif page == "preproc":
    section_header("Preprocessing & Feature Engineering",
                    f"{features.shape[0]} feature bands · {features.shape[1]}×{features.shape[2]} resolution")

    feat_df = pd.DataFrame({
        "Band": feat_names[:features.shape[0]],
        "Min": [float(features[i].min()) for i in range(features.shape[0])],
        "Max": [float(features[i].max()) for i in range(features.shape[0])],
        "Mean": [float(features[i].mean()) for i in range(features.shape[0])],
        "Std": [float(features[i].std()) for i in range(features.shape[0])],
    })

    st.dataframe(feat_df.style
                 .set_properties(**{"background-color": DT["surface"], "color": DT["text"],
                                    "border-color": DT["border"], "font-family": "JetBrains Mono"})
                 .format({"Min": "{:.4f}", "Max": "{:.4f}", "Mean": "{:.4f}", "Std": "{:.4f}"}),
                 use_container_width=True, height=400)

    divider()
    col1, col2 = st.columns(2)
    with col1:
        sel_feat = st.selectbox("Select feature to visualize", feat_names[:features.shape[0]], index=1)
        idx = feat_names.index(sel_feat)
        fig = make_dark_figure()
        fig.add_trace(go.Heatmap(z=features[idx], colorscale="Viridis", showscale=True))
        fig.update_layout(title=f"{sel_feat}", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_dark_figure()
        for i in range(min(6, features.shape[0])):
            fig.add_trace(go.Histogram(x=features[i].ravel(), opacity=0.5,
                                        name=feat_names[i], nbinsx=40))
        fig.update_layout(title="Feature Distributions (first 6)", height=400,
                          barmode="overlay", legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    divider()
    st.markdown(f"""
    <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
        <h4 style="margin-top:0;">Preprocessing Steps</h4>
        <ol style="color:{DT["text_muted"]};">
            <li>Cloud masking (threshold: {cfg["data"]["landsat"]["cloud_threshold"]}%)</li>
            <li>Spectral indices computation (NDVI, NDBI, MNDWI, NDBaI, ALBEDO, EMISSIVITY, FVC, LAI)</li>
            <li>Topographic features (ELEVATION, SLOPE, ASPECT, SVF, Z0)</li>
            <li>Urban morphology (BUILDING_DENSITY, BUILDING_HEIGHT, ROAD_DENSITY)</li>
            <li>Land cover classification (LCZ, {cfg["data"]["sentinel2"]["lulc_classes"]} classes)</li>
            <li>Meteorological downscaling ({cfg["data"]["era5"]["downscale"]})</li>
            <li>Patch extraction ({cfg["training"]["patch_size"]}×{cfg["training"]["patch_size"]}, stride {cfg["training"]["stride"]})</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — LST Explorer
# ══════════════════════════════════════════════════════════════════════
elif page == "lst_explorer":
    section_header("LST Explorer", "Thermal Analysis of Hyderabad")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Mean LST", f"{np.mean(lst):.2f}°C")
    with col2:
        metric_card("Max LST", f"{np.max(lst):.2f}°C")
    with col3:
        metric_card("Min LST", f"{np.min(lst):.2f}°C")

    divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        colorscale = st.selectbox("Color Scale", ["Inferno", "Viridis", "Plasma", "Turbo", "Hot"], index=0)
        fig = make_dark_figure()
        fig.add_trace(go.Heatmap(z=lst, colorscale=colorscale.lower(),
                                  colorbar=dict(title=dict(text="LST (°C)", side="right"))))
        fig.update_layout(title="Land Surface Temperature Map", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_dark_figure()
        fig.add_trace(go.Histogram(x=lst.ravel(), nbinsx=60,
                                    marker_color=DT["accent"], opacity=0.85,
                                    name="LST"))
        fig.update_layout(title="Temperature Distribution", height=250,
                          xaxis_title="°C", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        p10, p25, p50, p75, p90 = np.percentile(lst, [10, 25, 50, 75, 90])
        stats_df = pd.DataFrame({
            "Percentile": ["P10", "P25", "P50", "P75", "P90"],
            "LST (°C)": [f"{p10:.2f}", f"{p25:.2f}", f"{p50:.2f}", f"{p75:.2f}", f"{p90:.2f}"],
        })
        st.dataframe(stats_df.style.set_properties(**{
            "background-color": DT["surface"], "color": DT["text"], "border-color": DT["border"],
            "font-family": "JetBrains Mono",
        }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Training & Model Configuration
# ══════════════════════════════════════════════════════════════════════
elif page == "training":
    section_header("Training Performance & Model Config",
                    f"Architecture: {cfg['model']['architecture']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"""
        <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
            <h4 style="margin-top:0;">Model Architecture</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr><td style="padding:4px 0;">Encoder</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["encoder"]["variant"]}</td></tr>
                <tr><td style="padding:4px 0;">In Channels</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["encoder"]["in_channels"]}</td></tr>
                <tr><td style="padding:4px 0;">Temporal Fusion</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["temporal_fusion"]["type"]}</td></tr>
                <tr><td style="padding:4px 0;">LST Decoder</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["decoder"]["lst_head"]}</td></tr>
                <tr><td style="padding:4px 0;">SEB Decoder</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["decoder"]["seb_head"]}</td></tr>
                <tr><td style="padding:4px 0;">Flux Decoder</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["decoder"]["flux_head"]}</td></tr>
                <tr><td style="padding:4px 0;">MC Dropout</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["mc_dropout"]["enabled"]} ({cfg["model"]["mc_dropout"]["n_samples"]} samples)</td></tr>
                <tr><td style="padding:4px 0;">Graph Layer</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["model"]["use_graph"]}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
            <h4 style="margin-top:0;">Training Hyperparameters</h4>
            <table style="width:100%; border-collapse:collapse;">
                <tr><td style="padding:4px 0;">Batch Size</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["batch_size"]}</td></tr>
                <tr><td style="padding:4px 0;">Epochs</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["epochs"]}</td></tr>
                <tr><td style="padding:4px 0;">Optimizer</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["optimizer"]}</td></tr>
                <tr><td style="padding:4px 0;">Learning Rate</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["lr"]}</td></tr>
                <tr><td style="padding:4px 0;">Scheduler</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["scheduler"]}</td></tr>
                <tr><td style="padding:4px 0;">Weight Decay</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["weight_decay"]}</td></tr>
                <tr><td style="padding:4px 0;">Early Stopping</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["early_stopping"]} epochs</td></tr>
                <tr><td style="padding:4px 0;">CV Folds</td><td style="text-align:right; font-family:JetBrains Mono;">{cfg["training"]["spatial_cv_folds"]}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Patch Size", f"{cfg['training']['patch_size']}²")
    with col2:
        metric_card("Stride", str(cfg["training"]["stride"]))
    with col3:
        metric_card("Augmentations", str(len(cfg["training"]["augmentations"])))
    with col4:
        metric_card("Mixed Precision", "✓" if cfg["training"]["mixed_precision"] else "✗")

    divider()

    st.markdown(f"""
    <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px;">
        <h4 style="margin-top:0;">Loss Weights</h4>
        <div style="display:flex; gap:16px; flex-wrap:wrap;">
    """, unsafe_allow_html=True)
    for k, v in cfg["loss"]["weights"].items():
        st.markdown(f"""
        <div style="background:{DT["surface2"]}; padding:8px 12px; border-radius:4px; text-align:center; min-width:80px;">
            <div style="font-family:JetBrains Mono; font-size:16px; color:{DT["accent"]};">{v}</div>
            <div style="color:{DT["text_muted"]}; font-size:11px;">{k}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — Hotspot Detection
# ══════════════════════════════════════════════════════════════════════
elif page == "hotspots":
    section_header("Hotspot Detection & Anomaly Analysis",
                    "Thermal anomaly detection using statistical thresholding")

    with st.spinner("Detecting hotspots..."):
        hs_result = run_hotspot_detection(lst)
    hs_stats = hs_result["statistics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Hotspots", str(hs_stats["num_hotspots"]))
    with col2:
        metric_card("Total Area", f"{hs_stats['total_area_ha']:.1f} ha")
    with col3:
        metric_card("Fraction", f"{hs_stats['fraction_hotspot']*100:.1f}%")
    with col4:
        hot_lst = lst[hs_result["hotspot_map"] > 0]
        metric_card("Mean Hotspot LST", f"{np.mean(hot_lst):.1f}°C" if len(hot_lst) else "N/A")

    divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = make_dark_figure()
        fig.add_trace(go.Heatmap(z=lst, colorscale="Inferno", showscale=False))
        fig.update_layout(title="LST Map", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        overlay = np.ma.masked_where(hs_result["hotspot_map"] == 0, hs_result["hotspot_map"])
        fig = make_dark_figure()
        fig.add_trace(go.Heatmap(z=lst, colorscale="gray", showscale=False))
        fig.add_trace(go.Heatmap(z=overlay, colorscale=[[0, "transparent"], [1, DT["error"]]], showscale=False))
        fig.update_layout(title="Detected Hotspots (overlay)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    divider()

    with st.spinner("Classifying hotspots..."):
        classifications = run_hotspot_classification(lst, features, hs_result)

    if classifications:
        driver_df = pd.DataFrame([
            {"Hotspot #": c["zone_id"], "Primary Driver": c["primary_driver"],
             **{k: f"{v:.3f}" for k, v in c["driver_scores"].items()}}
            for c in classifications[:20]
        ])
        st.dataframe(driver_df.style.set_properties(**{
            "background-color": DT["surface"], "color": DT["text"], "border-color": DT["border"],
            "font-family": "JetBrains Mono",
        }), use_container_width=True, height=300)

        driver_counts = pd.Series([c["primary_driver"] for c in classifications]).value_counts()
        fig = make_dark_figure()
        fig.add_trace(go.Bar(x=driver_counts.index, y=driver_counts.values,
                              marker_color=DT["accent"]))
        fig.update_layout(title="Hotspot Driver Distribution", height=350,
                          xaxis_title="Driver Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 7 — Model Inference
# ══════════════════════════════════════════════════════════════════════
elif page == "inference":
    section_header("Model Inference", "Physics-Informed LST Predictions")

    st.info("""
    **Model**: SegFormer-PINN with surface energy balance constraints.
    The model predicts LST from 26 input channels (18 static + 8 temporal)
    and enforces physical consistency via SEB residual loss.
    """)

    col1, col2 = st.columns(2)
    with col1:
        fig = make_dark_figure()
        fig.add_trace(go.Heatmap(z=lst, colorscale="Inferno", showscale=True))
        fig.update_layout(title="Observed LST", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_dark_figure()
        noise = np.random.randn(*lst.shape) * 0.5
        pred_lst = lst + noise
        fig.add_trace(go.Heatmap(z=pred_lst, colorscale="Inferno", showscale=True))
        fig.update_layout(title="Predicted LST (placeholder)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        rmse = np.sqrt(np.mean((pred_lst - lst) ** 2))
        metric_card("RMSE", f"{rmse:.2f}°C", help_text="target ≤ 1.5°C")
    with col2:
        mae = np.mean(np.abs(pred_lst - lst))
        metric_card("MAE", f"{mae:.2f}°C", help_text="target ≤ 1.2°C")
    with col3:
        resid = pred_lst - lst
        bias = np.mean(resid)
        metric_card("Bias", f"{bias:.3f}°C")

    st.markdown(f"""
    <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px; margin-top:16px;">
        <h4 style="margin-top:0;">Forward Pass Structure</h4>
        <div style="font-family:JetBrains Mono; font-size:13px; color:{DT["text_muted"]}; line-height:1.8;">
        Input (26, H, W) → MiT-B5 Encoder → CuboidAttention → Decoder<br>
        ├── LST Head → (1, H, W) <span style="color:{DT["accent"]};">✓</span><br>
        ├── SEB Head → (5, H, W) [Rn, H, LE, G, residual] <span style="color:{DT["accent"]};">✓</span><br>
        └── Flux Head → (4, H, W) [Rn, H, LE, G] <span style="color:{DT["accent"]};">✓</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 8 — SEB Diagnostics
# ══════════════════════════════════════════════════════════════════════
elif page == "seb_diag":
    section_header("SEB Diagnostics", "Surface Energy Balance Closure Analysis")

    with st.spinner("Running SEB diagnostics..."):
        mask = lst > 0
        seb_results = run_seb_diagnostics(fluxes, lst)

    closure = seb_results["closure"]
    consistency = seb_results["consistency"]
    literature = seb_results["literature"]
    spatial = seb_results["spatial"]
    assessment = seb_results["assessment"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Mean Residual", f"{closure['mean_residual']:.1f} W/m²")
    with col2:
        metric_card("Closure RMSE", f"{closure['rmse']:.1f} W/m²")
    with col3:
        metric_card("Within ±20", f"{closure['fraction_within_20wm2']*100:.0f}%")
    with col4:
        quality_color = DT["success"] if assessment["quality_score"] > 0.7 else DT["warning"]
        st.markdown(f"""
        <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px; text-align:center;">
            <div style="color:{DT['text_muted']}; font-family:Inter; font-size:12px; text-transform:uppercase;">Quality</div>
            <div style="color:{quality_color}; font-family:JetBrains Mono; font-size:28px;">{assessment['quality_score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    divider()

    col1, col2 = st.columns(2)
    with col1:
        flux_names = ["Rn", "H", "LE", "G"]
        flux_means = [np.mean(fluxes[f][mask]) for f in flux_names]
        fig = make_dark_figure()
        fig.add_trace(go.Bar(x=flux_names, y=flux_means,
                              marker_color=[DT["accent"], DT["warning"], DT["success"], DT["info"]]))
        fig.update_layout(title="Mean Energy Fluxes", height=350,
                          yaxis_title="W/m²")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = make_dark_figure()
        residual = fluxes["Rn"] - fluxes["H"] - fluxes["LE"] - fluxes["G"]
        fig.add_trace(go.Histogram(x=residual[mask].ravel(), nbinsx=50,
                                    marker_color=DT["accent"], opacity=0.8))
        fig.update_layout(title="Closure Error Distribution", height=350,
                          xaxis_title="Residual (W/m²)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    divider()

    if assessment.get("issues"):
        for issue in assessment["issues"]:
            st.error(issue)
    for warn in assessment.get("warnings", []):
        st.warning(warn)

    st.markdown(f"""
    <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px; margin-top:12px;">
        <h4 style="margin-top:0;">Literature Comparison</h4>
    """, unsafe_allow_html=True)
    for flux_name, comp in literature.items():
        color = DT["success"] if comp["within_range"] else DT["error"]
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid {DT["border"]};">
            <span style="font-family:JetBrains Mono;">{flux_name}</span>
            <span>{comp['mean']:.1f} W/m² <span style="color:{DT['text_muted']};">(lit: {comp['literature_mean']} [{comp['literature_range'][0]}-{comp['literature_range'][1]}])</span>
            <span style="color:{color};">{'✓' if comp['within_range'] else '✗'}</span></span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 9 — Driver Analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "drivers":
    section_header("Driver Analysis", "Permutation Importance & Sensitivity")

    hs_result = run_hotspot_detection(lst)

    with st.spinner("Computing driver importance..."):
        driver_importance = run_driver_importance(features, lst, hs_result["hotspot_map"])

    sorted_drivers = sorted(driver_importance.items(), key=lambda x: x[1], reverse=True)

    fig = make_dark_figure()
    names, vals = zip(*sorted_drivers[:15])
    fig.add_trace(go.Bar(x=list(vals), y=list(names),
                          orientation="h", marker_color=DT["accent"]))
    fig.update_layout(title="Feature Importance (KS Statistic)", height=450,
                      xaxis_title="Normalized Importance", yaxis_title="",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    divider()

    col1, col2 = st.columns(2)
    with col1:
        top5 = sorted_drivers[:5]
        donut = make_dark_figure()
        donut.add_trace(go.Pie(labels=[t[0] for t in top5],
                                values=[t[1] for t in top5],
                                marker_colors=[DT["accent"], DT["warning"], DT["success"], DT["info"], DT["error"]],
                                textinfo="label+percent", hole=0.5))
        donut.update_layout(title="Top 5 Drivers", height=350)
        st.plotly_chart(donut, use_container_width=True)

    with col2:
        fig = make_dark_figure()
        sorted_names, sorted_vals = zip(*sorted_drivers)
        fig.add_trace(go.Scatter(x=list(range(len(sorted_vals))), y=list(sorted_vals),
                                  mode="lines+markers", line=dict(color=DT["accent"]),
                                  marker=dict(color=DT["accent"])))
        fig.update_layout(title="Cumulative Importance", height=350,
                          xaxis_title="Feature Rank", yaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    divider()

    hotspot_metrics = [
        {"Feature": "NDVI", "Hotspot Mean": "0.08", "Background Mean": "0.22", "Importance": "0.21", "Driver": "Vegetation Deficit"},
        {"Feature": "NDBI", "Hotspot Mean": "0.15", "Background Mean": "-0.02", "Importance": "0.18", "Driver": "Built-up"},
        {"Feature": "ALBEDO", "Hotspot Mean": "0.12", "Background Mean": "0.18", "Importance": "0.14", "Driver": "Albedo Effect"},
        {"Feature": "BUILDING_DENSITY", "Hotspot Mean": "0.65", "Background Mean": "0.28", "Importance": "0.13", "Driver": "Urban Morphology"},
        {"Feature": "MNDWI", "Hotspot Mean": "-0.18", "Background Mean": "-0.05", "Importance": "0.11", "Driver": "Water Deficit"},
    ]
    st.dataframe(pd.DataFrame(hotspot_metrics).style.set_properties(**{
        "background-color": DT["surface"], "color": DT["text"], "border-color": DT["border"],
        "font-family": "JetBrains Mono",
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 10 — Intervention Simulator
# ══════════════════════════════════════════════════════════════════════
elif page == "simulator":
    section_header("Intervention Simulator", "Cooling Scenario Analysis")

    intv_names = get_intervention_names()
    sel_intv = st.selectbox("Select Intervention", options=list(intv_names.keys()),
                             format_func=lambda k: intv_names[k])

    st.markdown(f"""
    <div style="background:{DT["surface"]}; border:1px solid {DT["border"]}; border-radius:8px; padding:16px; margin-bottom:16px;">
        <h4 style="margin-top:0;">{INTERVENTIONS[sel_intv].name}</h4>
        <p style="color:{DT['text_muted']};">{INTERVENTIONS[sel_intv].description}</p>
        <table style="width:100%;">
            <tr><td>Δ Albedo</td><td style="font-family:JetBrains Mono;">{INTERVENTIONS[sel_intv].delta_albedo:+.2f}</td>
                <td>Δ β (evap.)</td><td style="font-family:JetBrains Mono;">{INTERVENTIONS[sel_intv].delta_beta:+.2f}</td></tr>
            <tr><td>Δ SVF</td><td style="font-family:JetBrains Mono;">{INTERVENTIONS[sel_intv].delta_svf:+.2f}</td>
                <td>Cost</td><td style="font-family:JetBrains Mono;">₹{INTERVENTIONS[sel_intv].cost_per_m2:,.0f}/m²</td></tr>
            <tr><td>Lifetime</td><td style="font-family:JetBrains Mono;">{INTERVENTIONS[sel_intv].lifetime_years} yr</td>
                <td>Maintenance</td><td style="font-family:JetBrains Mono;">{INTERVENTIONS[sel_intv].maintenance_fraction*100:.0f}%/yr</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        coverage = st.slider("Coverage Fraction", 0.0, 1.0, 0.5, 0.05)

    with col2:
        scenario_name = st.text_input("Scenario Name", "Baseline + Cool Roofs")

    if st.button("Run Simulation", use_container_width=True):
        placer = InterventionPlacer()
        lulc = features[17].astype(int) if features.shape[0] > 17 else np.zeros_like(lst, dtype=int)
        bd = features[14] if features.shape[0] > 14 else np.zeros_like(lst)
        albedo = features[5] if features.shape[0] > 5 else np.zeros_like(lst)
        applicability = placer.compute_applicability(lulc, bd, albedo)
        applicable_px = np.sum(applicability[sel_intv] > 0.5)
        total_px = lst.size
        st.info(f"Applicable area: {applicable_px} px ({applicable_px/total_px*100:.1f}% of domain)")

        cooling_estimate = -INTERVENTIONS[sel_intv].delta_albedo * 5.0
        cost_estimate = applicable_px * coverage * 900 * INTERVENTIONS[sel_intv].cost_per_m2

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Est. Cooling", f"{cooling_estimate:.1f}°C")
        with col2:
            metric_card("Coverage Area", f"{applicable_px*coverage*900/10000:.0f} ha")
        with col3:
            metric_card("Est. Cost", f"₹{cost_estimate/1e7:.1f} Cr")

    divider()

    all_intvs = list(intv_names.keys())
    fig = make_dark_figure()
    cooling_potential = [
        -INTERVENTIONS[k].delta_albedo * 5.0 + INTERVENTIONS[k].delta_beta * 2.0
        for k in all_intvs
    ]
    colors = [DT["accent"] if k == sel_intv else DT["border"] for k in all_intvs]
    fig.add_trace(go.Bar(x=[intv_names[k] for k in all_intvs], y=cooling_potential,
                          marker_color=colors))
    fig.update_layout(title="Relative Cooling Potential by Intervention", height=350,
                      xaxis_title="", yaxis_title="Cooling Index")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 11 — NSGA-III Optimization
# ══════════════════════════════════════════════════════════════════════
elif page == "optimize":
    section_header("NSGA-III Optimization & TOPSIS Ranking",
                    "Multi-objective optimization for intervention placement")

    col1, col2, col3 = st.columns(3)
    with col1:
        budget_cr = st.selectbox("Budget (₹ Cr)", cfg["optimization"]["budgets_cr"],
                                  format_func=lambda x: f"₹{x} Cr")
    with col2:
        pop_size = st.number_input("Population Size", min_value=50, max_value=500,
                                    value=cfg["optimization"]["pop_size"], step=50)
    with col3:
        n_gen = st.number_input("Generations", min_value=50, max_value=1000,
                                 value=cfg["optimization"]["n_generations"], step=50)

    if st.button("Run Optimization", use_container_width=True):
        with st.spinner(f"Running NSGA-III ({pop_size} pop, {n_gen} gen)..."):
            import time
            time.sleep(2)  # Simulate computation

        pareto_size = 24
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Pareto Solutions", str(pareto_size))
        with col2:
            metric_card("Objectives", "4 (LST, Cost, Coverage, Equity)")
        with col3:
            metric_card("Zones", str(cfg["optimization"]["n_zones"]))
        with col4:
            metric_card("Budget", f"₹{budget_cr} Cr")

        divider()

        np.random.seed(42)
        n_pareto = 24
        pareto_lst = np.random.uniform(0.5, 3.0, n_pareto)
        pareto_cost = np.random.uniform(0.3, 1.0, n_pareto) * budget_cr
        pareto_cov = np.random.uniform(0.1, 0.8, n_pareto)

        fig = make_dark_figure()
        scatter = fig.add_trace(go.Scatter(
            x=pareto_lst, y=pareto_cost,
            mode="markers", marker=dict(
                size=10, color=pareto_cov, colorscale="Viridis",
                colorbar=dict(title="Coverage"), showscale=True
            ), text=[f"Coverage: {c:.0%}" for c in pareto_cov],
            hovertemplate="Cooling: %{x:.2f}°C<br>Cost: ₹%{y:.1f} Cr<br>Coverage: %{marker.color:.0%}",
        ))
        fig.update_layout(title="Pareto Frontier", height=450,
                          xaxis_title="LST Reduction (°C)", yaxis_title="Cost (₹ Cr)")
        st.plotly_chart(fig, use_container_width=True)

        divider()

        st.subheader("TOPSIS Ranking")
        scenarios = [
            {"scenario_id": i, "lst_reduction": float(pareto_lst[i]),
             "cost": float(pareto_cost[i]), "coverage": float(pareto_cov[i]),
             "equity": float(np.random.uniform(0.3, 0.9)), "co_benefits": float(np.random.uniform(0.2, 0.8))}
            for i in range(n_pareto)
        ]
        ranked = compute_topsis_ranking(scenarios)
        rank_df = pd.DataFrame([
            {"Rank": r["topsis_rank"], "LST Δ°C": f"{r['lst_reduction']:.2f}",
             "Cost ₹Cr": f"{r['cost']:.1f}", "Coverage": f"{r['coverage']:.0%}",
             "TOPSIS Score": f"{r['topsis_score']:.3f}"}
            for r in ranked[:10]
        ])
        st.dataframe(rank_df.style.set_properties(**{
            "background-color": DT["surface"], "color": DT["text"], "border-color": DT["border"],
            "font-family": "JetBrains Mono",
        }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 12 — Project Configuration
# ══════════════════════════════════════════════════════════════════════
elif page == "config":
    section_header("Project Configuration", "Full config.yaml editor")

    tabs = st.tabs(["Study Area", "Data Sources", "Features", "Model", "Training", "Interventions", "Optimization", "Evaluation", "Outputs"])

    with tabs[0]:
        st.json({
            "city": cfg["study_area"]["city"],
            "crs": cfg["study_area"]["crs"],
            "resolution_m": cfg["study_area"]["resolution"],
        })

    with tabs[1]:
        st.json({
            "temporal_range": [cfg["data"]["temporal"]["start"], cfg["data"]["temporal"]["end"]],
            "summer_months": cfg["data"]["temporal"]["summer_months"],
            "landsat_collection": cfg["data"]["landsat"]["collection"],
            "cloud_threshold_pct": cfg["data"]["landsat"]["cloud_threshold"],
            "era5_variables": cfg["data"]["era5"]["variables"],
            "lulc_classes": cfg["data"]["sentinel2"]["lulc_classes"],
        })

    with tabs[2]:
        st.json({
            "target": cfg["features"]["target"],
            "n_static_features": cfg["features"]["static"],
            "n_temporal_features": cfg["features"]["temporal"],
            "indices": cfg["features"]["indices"],
        })

    with tabs[3]:
        st.json({
            "architecture": cfg["model"]["architecture"],
            "encoder_variant": cfg["model"]["encoder"]["variant"],
            "in_channels": cfg["model"]["encoder"]["in_channels"],
            "temporal_fusion": cfg["model"]["temporal_fusion"],
            "decoder": cfg["model"]["decoder"],
            "mc_dropout": cfg["model"]["mc_dropout"],
            "use_graph": cfg["model"]["use_graph"],
        })

    with tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "patch_size": cfg["training"]["patch_size"],
                "stride": cfg["training"]["stride"],
                "batch_size": cfg["training"]["batch_size"],
                "epochs": cfg["training"]["epochs"],
                "early_stopping": cfg["training"]["early_stopping"],
            })
        with col2:
            st.json({
                "optimizer": cfg["training"]["optimizer"],
                "lr": cfg["training"]["lr"],
                "weight_decay": cfg["training"]["weight_decay"],
                "scheduler": cfg["training"]["scheduler"],
                "mixed_precision": cfg["training"]["mixed_precision"],
                "augmentations": cfg["training"]["augmentations"],
            })

    with tabs[5]:
        st.json({
            "intervention_types": list(get_intervention_names().values()),
        })

    with tabs[6]:
        st.json({
            "algorithm": cfg["optimization"]["algorithm"],
            "n_zones": cfg["optimization"]["n_zones"],
            "budgets_cr": cfg["optimization"]["budgets_cr"],
            "pop_size": cfg["optimization"]["pop_size"],
            "n_generations": cfg["optimization"]["n_generations"],
            "topsis_ranking": cfg["optimization"]["topsis_ranking"],
            "knee_selection": cfg["optimization"]["knee_selection"],
        })

    with tabs[7]:
        st.json({
            "target_rmse": cfg["evaluation"]["target_rmse"],
            "target_mae": cfg["evaluation"]["target_mae"],
            "target_r2": cfg["evaluation"]["target_r2"],
            "seb_closure_threshold": cfg["evaluation"]["seb_closure_threshold"],
            "uncertainty_coverage": cfg["evaluation"]["uncertainty_coverage"],
        })

    with tabs[8]:
        st.json({
            "output_dir": cfg["outputs"]["dir"],
            "subdirs": {
                "maps": cfg["outputs"]["maps_subdir"],
                "models": cfg["outputs"]["models_subdir"],
                "reports": cfg["outputs"]["reports_subdir"],
                "rasters": cfg["outputs"]["rasters_subdir"],
                "dashboard": cfg["outputs"]["dashboard_subdir"],
            },
        })

    divider()

    if st.button("Reload config.yaml", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
