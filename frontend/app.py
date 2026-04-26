"""
ApexQuant Streamlit Frontend -- Main Entry Point.

Quantitative trading research platform with interactive
Plotly charts, model status monitoring, and full pipeline control.

Usage::

    streamlit run frontend/app.py
"""

__all__: list[str] = []

import sys, os
from copy import deepcopy
from pathlib import Path

import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from components.styles import inject_global_css, render_logo

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ApexQuant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# ---------------------------------------------------------------------------
# Load config into session state
# ---------------------------------------------------------------------------

if "config" not in st.session_state:
    from config.default import CONFIG
    st.session_state["config"] = deepcopy(CONFIG)
    logger.info("Config loaded into session_state")

# ---------------------------------------------------------------------------
# Sidebar toggle
# ---------------------------------------------------------------------------

if st.button("☰", key="sidebar_toggle", help="Toggle sidebar"):
    if st.session_state.get("sidebar_open", True):
        st.session_state.sidebar_open = False
    else:
        st.session_state.sidebar_open = True
    st.rerun()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if st.session_state.get("sidebar_open", True):
    with st.sidebar:
        render_logo()
        st.caption("COMP3931 — University of Leeds 2024/25")

# ---------------------------------------------------------------------------
# Main content — Project Portal
# ---------------------------------------------------------------------------

# ── HERO BLOCK ──────────────────────────────────────────
st.html("""
<div style="padding: 48px 0 40px 0; border-bottom: 1px solid #21262d; margin-bottom: 40px;">
<div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
<svg width="44" height="44" viewBox="0 0 48 48" fill="none">
<rect width="48" height="48" rx="10" fill="#0d0d14"/>
<polygon points="24,7 43,40 5,40" fill="none" stroke="#f5a623" stroke-width="2.2" stroke-linejoin="round"/>
<polygon points="24,19 34,40 14,40" fill="#f5a623" fill-opacity="0.12"/>
<circle cx="24" cy="18" r="2.8" fill="#f5a623"/>
</svg>
<div>
<div style="font-family:'Outfit',sans-serif; font-size:26px; font-weight:700; letter-spacing:-0.02em; color:#e6edf3; line-height:1.1;">
Apex<span style="color:#f5a623;">Quant</span>
</div>
<div style="font-family:'Outfit',sans-serif; font-size:12px; color:#8b949e; letter-spacing:0.04em; margin-top:2px;">
COMP3931 · University of Leeds · 2024/25
</div>
</div>
</div>
<p style="font-family:'Outfit',sans-serif; font-size:15px; color:#8b949e; max-width:600px; line-height:1.6; margin:0 0 20px 0;">
A cascaded task decomposition framework for systematic stock trading research.
Direct price prediction converges to ~50% accuracy — this platform decomposes
the problem into three solvable sub-tasks.
</p>
<div style="display:flex; gap:8px; flex-wrap:wrap;">
<span style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600; padding:4px 10px; border-radius:20px; letter-spacing:0.04em; background:rgba(245,166,35,0.12); color:#f5a623; border:1px solid rgba(245,166,35,0.25);">Layer 1 · Vol DA 83.6%</span>
<span style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600; padding:4px 10px; border-radius:20px; letter-spacing:0.04em; background:rgba(56,139,253,0.12); color:#388bfd; border:1px solid rgba(56,139,253,0.25);">Layer 2 · CNN AUC 0.826</span>
<span style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600; padding:4px 10px; border-radius:20px; letter-spacing:0.04em; background:rgba(63,185,80,0.12); color:#3fb950; border:1px solid rgba(63,185,80,0.25);">Layer 3 · Dir p=0.0008</span>
<span style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600; padding:4px 10px; border-radius:20px; letter-spacing:0.04em; background:rgba(139,148,158,0.1); color:#8b949e; border:1px solid rgba(139,148,158,0.2);">Backtest +28.82% · Sharpe 1.89</span>
</div>
</div>
""")


# ── THREE-PHASE NAVIGATION CARDS ────────────────────────
st.html("""
<div style="font-family:'Outfit',sans-serif; font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#484f58; margin-bottom:16px;">
RESEARCH PIPELINE
</div>
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.html("""
<div style="background:#161b22; border:1px solid #21262d; border-top:2px solid #388bfd; border-radius:8px; padding:24px; height:200px; box-sizing:border-box;">
<div style="font-family:'Outfit',sans-serif; font-size:10px; font-weight:700; letter-spacing:0.14em; color:#388bfd; margin-bottom:12px;">① SETUP</div>
<div style="font-family:'Outfit',sans-serif; font-size:16px; font-weight:600; color:#e6edf3; margin-bottom:8px;">Configure</div>
<div style="font-family:'Outfit',sans-serif; font-size:13px; color:#8b949e; line-height:1.5; margin-bottom:16px;">Load data, sync model weights from Google Drive, verify feature pipelines.</div>
<div style="display:flex; align-items:center; gap:6px;">
<div style="width:6px; height:6px; border-radius:50%; background:#3fb950;"></div>
<span style="font-family:'Outfit',sans-serif; font-size:11px; color:#3fb950;">Ready</span>
</div>
</div>
""")
    st.page_link("pages/2_①_Setup.py", label="→ Open Setup", use_container_width=True)

with col2:
    st.html("""
<div style="background:#161b22; border:1px solid #21262d; border-top:2px solid #f5a623; border-radius:8px; padding:24px; height:200px; box-sizing:border-box;">
<div style="font-family:'Outfit',sans-serif; font-size:10px; font-weight:700; letter-spacing:0.14em; color:#f5a623; margin-bottom:12px;">② RESEARCH</div>
<div style="font-family:'Outfit',sans-serif; font-size:16px; font-weight:600; color:#e6edf3; margin-bottom:8px;">Strategy & Pipeline</div>
<div style="font-family:'Outfit',sans-serif; font-size:13px; color:#8b949e; line-height:1.5; margin-bottom:16px;">Edit signal thresholds, configure the three-layer cascade, run quick validation.</div>
<div style="display:flex; align-items:center; gap:6px;">
<div style="width:6px; height:6px; border-radius:50%; background:#f5a623;"></div>
<span style="font-family:'Outfit',sans-serif; font-size:11px; color:#f5a623;">In Progress</span>
</div>
</div>
""")
    st.page_link("pages/4_②_Pipeline.py", label="→ Open Pipeline", use_container_width=True)

with col3:
    st.html("""
<div style="background:#161b22; border:1px solid #21262d; border-top:2px solid #3fb950; border-radius:8px; padding:24px; height:200px; box-sizing:border-box;">
<div style="font-family:'Outfit',sans-serif; font-size:10px; font-weight:700; letter-spacing:0.14em; color:#3fb950; margin-bottom:12px;">③ EVALUATE</div>
<div style="font-family:'Outfit',sans-serif; font-size:16px; font-weight:600; color:#e6edf3; margin-bottom:8px;">Backtest & Results</div>
<div style="font-family:'Outfit',sans-serif; font-size:13px; color:#8b949e; line-height:1.5; margin-bottom:16px;">Run full backtest, compare AI vs baseline, view Sharpe / drawdown breakdown.</div>
<div style="display:flex; align-items:center; gap:6px;">
<div style="width:6px; height:6px; border-radius:50%; background:#3fb950;"></div>
<span style="font-family:'Outfit',sans-serif; font-size:11px; color:#3fb950;">+28.82% · Sharpe 1.89</span>
</div>
</div>
""")
    st.page_link("pages/5_③_Backtest_Analysis.py", label="→ Open Backtest", use_container_width=True)


# ── SYSTEM STATUS ROW ────────────────────────────────────
st.html("<div style='margin-top:40px; border-top:1px solid #21262d; padding-top:24px;'>")

from predictors.registry import REGISTRY
import pandas as pd

model_names = REGISTRY.list_all()
n_models = len(model_names)

data_dir = Path("data/features")
csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
n_data = len(csv_files)

col1, col2 = st.columns(2)
with col1:
    st.metric("Models Registered", n_models)
with col2:
    st.metric("Data Files (CSV)", n_data)

if model_names:
    rows = []
    for name in model_names:
        try:
            pred = REGISTRY._predictors[name]
            adapter = getattr(pred, "_meta", {}).get("adapter", "—")
            if not adapter or adapter == "—":
                cls_name = type(pred).__name__
                if "Cnn" in cls_name:
                    adapter = "multiscale_cnn"
                elif "Vol" in cls_name:
                    adapter = "lightgbm"
                elif "Meta" in cls_name:
                    adapter = "lightgbm"
            label = getattr(pred, "output_label", "—")
            ready = pred.is_ready() if hasattr(pred, "is_ready") else "—"
            rows.append({
                "Name": name,
                "Adapter": adapter,
                "Output Label": label,
                "Weights Ready": "✓" if ready is True else ("✗" if ready is False else str(ready)),
            })
        except Exception:
            rows.append({
                "Name": name,
                "Adapter": "—",
                "Output Label": "—",
                "Weights Ready": "?",
            })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No models registered. Go to **Configuration → Google Drive** "
        "to sync model files, then reload."
    )

st.html("</div>")
