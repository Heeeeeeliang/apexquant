"""
Model Diagnostics page -- dynamic per-adapter introspection.

Generates one tab per registered adapter from REGISTRY.list_all(),
plus a final Aggregator tab.  Each adapter tab shows readiness,
metadata, feature importance (LightGBM) or architecture summary (CNN).
"""

__all__: list[str] = []

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from components.styles import inject_global_css

logger.info("Model Diagnostics page loaded")

if "config" not in st.session_state:
    from config.default import CONFIG
    st.session_state["config"] = deepcopy(CONFIG)

cfg = st.session_state["config"]

# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e"),
    xaxis=dict(gridcolor="#2d333b"),
)


def _horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str,
                    title: str = "", height: int = 350,
                    color: str = "#3fb950") -> go.Figure:
    """Create a horizontal bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col].values[::-1],
        y=df[y_col].values[::-1],
        orientation="h",
        marker_color=color,
    ))
    fig.update_layout(
        **_DARK,
        height=height,
        margin=dict(l=160, r=20, t=40, b=40),
        title=title,
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

inject_global_css()
st.markdown("""
<div style="padding: 0 0 20px 0; border-bottom: 1px solid #21262d; margin-bottom: 24px;">
  <div style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600;
              letter-spacing:0.1em; text-transform:uppercase; color:#8b949e; margin-bottom:4px;">
    ③ EVALUATE
  </div>
  <h1 style="margin:0; font-family:'Outfit',sans-serif;">Model Diagnostics</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Discover registered adapters
# ---------------------------------------------------------------------------

from predictors.registry import REGISTRY

adapter_names = REGISTRY.list_all()

if not adapter_names:
    st.warning(
        "No models registered. Sync model files from Google Drive "
        "or place them in `models/` with a `meta.json` file."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Build tabs: one per adapter + Aggregator
# ---------------------------------------------------------------------------

tab_labels = adapter_names + ["Aggregator"]
tabs = st.tabs(tab_labels)

# ---------------------------------------------------------------------------
# Per-adapter tabs
# ---------------------------------------------------------------------------

for tab, adapter_name in zip(tabs[:-1], adapter_names):
    with tab:
        predictor = REGISTRY.get(adapter_name)

        # --- Basic info ---
        st.subheader(adapter_name)

        model_dir = getattr(predictor, "_model_dir", None)
        output_label = getattr(predictor, "output_label", "—")
        ready = predictor.is_ready() if hasattr(predictor, "is_ready") else None

        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Output Label", output_label)
        with info_cols[1]:
            if ready is not None:
                st.metric("Status", "Ready" if ready else "Not Ready")
            else:
                st.metric("Status", "Unknown")
        with info_cols[2]:
            st.metric("Model Dir", str(model_dir) if model_dir else "—")

        # --- Not ready: show warning with expected path ---
        if ready is False:
            if model_dir is not None:
                model_dir_path = Path(model_dir)
                # Detect expected weight file
                expected_files = [
                    model_dir_path / "weights.joblib",
                    model_dir_path / "weights.pt",
                ]
                expected_str = " or ".join(
                    f"`{p}`" for p in expected_files
                )
                st.warning(
                    f"Weight file not found. Expected at {expected_str}. "
                    "Sync from Google Drive or train the model."
                )
            else:
                st.warning("Model directory not configured.")
            continue

        # --- Ready: try to load and show details ---
        # Detect adapter type by checking for known attributes
        is_lgb = hasattr(predictor, "_extract_features") and not hasattr(predictor, "_short_win")
        is_cnn = hasattr(predictor, "_short_win")

        if is_lgb:
            # --- LightGBM adapter (vol or meta) ---
            try:
                if predictor._model is None:
                    predictor.load()

                model_obj = predictor._model

                # Feature count
                n_feat = getattr(model_obj, "n_features_in_", None)
                if n_feat is not None:
                    st.caption(f"Features: {n_feat}")

                # Feature importance
                importances = getattr(model_obj, "feature_importances_", None)
                feature_names = getattr(model_obj, "feature_names_in_", None)

                if importances is not None:
                    st.markdown("##### Feature Importance")

                    if feature_names is not None and len(feature_names) == len(importances):
                        names = list(feature_names)
                    else:
                        names = [f"f{i}" for i in range(len(importances))]

                    fi_df = pd.DataFrame({
                        "feature": names,
                        "importance": importances,
                    }).sort_values("importance", ascending=False).reset_index(drop=True)

                    top_n = st.slider(
                        "Top N features",
                        5, min(30, len(fi_df)),
                        value=min(15, len(fi_df)),
                        key=f"diag_fi_{adapter_name}",
                    )
                    fig = _horizontal_bar(
                        fi_df.head(top_n), "importance", "feature",
                        height=max(250, top_n * 25),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Model loaded but does not expose feature importances.")

            except Exception as exc:
                st.error(f"Could not load model for inspection: {exc}")

        elif is_cnn:
            # --- CNN adapter ---
            try:
                if predictor._model is None:
                    predictor.load()

                st.markdown("##### Architecture Summary")

                n_short = predictor._s_mu.shape[0] if predictor._s_mu is not None else "?"
                n_long = predictor._l_mu.shape[0] if predictor._l_mu is not None else "?"
                short_win = getattr(predictor, "_short_win", "?")
                long_win = getattr(predictor, "_long_win", "?")
                task = getattr(predictor, "_task", "?")

                arch_data = {
                    "Property": [
                        "Task", "Short features", "Long features",
                        "Short window", "Long window", "Device",
                    ],
                    "Value": [
                        task, str(n_short), str(n_long),
                        str(short_win), str(long_win),
                        str(getattr(predictor, "_device", "cpu")),
                    ],
                }
                st.dataframe(
                    pd.DataFrame(arch_data),
                    use_container_width=True,
                    hide_index=True,
                )

                # Show layer summary
                model_obj = predictor._model
                if model_obj is not None:
                    param_count = sum(
                        p.numel() for p in model_obj.parameters()
                    )
                    st.caption(f"Total parameters: {param_count:,}")

            except Exception as exc:
                st.error(f"Could not load CNN model for inspection: {exc}")

        else:
            # --- Unknown / legacy predictor type ---
            st.info(
                "This predictor type does not support detailed diagnostics. "
                f"Class: `{type(predictor).__name__}`"
            )

        # --- meta.json contents ---
        meta_dict = getattr(predictor, "_meta", None)
        if meta_dict:
            with st.expander("meta.json"):
                st.json(meta_dict)


# ---------------------------------------------------------------------------
# Aggregator tab (data-driven, not hardcoded)
# ---------------------------------------------------------------------------

with tabs[-1]:
    st.subheader("Learned Aggregator")

    agg_path = Path("results/runs/latest/aggregator.joblib")
    agg_loaded = False
    agg_state: dict | None = None

    if agg_path.exists():
        try:
            import joblib
            agg_state = joblib.load(agg_path)
            agg_loaded = True
        except Exception as exc:
            logger.warning("Failed to load aggregator: {}", exc)

    if not agg_loaded:
        st.info(
            "Aggregator not fitted. Run **Retrain Aggregator** from the "
            "Dashboard to populate this tab."
        )
    else:
        model = agg_state.get("model")
        feature_names = agg_state.get("feature_names", [])
        threshold = agg_state.get("threshold", 0.50)
        model_type = agg_state.get("model_type", "unknown")

        st.caption(f"Model type: `{model_type}` | Threshold: `{threshold:.2f}`")

        # Feature importance
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is not None and len(feature_names) == len(importances):
            st.markdown("##### Feature Importance")

            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            fig_fi = _horizontal_bar(
                fi_df, "importance", "feature",
                height=max(250, len(fi_df) * 30),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Predictor weight pie chart
            st.markdown("##### Predictor Weight Breakdown")

            _PIE_COLORS = [
                "#3fb950", "#58a6ff", "#ff9800", "#f85149",
                "#d29922", "#8b949e", "#bc8cff", "#39d353", "#f778ba",
            ]
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=fi_df["feature"].tolist(),
                    values=fi_df["importance"].tolist(),
                    hole=0.4,
                    marker=dict(colors=_PIE_COLORS[:len(fi_df)]),
                    textinfo="label+percent",
                    textfont=dict(color="#e6edf3"),
                )
            ])
            fig_pie.update_layout(
                **_DARK,
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Validation performance
        st.markdown("##### Validation Set Performance")

        val_metrics_path = Path("results/runs/latest/val_metrics.json")
        if val_metrics_path.exists():
            try:
                with open(val_metrics_path, encoding="utf-8") as f:
                    val_data = json.load(f)

                vc1, vc2, vc3 = st.columns(3)
                with vc1:
                    before = val_data.get("sharpe_before_agg")
                    st.metric(
                        "Sharpe (before aggregation)",
                        f"{before:.2f}" if before is not None else "--",
                    )
                with vc2:
                    after = val_data.get("sharpe_after_agg")
                    st.metric(
                        "Sharpe (after aggregation)",
                        f"{after:.2f}" if after is not None else "--",
                    )
                with vc3:
                    if before is not None and after is not None:
                        delta = after - before
                        st.metric(
                            "Improvement",
                            f"{delta:+.2f}",
                            delta=f"{delta:+.2f} Sharpe",
                        )
                    else:
                        st.metric("Improvement", "--")
            except Exception as exc:
                logger.warning("Failed to load val metrics: {}", exc)
                st.info("Could not parse validation metrics.")
        else:
            st.info(
                f"No validation metrics file found at `{val_metrics_path}`. "
                "Run the full pipeline to generate."
            )
