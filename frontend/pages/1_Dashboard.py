"""
Dashboard page -- overview of model performance and latest backtest results.

Displays key metrics, registered model status, quick actions, and a mini
equity curve using interactive Plotly charts.
"""

__all__: list[str] = []

import json
from pathlib import Path

import sys, os
import pandas as pd
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from components.styles import inject_global_css

logger.info("Dashboard page loaded")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_latest_metrics() -> dict | None:
    """Load metrics from the latest backtest result."""
    candidates = [
        Path("results/latest_backtest.json"),
        Path("results/runs/latest/backtest_results.json"),
        Path("results/runs/latest/metrics.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get("metrics", data)
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _load_baseline_metrics() -> dict | None:
    """Load baseline (technical) metrics if available."""
    candidates = [
        Path("results/latest_baseline.json"),
        Path("results/runs/latest/baseline_results.json"),
        Path("results/runs/latest/technical_metrics.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get("metrics", data)
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _find_equity_csv() -> Path | None:
    """Find the latest equity CSV file."""
    candidates = [
        Path("results/latest_equity.csv"),
        Path("results/runs/latest/equity.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _save_latest_results(result, baseline_result=None) -> None:
    """Save backtest results as latest_* files for dashboard pickup."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "strategy": result.strategy_name,
        "total_trades": len(result.trades),
        "metrics": result.metrics,
    }
    with open(results_dir / "latest_backtest.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=str)

    equity = result.equity_curve
    if len(equity) > 0:
        eq_df = pd.DataFrame({
            "timestamp": equity.index,
            "portfolio_value": equity.values,
        })
        eq_df.to_csv(results_dir / "latest_equity.csv", index=False)

    if result.trades:
        trade_rows = []
        for t in result.trades:
            trade_rows.append({
                "trade_id": t.trade_id,
                "ticker": t.ticker,
                "timestamp": t.timestamp,
                "signal": t.signal.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })
        pd.DataFrame(trade_rows).to_csv(
            results_dir / "latest_trades.csv", index=False
        )

    if baseline_result is not None:
        baseline_data = {
            "strategy": baseline_result.strategy_name,
            "total_trades": len(baseline_result.trades),
            "metrics": baseline_result.metrics,
        }
        with open(results_dir / "latest_baseline.json", "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2, default=str)

    logger.info("Saved latest results to {}", results_dir)


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "--"
    return f"{val:.1%}"


def _fmt_float(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

inject_global_css()
st.markdown("""
<div style="padding: 0 0 20px 0; border-bottom: 1px solid #21262d; margin-bottom: 24px;">
  <div style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600;
              letter-spacing:0.1em; text-transform:uppercase; color:#8b949e; margin-bottom:4px;">
    OVERVIEW
  </div>
  <h1 style="margin:0; font-family:'Outfit',sans-serif;">Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Preflight Health Bar
# ---------------------------------------------------------------------------

from analytics.health import run_preflight
from frontend.components.health_bar import render_health_bar

if "config" not in st.session_state:
    from config.default import CONFIG
    from copy import deepcopy
    st.session_state["config"] = deepcopy(CONFIG)

_health_report = run_preflight(st.session_state["config"])
render_health_bar(_health_report)

# ---------------------------------------------------------------------------
# Row 1 -- System status cards
# ---------------------------------------------------------------------------

from predictors.registry import REGISTRY

model_names = REGISTRY.list_all()
n_models = len(model_names)

data_dir = Path("data/features")
csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
n_data = len(csv_files)

metrics = _load_latest_metrics()
baseline = _load_baseline_metrics()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Models Registered", n_models)

with col2:
    st.metric("Data Files", n_data)

with col3:
    val = _fmt_float(metrics.get("sharpe_ratio") if metrics else None)
    st.metric("Sharpe Ratio", val)

with col4:
    val = _fmt_pct(metrics.get("win_rate") if metrics else None)
    st.metric("Win Rate", val)

with col5:
    dd = metrics.get("max_drawdown") if metrics else None
    st.metric("Max Drawdown", _fmt_pct(dd))

# ---------------------------------------------------------------------------
# Row 2 -- Model Status + Quick Actions
# ---------------------------------------------------------------------------

st.markdown("---")

left_col, right_col = st.columns([3, 2])

# --- Model Status Table (dynamic from registry) ---
with left_col:
    st.subheader("Model Status")

    if model_names:
        rows = []
        for name in model_names:
            try:
                pred = REGISTRY._predictors[name]
                cls_name = type(pred).__name__
                adapter = getattr(pred, "_meta", {}).get("adapter", "")
                if not adapter:
                    if "Cnn" in cls_name:
                        adapter = "multiscale_cnn"
                    elif "Vol" in cls_name:
                        adapter = "lightgbm"
                    elif "Meta" in cls_name:
                        adapter = "lightgbm"
                    else:
                        adapter = "—"
                label = getattr(pred, "output_label", "—")
                ready = pred.is_ready() if hasattr(pred, "is_ready") else None
                rows.append({
                    "Name": name,
                    "Adapter": adapter,
                    "Output Label": label,
                    "Status": "✅ Ready" if ready else "⚠️ No weights",
                })
            except Exception:
                rows.append({
                    "Name": name,
                    "Adapter": "—",
                    "Output Label": "—",
                    "Status": "❓ Unknown",
                })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No models registered. Go to **Configuration → Google Drive** "
            "to sync model files."
        )

# --- Quick Actions ---
with right_col:
    st.subheader("Quick Actions")

    _bt_disabled = not _health_report.can_run
    if _bt_disabled:
        st.warning("Backtest blocked — fix RED preflight checks above.")

    if st.button("▶ Run Full Backtest", use_container_width=True, key="dash_run_bt", disabled=_bt_disabled):
        with st.spinner("Running backtest..."):
            try:
                from backtest.runner import run_backtest

                config = st.session_state.get("config", {})
                result = run_backtest(config, strategy_name="ai", save_results=True)
                _save_latest_results(result)
                st.success(
                    f"Backtest complete: {result.metrics.get('total_trades', 0)} trades, "
                    f"Sharpe={result.metrics.get('sharpe_ratio', 0):.2f}"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")

    if st.button("⚡ Run AI vs Baseline", use_container_width=True, key="dash_run_cmp", disabled=_bt_disabled):
        with st.spinner("Running comparison..."):
            try:
                from backtest.runner import run_comparison

                config = st.session_state.get("config", {})
                ai_r, tech_r = run_comparison(config, save_results=True)
                _save_latest_results(ai_r, tech_r)
                ai_sharpe = ai_r.metrics.get("sharpe_ratio", 0)
                tech_sharpe = tech_r.metrics.get("sharpe_ratio", 0)
                st.success(
                    f"AI Sharpe={ai_sharpe:.2f} vs Technical Sharpe={tech_sharpe:.2f} "
                    f"(Δ={ai_sharpe - tech_sharpe:+.2f})"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    if st.button("🔄 Retrain Aggregator", use_container_width=True, key="dash_retrain"):
        pred_dir = Path("results/predictions")
        has_predictions = pred_dir.exists() and any(pred_dir.glob("*_predictions.csv"))
        if not has_predictions:
            st.warning("No prediction files found. Run inference first (generate predictions), then retrain.")
        else:
            st.info("Aggregator retraining is not yet wired into the dashboard. "
                    "Predictions exist — use the backtest pipeline to retrain.")

# ---------------------------------------------------------------------------
# Row 3 -- Mini equity curve
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Latest Equity Curve")

equity_path = _find_equity_csv()
if equity_path is not None:
    try:
        import plotly.graph_objects as go

        eq_df = pd.read_csv(equity_path)

        ts_col = "timestamp" if "timestamp" in eq_df.columns else eq_df.columns[0]
        val_col = "portfolio_value" if "portfolio_value" in eq_df.columns else eq_df.columns[1]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eq_df[ts_col],
                y=eq_df[val_col],
                mode="lines",
                name="Portfolio Value",
                line=dict(width=2),
                fill="tozeroy",
            )
        )
        fig.update_layout(
            height=350,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"Could not render equity curve: {exc}")
else:
    st.info("Run a backtest to display the equity curve.")
