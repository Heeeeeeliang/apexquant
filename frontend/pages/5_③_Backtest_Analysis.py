"""
Backtest Analysis page -- browse, compare, and analyse saved backtest runs.

Lists all run directories in ``results/runs/``, loads metrics, equity
curves, and trade logs.  Three tabs: Overview (metric cards, equity chart,
comparison table), Per Ticker (per-ticker breakdown), Trade Log (filtered,
paginated with donut chart).
"""

__all__: list[str] = []

import io
import json
import zipfile
from copy import deepcopy
from datetime import date
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

logger.info("Backtest Analysis page loaded")

if "config" not in st.session_state:
    from config.default import CONFIG
    st.session_state["config"] = deepcopy(CONFIG)

cfg = st.session_state["config"]

# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e"),
    xaxis=dict(gridcolor="#2d333b"),
    yaxis=dict(gridcolor="#2d333b"),
)

# ---------------------------------------------------------------------------
# QC Baseline constants
# ---------------------------------------------------------------------------

_QC_BASELINE: dict[str, object] = {
    "sharpe_ratio": -1.12,
    "total_trades": 589,
    "win_rate": 0.50,
    "total_return": None,
    "sortino_ratio": None,
    "max_drawdown": None,
    "profit_factor": None,
    "avg_trade_pnl": None,
    "avg_bars_held": None,
    "calmar_ratio": None,
}

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
  <h1 style="margin:0; font-family:'Outfit',sans-serif;">Backtest Analysis</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config guard check + active preset badge
# ---------------------------------------------------------------------------

from config.schema import validate_config
from frontend.components.config_tabs import render_guard_banner, render_active_preset_badge

_bt_violations = validate_config(cfg)
_bt_errors = [v for v in _bt_violations if v.level == "error"]

if _bt_errors:
    render_guard_banner(_bt_violations)
    st.warning(
        "Config has guard errors — backtest results may be unreliable. "
        "Fix issues on the **Configuration** page."
    )

_bt_active_preset = st.session_state.get("active_preset")
if _bt_active_preset is not None:
    from config.presets import list_presets as _bt_list_presets
    _bt_preset_map = {p["id"]: p["name"] for p in _bt_list_presets()}
    render_active_preset_badge(_bt_preset_map.get(str(_bt_active_preset), None))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUNS_DIR = Path("results/runs")


def _list_runs() -> list[Path]:
    """Return run directories sorted newest first."""
    if not _RUNS_DIR.exists():
        return []
    dirs = [
        d for d in _RUNS_DIR.iterdir()
        if d.is_dir() and d.name != "latest"
    ]
    return sorted(dirs, key=lambda d: d.name, reverse=True)


def _load_run_metrics(run_dir: Path) -> dict | None:
    """Load metrics JSON from a run directory."""
    for name in ("metrics.json", "backtest_results.json"):
        p = run_dir / name
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug("Loaded metrics from {}", p)
                return data.get("metrics", data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load {}: {}", p, exc)
    return None


def _load_run_equity(run_dir: Path) -> pd.DataFrame | None:
    """Load equity CSV from a run directory."""
    p = run_dir / "equity.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        logger.debug("Loaded equity from {} ({} rows)", p, len(df))
        return df
    except Exception as exc:
        logger.warning("Failed to load equity from {}: {}", p, exc)
        return None


def _load_run_trades(run_dir: Path) -> pd.DataFrame | None:
    """Load trades CSV from a run directory."""
    p = run_dir / "trades.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        logger.debug("Loaded trades from {} ({} rows)", p, len(df))
        return df
    except Exception as exc:
        logger.warning("Failed to load trades from {}: {}", p, exc)
        return None


def _find_companion_run(run_dir: Path) -> Path | None:
    """Find a companion AI or Technical run in the same batch.

    If the run dir ends with ``_ai`` look for ``_technical`` and vice versa.
    """
    name = run_dir.name
    parent = run_dir.parent
    if name.endswith("_ai"):
        companion = parent / name.replace("_ai", "_technical")
    elif name.endswith("_technical"):
        companion = parent / name.replace("_technical", "_ai")
    else:
        # Try common sibling patterns
        for suffix in ("_ai", "_technical"):
            candidate = parent / (name + suffix)
            if candidate.exists():
                return candidate
        return None
    return companion if companion.exists() else None


def _fmt_metric(val: object, is_pct: bool = False) -> str:
    if val is None:
        return "--"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.2%}" if is_pct else f"{val:.2f}"
    return str(val)


# ---------------------------------------------------------------------------
# Run selector
# ---------------------------------------------------------------------------

_qr1, _qr2, _qr3, _ = st.columns([1, 1, 1, 2])
with _qr1:
    if st.button("▶ Run AI Backtest", use_container_width=True, key="bt_quick_ai"):
        _prog_bar = st.progress(0, text="Starting backtest...")

        def _bt_progress(current: int, total: int, ticker: str):
            pct = min(current / max(total, 1), 1.0)
            _prog_bar.progress(pct, text=f"Processing {ticker}... ({current}/{total} bars)")

        try:
            from backtest.runner import run_backtest
            _cfg = st.session_state.get("config", cfg)
            _result = run_backtest(
                _cfg, strategy_name="ai", save_results=True,
                progress_callback=_bt_progress,
            )
            _prog_bar.progress(1.0, text="Complete!")
            st.success(f"Done: {_result.metrics.get('total_trades', 0)} trades")
            st.rerun()
        except Exception as _exc:
            st.error(f"Failed: {_exc}")
with _qr2:
    if st.button("⚡ Run AI vs Baseline", use_container_width=True, key="bt_quick_cmp"):
        _prog_bar_cmp = st.progress(0, text="Starting comparison...")

        def _cmp_progress(current: int, total: int, ticker: str):
            pct = min(current / max(total, 1), 1.0)
            _prog_bar_cmp.progress(pct, text=f"Processing {ticker}... ({current}/{total} bars)")

        try:
            from backtest.runner import run_comparison
            _cfg = st.session_state.get("config", cfg)
            _ai, _tech = run_comparison(_cfg, save_results=True)
            _prog_bar_cmp.progress(1.0, text="Complete!")
            st.success(
                f"AI Sharpe={_ai.metrics.get('sharpe_ratio', 0):.2f} "
                f"vs Tech={_tech.metrics.get('sharpe_ratio', 0):.2f}"
            )
            st.rerun()
        except Exception as _exc:
            import traceback
            _tb = traceback.format_exc()
            logger.error("Run AI vs Baseline failed:\n{}", _tb)
            st.error(f"Failed: {_exc}")
            st.code(_tb, language="python")
with _qr3:
    # Show prediction freshness indicator
    _pred_dir = Path(
        cfg.get("data", {}).get("predictions_dir", "results/predictions")
    )
    if _pred_dir.exists():
        _pred_csvs = sorted(_pred_dir.glob("*.csv"))
        if _pred_csvs:
            from datetime import datetime as _dt
            _latest_mtime = max(f.stat().st_mtime for f in _pred_csvs)
            _pred_age = _dt.fromtimestamp(_latest_mtime)
            _pred_cols: list[str] = []
            try:
                _sample = pd.read_csv(_pred_csvs[0], nrows=0)
                _pred_cols = [
                    c for c in _sample.columns
                    if c.lower() not in {"timestamp", "ts_event", "datetime", "date"}
                ]
            except Exception:
                pass
            st.caption(
                f"Predictions: {len(_pred_csvs)} files, "
                f"updated {_pred_age:%Y-%m-%d %H:%M}"
                + (f", cols: {_pred_cols}" if _pred_cols else "")
            )
        else:
            st.caption("Predictions: no CSV files found")
    else:
        st.caption("Predictions: not generated yet")

    if st.button("🔮 Generate Predictions", use_container_width=True, key="bt_gen_preds"):
        _pred_bar = st.progress(0, text="Generating predictions...")

        def _pred_progress(current: int, total: int, ticker: str):
            pct = min(current / max(total, 1), 1.0)
            _pred_bar.progress(pct, text=f"Predicting {ticker}... ({current}/{total} bars)")

        try:
            from backtest.inference import generate_predictions
            _cfg = st.session_state.get("config", cfg)
            _summary = generate_predictions(
                _cfg, progress_callback=_pred_progress,
            )
            _pred_bar.progress(1.0, text="Complete!")
            st.success(
                f"Generated predictions for {_summary['tickers_processed']} ticker(s), "
                f"{_summary['total_rows']} total rows. "
                f"Models: {', '.join(_summary['predictors_used']) or 'none'}"
            )
            st.rerun()
        except Exception as _exc:
            st.error(f"Prediction generation failed: {_exc}")

st.markdown("---")

runs = _list_runs()

if not runs:
    st.info(
        "No backtest runs found in `results/runs/`. "
        "Run a backtest using the buttons above, from the Strategy Editor, or Dashboard."
    )
    st.stop()

run_names = [d.name for d in runs]
selected_run_name = st.selectbox(
    "Select Run",
    run_names,
    key="bt_run_select",
)

sel_col1, sel_col2, _ = st.columns([1, 1, 4])
with sel_col1:
    load_btn = st.button("📂 Load", use_container_width=True, key="bt_load_btn")
with sel_col2:
    if st.session_state.get("bt_loaded_run"):
        st.caption(f"Loaded: `{st.session_state['bt_loaded_run']}`")

if load_btn:
    run_dir = _RUNS_DIR / selected_run_name
    metrics = _load_run_metrics(run_dir)
    equity_df = _load_run_equity(run_dir)
    trades_df = _load_run_trades(run_dir)

    st.session_state["bt_loaded_run"] = selected_run_name
    st.session_state["bt_run_dir"] = str(run_dir)
    st.session_state["bt_metrics"] = metrics
    st.session_state["bt_equity_df"] = equity_df
    st.session_state["bt_trades_df"] = trades_df

    # Try loading companion run
    companion = _find_companion_run(run_dir)
    if companion:
        st.session_state["bt_companion_metrics"] = _load_run_metrics(companion)
        st.session_state["bt_companion_equity"] = _load_run_equity(companion)
        st.session_state["bt_companion_name"] = companion.name
        logger.info("Loaded companion run: {}", companion.name)
    else:
        st.session_state.pop("bt_companion_metrics", None)
        st.session_state.pop("bt_companion_equity", None)
        st.session_state.pop("bt_companion_name", None)

    st.rerun()

# ---------------------------------------------------------------------------
# Check loaded state
# ---------------------------------------------------------------------------

metrics = st.session_state.get("bt_metrics")
if metrics is None:
    st.info("Select a run and click **Load** to view results.")
    st.stop()

equity_df = st.session_state.get("bt_equity_df")
trades_df = st.session_state.get("bt_trades_df")
run_dir = Path(st.session_state.get("bt_run_dir", ""))

companion_metrics = st.session_state.get("bt_companion_metrics")
companion_equity = st.session_state.get("bt_companion_equity")
companion_name = st.session_state.get("bt_companion_name", "")

# Determine which is AI and which is Technical
is_ai_run = "ai" in st.session_state.get("bt_loaded_run", "").lower()
ai_metrics = metrics if is_ai_run else companion_metrics
tech_metrics = companion_metrics if is_ai_run else metrics

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_ticker, tab_trades, tab_verdict, tab_markers, tab_diag = st.tabs(
    ["Overview", "Per Ticker", "Trade Log", "Verdict & Attribution", "Trade Markers", "Diagnostics"]
)

# =========================================================================
# Tab 1: Overview
# =========================================================================

with tab_overview:
    # --- Metric cards ---
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.metric("Sharpe", _fmt_metric(metrics.get("sharpe_ratio")))
    with c2:
        st.metric("Win Rate", _fmt_metric(metrics.get("win_rate"), is_pct=True))
    with c3:
        st.metric(
            "TW Win Rate",
            _fmt_metric(metrics.get("weighted_win_rate"), is_pct=True),
            help="Win rate weighted by bars held per trade",
        )
    with c4:
        st.metric("Max DD", _fmt_metric(metrics.get("max_drawdown"), is_pct=True))
    with c5:
        st.metric("Total Return", _fmt_metric(metrics.get("total_return"), is_pct=True))
    with c6:
        st.metric("Trades", _fmt_metric(metrics.get("total_trades")))
    with c7:
        st.metric("Profit Factor", _fmt_metric(metrics.get("profit_factor")))

    # --- Equity curve ---
    st.markdown("##### Equity Curve")

    initial_capital = cfg.get("backtest", {}).get("initial_capital", 100_000)

    if equity_df is not None and len(equity_df) > 0:
        fig = go.Figure()

        ts_col = "timestamp" if "timestamp" in equity_df.columns else equity_df.columns[0]
        val_col = "portfolio_value" if "portfolio_value" in equity_df.columns else equity_df.columns[1]

        # Primary equity line (AI / main)
        label_main = "AI Strategy" if is_ai_run else st.session_state.get("bt_loaded_run", "Strategy")
        fig.add_trace(
            go.Scatter(
                x=equity_df[ts_col],
                y=equity_df[val_col],
                mode="lines",
                name=label_main,
                line=dict(color="#3fb950", width=2),
                hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
            )
        )

        # Companion equity overlay
        if companion_equity is not None and len(companion_equity) > 0:
            c_ts = "timestamp" if "timestamp" in companion_equity.columns else companion_equity.columns[0]
            c_val = "portfolio_value" if "portfolio_value" in companion_equity.columns else companion_equity.columns[1]
            label_comp = "EMA+RSI Baseline" if is_ai_run else "AI Strategy"
            fig.add_trace(
                go.Scatter(
                    x=companion_equity[c_ts],
                    y=companion_equity[c_val],
                    mode="lines",
                    name=label_comp,
                    line=dict(color="#ff9800", width=2, dash="dash"),
                    hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
                )
            )

        # Initial capital line
        fig.add_hline(
            y=initial_capital,
            line_dash="dot",
            line_color="#8b949e",
            line_width=1,
            annotation_text=f"Initial: ${initial_capital:,}",
            annotation_position="bottom right",
        )

        # Drawdown shading
        vals = equity_df[val_col].values
        cummax = np.maximum.accumulate(vals)
        fig.add_trace(
            go.Scatter(
                x=equity_df[ts_col], y=cummax,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=equity_df[ts_col], y=vals,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(248, 81, 73, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Drawdown",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            **_DARK_LAYOUT,
            height=420,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity data available for this run.")

    # --- Metrics comparison table ---
    st.markdown("##### Metrics Comparison")

    metric_defs = [
        ("total_return", "Total Return", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
        ("sortino_ratio", "Sortino Ratio", False),
        ("calmar_ratio", "Calmar Ratio", False),
        ("max_drawdown", "Max Drawdown", True),
        ("win_rate", "Win Rate", True),
        ("weighted_win_rate", "TW Win Rate", True),
        ("profit_factor", "Profit Factor", False),
        ("total_trades", "Total Trades", False),
        ("avg_trade_pnl", "Avg Trade PnL", False),
        ("avg_bars_held", "Avg Bars Held", False),
    ]

    cmp_rows = []
    for key, label, is_pct in metric_defs:
        ai_val = ai_metrics.get(key) if ai_metrics else None
        tech_val = tech_metrics.get(key) if tech_metrics else None
        qc_val = _QC_BASELINE.get(key)

        cmp_rows.append({
            "Metric": label,
            "AI Strategy": _fmt_metric(ai_val, is_pct),
            "EMA+RSI": _fmt_metric(tech_val, is_pct),
            "QC Baseline": _fmt_metric(qc_val, is_pct),
        })

    cmp_df = pd.DataFrame(cmp_rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)


# =========================================================================
# Tab 2: Per Ticker
# =========================================================================

with tab_ticker:
    if trades_df is not None and len(trades_df) > 0 and "ticker" in trades_df.columns:
        all_tickers = sorted(trades_df["ticker"].unique().tolist())

        ticker_choice = st.selectbox(
            "Select Ticker", all_tickers, key="bt_ticker_select"
        )

        ticker_trades = trades_df[trades_df["ticker"] == ticker_choice]

        # Per-ticker metrics
        n_trades = len(ticker_trades)
        if n_trades > 0 and "pnl" in ticker_trades.columns:
            pnls = ticker_trades["pnl"].dropna()
            wins = (pnls > 0).sum()
            wr = wins / len(pnls) if len(pnls) > 0 else 0
            total_pnl = pnls.sum()

            # Build full daily PnL series (including zero-return days)
            # to match the portfolio-level Sharpe methodology.
            time_col = "exit_time" if "exit_time" in ticker_trades.columns else (
                "timestamp" if "timestamp" in ticker_trades.columns else None
            )
            eq_ts_col = "timestamp" if (equity_df is not None and "timestamp" in equity_df.columns) else None
            if time_col is not None and equity_df is not None and eq_ts_col is not None:
                trade_pnls = ticker_trades[[time_col, "pnl"]].copy()
                trade_pnls[time_col] = pd.to_datetime(trade_pnls[time_col])
                trade_pnls["date"] = trade_pnls[time_col].dt.normalize()
                daily_pnl = trade_pnls.groupby("date")["pnl"].sum()
                # Build full trading date index from equity curve
                eq_dates = pd.to_datetime(equity_df[eq_ts_col]).dt.normalize()
                full_date_idx = pd.DatetimeIndex(eq_dates.unique()).sort_values()
                daily_pnl = daily_pnl.reindex(full_date_idx, fill_value=0.0)
                std_pnl = daily_pnl.std() if len(daily_pnl) > 1 else 0
                mean_pnl = daily_pnl.mean()
            else:
                mean_pnl = pnls.mean()
                std_pnl = pnls.std() if len(pnls) > 1 else 0
            sharpe_t = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 1e-10 else 0

            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                st.metric("Sharpe", f"{sharpe_t:.2f}")
            with tc2:
                st.metric("Win Rate", f"{wr:.1%}")
            with tc3:
                st.metric("Trades", str(n_trades))
            with tc4:
                st.metric("Total PnL", f"{total_pnl:.4f}")

        # Per-ticker price chart with trade markers
        if len(ticker_trades) > 0:
            st.markdown(f"##### {ticker_choice} Price & Trades")

            # Load price data for this ticker
            freq = cfg.get("data", {}).get("freq_short", "15min")
            features_dir = Path(cfg.get("data", {}).get("features_dir", "data/features"))
            price_path = features_dir / f"{ticker_choice}_{freq}.csv"

            fig_t = go.Figure()
            has_price = False

            if price_path.exists():
                price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
                # Normalise column name
                close_col = "Close" if "Close" in price_df.columns else "close"
                if close_col in price_df.columns:
                    price_series = price_df[close_col]
                    # Filter to trade date range if timestamps available
                    if "timestamp" in ticker_trades.columns:
                        ts = pd.to_datetime(ticker_trades["timestamp"])
                        t_min, t_max = ts.min(), ts.max()
                        # Pad ±5 days for context
                        pad = pd.Timedelta(days=5)
                        price_series = price_series[
                            (price_series.index >= t_min - pad)
                            & (price_series.index <= t_max + pad)
                        ]
                    fig_t.add_trace(go.Scatter(
                        x=price_series.index,
                        y=price_series.values,
                        mode="lines",
                        name=f"{ticker_choice} Close",
                        line=dict(color="#888888", width=1),
                    ))
                    has_price = True

            # Trade entry markers (BUY / SHORT)
            sorted_trades = ticker_trades.sort_index()
            if "signal" in sorted_trades.columns and "timestamp" in sorted_trades.columns:
                buy_mask = sorted_trades["signal"].str.upper() == "BUY"
                short_mask = sorted_trades["signal"].str.upper() == "SHORT"

                if buy_mask.any():
                    buy_rows = sorted_trades[buy_mask]
                    fig_t.add_trace(go.Scatter(
                        x=pd.to_datetime(buy_rows["timestamp"]),
                        y=buy_rows["entry_price"].values if "entry_price" in buy_rows.columns else None,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=12, color="#3fb950"),
                        name="BUY Entry",
                    ))

                if short_mask.any():
                    short_rows = sorted_trades[short_mask]
                    fig_t.add_trace(go.Scatter(
                        x=pd.to_datetime(short_rows["timestamp"]),
                        y=short_rows["entry_price"].values if "entry_price" in short_rows.columns else None,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=12, color="#f85149"),
                        name="SHORT Entry",
                    ))

                # Exit markers at exit_timestamp (or entry timestamp fallback)
                if "exit_price" in sorted_trades.columns:
                    exited = sorted_trades.dropna(subset=["exit_price"])
                    if len(exited) > 0:
                        exit_x_col = "exit_timestamp" if "exit_timestamp" in exited.columns else "timestamp"
                        fig_t.add_trace(go.Scatter(
                            x=pd.to_datetime(exited[exit_x_col]),
                            y=exited["exit_price"].values,
                            mode="markers",
                            marker=dict(symbol="x", size=8, color="#d29922"),
                            name="Exit",
                        ))

            if not has_price and "entry_price" not in sorted_trades.columns:
                st.info(f"No price data found at `{price_path}`")

            fig_t.update_layout(
                **_DARK_LAYOUT,
                height=380,
                margin=dict(l=40, r=20, t=30, b=40),
                yaxis_title="Price ($)",
            )
            fig_t.update_xaxes(title="Date", type="date")
            st.plotly_chart(fig_t, use_container_width=True)

        # Per-ticker trade table
        with st.expander(f"{ticker_choice} Trades"):
            st.dataframe(ticker_trades, use_container_width=True, hide_index=True)

    else:
        st.info("No trade data with ticker information available.")


# =========================================================================
# Tab 3: Trade Log
# =========================================================================

with tab_trades:
    if trades_df is not None and len(trades_df) > 0:
        # --- Filters ---
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            tickers_available = (
                sorted(trades_df["ticker"].unique().tolist())
                if "ticker" in trades_df.columns else []
            )
            ticker_filter = st.multiselect(
                "Ticker Filter",
                tickers_available,
                default=tickers_available,
                key="bt_trade_ticker_filter",
            )

        with fc2:
            result_filter = st.radio(
                "Result",
                ["All", "Winners", "Losers"],
                horizontal=True,
                key="bt_trade_result_filter",
            )

        with fc3:
            if "timestamp" in trades_df.columns:
                try:
                    ts_series = pd.to_datetime(trades_df["timestamp"])
                    min_date = ts_series.min().date()
                    max_date = ts_series.max().date()
                except Exception:
                    min_date = date(2020, 1, 1)
                    max_date = date(2025, 12, 31)
            else:
                min_date = date(2020, 1, 1)
                max_date = date(2025, 12, 31)

            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                key="bt_trade_date_range",
            )

        # --- Apply filters ---
        filtered = trades_df.copy()

        if ticker_filter and "ticker" in filtered.columns:
            filtered = filtered[filtered["ticker"].isin(ticker_filter)]

        if result_filter == "Winners" and "pnl" in filtered.columns:
            filtered = filtered[filtered["pnl"] > 0]
        elif result_filter == "Losers" and "pnl" in filtered.columns:
            filtered = filtered[filtered["pnl"] <= 0]

        if "timestamp" in filtered.columns and len(date_range) == 2:
            try:
                ts = pd.to_datetime(filtered["timestamp"])
                start_dt, end_dt = date_range
                filtered = filtered[
                    (ts.dt.date >= start_dt) & (ts.dt.date <= end_dt)
                ]
            except Exception:
                pass

        st.caption(f"{len(filtered)} trades after filtering")

        # --- Ensure numeric columns for sorting ---
        for _nc in ["pnl", "pnl_dollars", "position_value", "bars_held"]:
            if _nc in filtered.columns:
                filtered[_nc] = pd.to_numeric(filtered[_nc], errors="coerce")

        # --- Sort controls (applied to full df before pagination) ---
        _sort_map = {
            "timestamp": "timestamp",
            "pnl_pct": "pnl",
            "pnl_dollars": "pnl_dollars",
            "bars_held": "bars_held",
            "ticker": "ticker",
            "position_value": "position_value",
        }
        _sort_options = {k: v for k, v in _sort_map.items() if v in filtered.columns}
        _sort_labels = list(_sort_options.keys())
        if not _sort_labels:
            _sort_labels = ["timestamp"]
            _sort_options = {"timestamp": "timestamp"}

        _sc1, _sc2 = st.columns([2, 2])
        with _sc1:
            sort_label = st.selectbox(
                "Sort by", _sort_labels,
                index=0,
                key="bt_trade_sort_col",
            )
        with _sc2:
            sort_order = st.radio(
                "Order", ["Descending", "Ascending"],
                horizontal=True,
                key="bt_trade_sort_order",
            )

        sort_col = _sort_options[sort_label]
        ascending = sort_order == "Ascending"
        try:
            filtered = filtered.sort_values(
                by=sort_col, ascending=ascending, na_position="last",
            ).reset_index(drop=True)
        except Exception:
            pass

        # --- Pagination ---
        page_size = 100
        total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input(
            "Page", min_value=1, max_value=total_pages,
            value=1, step=1, key="bt_trade_page",
        )
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered))
        st.caption(f"Showing {start_idx + 1}–{end_idx} of {len(filtered)}")

        # --- Build display dataframe (format AFTER sort+paginate) ---
        _display = filtered.iloc[start_idx:end_idx].copy()

        # Format position_value as $X,XXX.XX if present
        if "position_value" in _display.columns:
            _display["position_value"] = _display["position_value"].apply(
                lambda v: f"${v:,.2f}" if pd.notna(v) and v != 0 else ""
            )

        # Format pnl as percentage (X.XX%)
        if "pnl" in _display.columns:
            _display["pnl"] = _display["pnl"].apply(
                lambda v: f"{v * 100:+.2f}%" if pd.notna(v) else ""
            )

        # Format pnl_dollars as $X.XX with sign
        if "pnl_dollars" in _display.columns:
            _display["pnl_dollars"] = _display["pnl_dollars"].apply(
                lambda v: f"${v:+,.2f}" if pd.notna(v) else ""
            )

        # Format size to 2 decimal places if present
        if "size" in _display.columns:
            _display["size"] = _display["size"].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else ""
            )

        # Format conviction_tier as uppercase if present
        if "conviction_tier" in _display.columns:
            _display["conviction_tier"] = _display["conviction_tier"].apply(
                lambda v: str(v).upper() if pd.notna(v) and v else ""
            )

        # Preferred column order (only include columns that exist)
        _col_order = [
            "timestamp", "ticker", "signal", "conviction_tier",
            "size", "position_value",
            "pnl", "pnl_dollars", "exit_reason", "bars_held",
            "entry_price", "exit_price",
            "entry_tp_pct", "entry_sl_pct", "tranches_exited",
            "portfolio_value", "commission", "slippage",
            "exit_timestamp", "trade_id",
        ]
        _visible_cols = [c for c in _col_order if c in _display.columns]
        # Append any remaining columns not in the preferred order
        _remaining = [c for c in _display.columns if c not in _visible_cols]
        _visible_cols.extend(_remaining)

        _display = _display[_visible_cols]

        # Column config
        _col_config = {}
        if "conviction_tier" in _display.columns:
            _col_config["conviction_tier"] = st.column_config.TextColumn(
                "Conviction",
                help="HIGH = vol_prob > 0.7, MID = vol_prob 0.5–0.7",
            )
        if "position_value" in _display.columns:
            _col_config["position_value"] = st.column_config.TextColumn(
                "Pos Value",
                help="Actual dollars invested in this trade (portfolio_value x size)",
            )
        if "pnl" in _display.columns:
            _col_config["pnl"] = st.column_config.TextColumn(
                "PnL %",
                help="Return on this trade as percentage",
            )
        if "pnl_dollars" in _display.columns:
            _col_config["pnl_dollars"] = st.column_config.TextColumn(
                "PnL $",
                help="Dollar profit/loss on this trade",
            )

        st.dataframe(
            _display,
            use_container_width=True,
            hide_index=True,
            column_config=_col_config,
        )

        # --- Win / Loss donut chart ---
        if "pnl" in filtered.columns:
            pnls = filtered["pnl"].dropna()
            n_wins = int((pnls > 0).sum())
            n_losses = int((pnls <= 0).sum())

            if n_wins + n_losses > 0:
                st.markdown("##### Win / Loss Breakdown")

                fig_donut = go.Figure(data=[
                    go.Pie(
                        labels=["Winners", "Losers"],
                        values=[n_wins, n_losses],
                        hole=0.5,
                        marker=dict(colors=["#3fb950", "#f85149"]),
                        textinfo="label+percent+value",
                        textfont=dict(color="#e6edf3"),
                    )
                ])
                fig_donut.update_layout(
                    **_DARK_LAYOUT,
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No trade data available for this run.")


# =========================================================================
# Tab 4: Verdict & Attribution
# =========================================================================

with tab_verdict:
    from analytics.verdict import compute_verdict
    from analytics.attribution import by_exit_reason, by_conviction_tier
    from frontend.components.analytics_ui import (
        render_verdict_card,
        render_attribution_table,
        attribution_to_markdown,
    )

    verdict = compute_verdict(metrics)
    render_verdict_card(verdict)

    st.markdown("---")

    # Attribution by exit reason
    if trades_df is not None and len(trades_df) > 0:
        exit_attr = by_exit_reason(trades_df)
        render_attribution_table(exit_attr, title="Attribution by Exit Reason")

        # Copy as Markdown
        if len(exit_attr) > 0:
            md_exit = attribution_to_markdown(exit_attr)
            st.text_area(
                "Copy as Markdown",
                value=md_exit,
                height=200,
                key="bt_exit_md",
            )

        st.markdown("---")

        # Attribution by conviction tier
        conviction_attr = by_conviction_tier(trades_df)
        if len(conviction_attr) > 0:
            render_attribution_table(conviction_attr, title="Attribution by Conviction Tier")
            md_conv = attribution_to_markdown(conviction_attr)
            st.text_area(
                "Copy as Markdown",
                value=md_conv,
                height=150,
                key="bt_conv_md",
            )
        else:
            st.info(
                "Conviction tier data not available for this run. "
                "Runs saved with the full reporter include conviction_tier."
            )
    else:
        st.info("No trade data available for attribution.")


# =========================================================================
# Tab 5: Trade Markers
# =========================================================================

with tab_markers:
    from frontend.components.analytics_ui import render_equity_with_markers

    initial_capital_markers = cfg.get("backtest", {}).get("initial_capital", 100_000)
    render_equity_with_markers(equity_df, trades_df, initial_capital_markers)


# =========================================================================
# Tab 6: Diagnostics
# =========================================================================

with tab_diag:
    from diagnostics.engine_hooks import collect_diagnostics
    from frontend.components.diagnostics_panel import render_diagnostics_panel

    if trades_df is not None and len(trades_df) > 0:
        _diag_report = collect_diagnostics(trades_df)
        render_diagnostics_panel(_diag_report)
    else:
        st.info("No trade data available for diagnostics.")


# ---------------------------------------------------------------------------
# Download section
# ---------------------------------------------------------------------------

st.markdown("---")

dl1, dl2, dl3, _ = st.columns([1, 1, 1, 2])

with dl1:
    if trades_df is not None and len(trades_df) > 0:
        csv_buf = io.StringIO()
        trades_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Trades CSV",
            data=csv_buf.getvalue(),
            file_name="trades.csv",
            mime="text/csv",
            use_container_width=True,
            key="bt_dl_trades",
        )

with dl2:
    if equity_df is not None and len(equity_df) > 0:
        csv_buf = io.StringIO()
        equity_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Equity CSV",
            data=csv_buf.getvalue(),
            file_name="equity_curve.csv",
            mime="text/csv",
            use_container_width=True,
            key="bt_dl_equity",
        )

with dl3:
    charts_dir = run_dir / "charts" if run_dir.exists() else None
    if charts_dir and charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        if chart_files:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for cf in chart_files:
                    zf.write(cf, cf.name)
            zip_buf.seek(0)
            st.download_button(
                "📥 Charts ZIP",
                data=zip_buf.getvalue(),
                file_name="charts.zip",
                mime="application/zip",
                use_container_width=True,
                key="bt_dl_charts",
            )
