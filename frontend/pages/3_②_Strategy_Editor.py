"""
Strategy Editor page -- write, test, and run trading strategies.

Two-column layout: code editor on the left, backtest controls and
results on the right.  Users can load built-in or custom strategies,
edit them in-browser, save to ``strategies/user/``, and run backtests
with optional EMA+RSI baseline comparison.
"""

__all__: list[str] = []

import io
import json
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import sys, os
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from components.styles import inject_global_css

try:
    from streamlit_ace import st_ace
    _HAS_ACE = True
except ImportError:
    _HAS_ACE = False

logger.info("Strategy Editor page loaded")

if "config" not in st.session_state:
    from config.default import CONFIG
    st.session_state["config"] = deepcopy(CONFIG)

cfg = st.session_state["config"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_STRATEGIES_DIR = Path("strategies/user")

_DEFAULT_TEMPLATE = '''\
from strategies.base import BaseStrategy, Signal
from data.bar import Bar


class MyStrategy(BaseStrategy):
    """Custom strategy template."""

    name = "my_strategy"

    def on_bar(self, bar: Bar) -> Signal:
        # --- SignalsProxy (path-based access) ---
        # Specific model by folder path:
        #   bar.signals["layer1/volatility/lightgbm_v3"]   -> float 0-1
        #   bar.signals.get("layer2/tp_bottom/multiscale_cnn_v1", 0.5)
        #
        # Per-layer aggregation (mean of all models in that layer):
        #   bar.signals.layer1   -> float 0-1 (volatility)
        #   bar.signals.layer2   -> float 0-1 (turning-point)
        #   bar.signals.layer3   -> float 0-1 (meta-label)
        #
        # --- Legacy access (still works) ---
        #   bar.get_prob("vol_prob", 0.5)
        #
        # Aggregated signal (if aggregator ran):
        #   bar.aggregated_signal.direction   [-1.0, 1.0]
        #   bar.aggregated_signal.strength    [0.0, 1.0]
        #   bar.aggregated_signal.confidence  [0.0, 1.0]
        #
        # Technical indicators:
        #   bar.ema_8, bar.ema_21, bar.ema_50
        #   bar.rsi_14, bar.atr_14, bar.macd
        #   bar.bb_upper, bar.bb_lower, bar.volume_ratio
        #
        # OHLCV:
        #   bar.open, bar.high, bar.low, bar.close, bar.volume

        signals = bar.signals
        vol = signals.layer1      # aggregated volatility probability
        tp = signals.layer2       # aggregated turning-point probability

        if vol > 0.7 and tp > 0.6:
            return Signal.BUY
        if vol < 0.3 and tp < 0.4:
            return Signal.SHORT
        return Signal.HOLD
'''

# ---------------------------------------------------------------------------
# Monospace CSS for the code editor (fallback when streamlit-ace unavailable)
# ---------------------------------------------------------------------------

if not _HAS_ACE:
    st.markdown(
        """
        <style>
        textarea[aria-label="Strategy Code"] {
            font-family: 'Courier New', 'Consolas', 'Monaco', monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            tab-size: 4 !important;
            background-color: #1e1e2e !important;
            color: #cdd6f4 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

inject_global_css()
st.markdown("""
<div style="padding: 0 0 20px 0; border-bottom: 1px solid #21262d; margin-bottom: 24px;">
  <div style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600;
              letter-spacing:0.1em; text-transform:uppercase; color:#8b949e; margin-bottom:4px;">
    ② RESEARCH
  </div>
  <h1 style="margin:0; font-family:'Outfit',sans-serif;">Strategy Editor</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_user_strategies() -> list[str]:
    """List .py files in strategies/user/."""
    if not _USER_STRATEGIES_DIR.exists():
        return []
    return sorted(
        p.stem for p in _USER_STRATEGIES_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def _load_builtin_source(name: str) -> str:
    """Load source code of a built-in strategy."""
    if name == "AI Strategy (default)":
        p = Path("strategies/builtin/ai_strategy.py")
    elif name == "EMA+RSI Baseline":
        p = Path("strategies/builtin/technical.py")
    else:
        return _DEFAULT_TEMPLATE

    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"# File not found: {p}"


def _load_user_source(name: str) -> str:
    """Load source code of a user strategy."""
    p = _USER_STRATEGIES_DIR / f"{name}.py"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"# File not found: {p}"


# ---------------------------------------------------------------------------
# Build strategy options
# ---------------------------------------------------------------------------

builtin_options = ["New Strategy (template)", "AI Strategy (default)", "EMA+RSI Baseline"]
user_files = _list_user_strategies()
user_options = [f"user/{n}" for n in user_files]
all_options = builtin_options + user_options

# ---------------------------------------------------------------------------
# Layout: 60% editor, 40% backtest
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([3, 2])

# =========================================================================
# LEFT COLUMN -- Strategy Editor
# =========================================================================

with col_left:
    st.subheader("Strategy Editor")

    # --- Load strategy selector ---
    load_choice = st.selectbox(
        "Load Strategy",
        all_options,
        key="strat_load_choice",
    )

    # Determine initial code
    if "strat_editor_code" not in st.session_state:
        st.session_state["strat_editor_code"] = _DEFAULT_TEMPLATE

    # Load on selection change
    if load_choice == "New Strategy (template)":
        code_to_load = _DEFAULT_TEMPLATE
    elif load_choice in ("AI Strategy (default)", "EMA+RSI Baseline"):
        code_to_load = _load_builtin_source(load_choice)
    elif load_choice.startswith("user/"):
        code_to_load = _load_user_source(load_choice.removeprefix("user/"))
    else:
        code_to_load = _DEFAULT_TEMPLATE

    # Track selection changes to reload content
    prev_choice = st.session_state.get("_strat_prev_choice")
    if prev_choice != load_choice:
        st.session_state["strat_editor_code"] = code_to_load
        st.session_state["_strat_prev_choice"] = load_choice

    # --- Code editor ---
    if _HAS_ACE:
        code = st_ace(
            value=st.session_state["strat_editor_code"],
            language="python",
            theme="monokai",
            font_size=14,
            height=500,
            key="strat_code_area",
            show_gutter=True,
            wrap=False,
            auto_update=False,
        )
    else:
        code = st.text_area(
            "Strategy Code",
            value=st.session_state["strat_editor_code"],
            height=500,
            key="strat_code_area",
        )
    # Sync back
    st.session_state["strat_editor_code"] = code

    # --- Validate button ---
    if st.button("✓ Validate Syntax", key="strat_validate_btn"):
        try:
            compile(code, "<strategy_editor>", "exec")
            st.success("Syntax OK — no errors found.")
        except SyntaxError as e:
            st.error(f"**Syntax Error on line {e.lineno}:** {e.msg}")
            lines = code.split("\n")
            if e.lineno and e.lineno <= len(lines):
                st.code(f"Line {e.lineno}: {lines[e.lineno - 1]}", language="python")

    # --- AI Strategy Generator ---
    with st.expander("💬 Describe strategy in plain English"):
        nl_description = st.text_area(
            "Strategy description",
            placeholder=(
                "e.g. Buy when AI direction is strongly bullish and RSI is oversold. "
                "Short when direction is bearish and we're near a CNN-detected top."
            ),
            height=100,
            key="strat_nl_input",
        )

        if st.button("✨ Generate Code", key="strat_gen_btn"):
            llm_cfg = cfg.get("llm", {})
            if llm_cfg.get("enabled", False) and nl_description.strip():
                try:
                    from llm.strategy_generator import generate_strategy_code

                    with st.spinner("Generating strategy code..."):
                        generated = generate_strategy_code(
                            nl_description, config=cfg
                        )
                    st.session_state["strat_editor_code"] = generated
                    st.success("Code generated! Review and edit above.")
                    st.rerun()
                except ImportError:
                    st.error(
                        "LLM strategy generator module not found. "
                        "Ensure `llm/strategy_generator.py` is implemented."
                    )
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
            else:
                st.info(
                    "LLM integration not enabled. "
                    "Set `config['llm']['enabled'] = True` in Configuration to use this feature."
                )

    # --- Save strategy ---
    save_col1, save_col2 = st.columns([2, 3])
    with save_col1:
        save_name = st.text_input(
            "Strategy name",
            value="my_strategy",
            key="strat_save_name",
        )
    with save_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Strategy", use_container_width=True, key="strat_save_btn"):
            safe_name = "".join(
                c if c.isalnum() or c == "_" else "_"
                for c in save_name.strip()
            )
            if not safe_name:
                st.error("Please enter a valid strategy name.")
            else:
                _USER_STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
                save_path = _USER_STRATEGIES_DIR / f"{safe_name}.py"
                save_path.write_text(code, encoding="utf-8")
                st.success(f"Saved to `{save_path}`")
                logger.info("Strategy saved to {}", save_path)

# =========================================================================
# RIGHT COLUMN -- Backtest Settings & Results
# =========================================================================

with col_right:
    st.subheader("Run Backtest")

    # --- Date inputs ---
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2022, 1, 1),
            key="strat_bt_start",
        )
    with date_col2:
        end_date = st.date_input(
            "End Date",
            value=date(2022, 12, 31),
            key="strat_bt_end",
        )

    # --- Comparison selector ---
    compare_against = st.selectbox(
        "Compare Against",
        ["EMA+RSI Baseline", "None"],
        key="strat_compare",
    )

    # --- Run button ---
    run_btn = st.button(
        "▶ Run Backtest",
        use_container_width=True,
        key="strat_run_btn",
        type="primary",
    )

    # --- Execution ---
    if run_btn:
        log_container = st.empty()
        log_container.code("Validating strategy code...", language="text")

        try:
            # Step 1: exec() user code to extract strategy class
            from strategies.base import BaseStrategy

            namespace: dict = {}
            exec(compile(code, "<strategy_editor>", "exec"), namespace)

            # Step 2: Find BaseStrategy subclass
            user_class = None
            for obj in namespace.values():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseStrategy)
                    and obj is not BaseStrategy
                ):
                    user_class = obj
                    break

            if user_class is None:
                st.error(
                    "No BaseStrategy subclass found in your code. "
                    "Define a class that inherits from `BaseStrategy` "
                    "and implements `on_bar()`."
                )
            else:
                log_container.code(
                    f"Found strategy: {user_class.__name__}. Running backtest...",
                    language="text",
                )

                # Step 3: Run backtest with the strategy class
                from backtest.runner import run_backtest

                with st.spinner("Running your strategy..."):
                    result = run_backtest(
                        cfg,
                        strategy_name=user_class.name if hasattr(user_class, 'name') else "user_strategy",
                        strategy_class=user_class,
                        start_date=str(start_date) if start_date else None,
                        end_date=str(end_date) if end_date else None,
                        save_results=True,
                    )

                st.session_state["strat_bt_result"] = result
                log_container.code(
                    f"Backtest complete: {result.metrics.get('total_trades', 0)} trades, "
                    f"Sharpe={result.metrics.get('sharpe_ratio', 0):.2f}, "
                    f"Return={result.metrics.get('total_return', 0):.2%}",
                    language="text",
                )

                # Step 4: Run comparison if selected
                if compare_against == "EMA+RSI Baseline":
                    with st.spinner("Running EMA+RSI baseline..."):
                        baseline = run_backtest(
                            cfg,
                            strategy_name="technical",
                            start_date=str(start_date) if start_date else None,
                            end_date=str(end_date) if end_date else None,
                            save_results=False,
                        )
                    st.session_state["strat_bt_baseline"] = baseline
                else:
                    st.session_state.pop("strat_bt_baseline", None)

                st.success("Backtest complete!")

        except SyntaxError as exc:
            log_container.code(f"SYNTAX ERROR: {exc}", language="text")
            st.error(f"**Syntax Error on line {exc.lineno}:** {exc.msg}")
            lines = code.split("\n")
            if exc.lineno and exc.lineno <= len(lines):
                st.code(f"Line {exc.lineno}: {lines[exc.lineno - 1]}", language="python")
        except Exception as exc:
            log_container.code(f"ERROR: {exc}", language="text")
            st.error(f"**{type(exc).__name__}:** {exc}")
            logger.exception("Strategy backtest failed")

    # --- Results display ---
    st.markdown("---")

    result = st.session_state.get("strat_bt_result")
    if result is None:
        st.info("Write a strategy and click **Run Backtest** to see results.")
        st.stop()

    # --- Metric cards ---
    m = result.metrics
    if m:
        mc1, mc2 = st.columns(2)
        mc3, mc4 = st.columns(2)

        with mc1:
            st.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
        with mc2:
            st.metric("Win Rate", f"{m.get('win_rate', 0):.1%}")
        with mc3:
            st.metric("Max Drawdown", f"{m.get('max_drawdown', 0):.2%}")
        with mc4:
            st.metric("Trades", str(m.get("total_trades", 0)))

    # --- Equity curve ---
    st.markdown("##### Equity Curve")

    try:
        import plotly.graph_objects as go

        equity = result.equity_curve
        if len(equity) > 1 and isinstance(equity.index, pd.DatetimeIndex):
            # Resample to daily, convert to % return for readable Y-axis
            eq_daily = equity.resample("1D").last().dropna()
            initial = eq_daily.iloc[0]
            eq_pct = (eq_daily / initial - 1.0) * 100  # percent

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=eq_pct.index,
                    y=eq_pct.values,
                    mode="lines",
                    name="Your Strategy",
                    line=dict(color="#3fb950", width=2),
                )
            )

            # Baseline overlay — aligned to same daily index
            baseline = st.session_state.get("strat_bt_baseline")
            if baseline and len(baseline.equity_curve) > 1 and isinstance(baseline.equity_curve.index, pd.DatetimeIndex):
                bl_daily = baseline.equity_curve.resample("1D").last().dropna()
                bl_initial = bl_daily.iloc[0]
                bl_pct = (bl_daily / bl_initial - 1.0) * 100
                # Reindex to shared date range
                shared_idx = eq_pct.index.union(bl_pct.index).sort_values()
                bl_aligned = bl_pct.reindex(shared_idx).ffill().bfill()

                fig.add_trace(
                    go.Scatter(
                        x=bl_aligned.index,
                        y=bl_aligned.values,
                        mode="lines",
                        name="EMA+RSI Baseline",
                        line=dict(color="#ff9800", width=2, dash="dash"),
                    )
                )

            fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1)

            # Force Y-axis to show full range of both curves
            y_min = eq_pct.min()
            y_max = eq_pct.max()
            if baseline and len(baseline.equity_curve) > 0:
                y_min = min(y_min, bl_aligned.min())
                y_max = max(y_max, bl_aligned.max())
            y_pad = max(0.1, (y_max - y_min) * 0.15)  # at least ±0.1%

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#161b22",
                height=320,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title="Date",
                yaxis_title="Return (%)",
                font=dict(color="#8b949e"),
                xaxis=dict(gridcolor="#2d333b"),
                yaxis=dict(
                    gridcolor="#2d333b",
                    zeroline=True,
                    range=[y_min - y_pad, y_max + y_pad],
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity data to display.")

    except ImportError:
        st.warning("Install plotly for interactive charts: `pip install plotly`")

    # --- Metrics comparison table ---
    baseline = st.session_state.get("strat_bt_baseline")
    if baseline:
        st.markdown("##### Metrics Comparison")

        metric_names = [
            ("total_return", "Total Return", True),
            ("sharpe_ratio", "Sharpe Ratio", False),
            ("sortino_ratio", "Sortino Ratio", False),
            ("max_drawdown", "Max Drawdown", True),
            ("win_rate", "Win Rate", True),
            ("profit_factor", "Profit Factor", False),
            ("total_trades", "Total Trades", False),
            ("avg_trade_pnl", "Avg Trade PnL", False),
            ("avg_bars_held", "Avg Bars Held", False),
        ]

        cmp_rows = []
        for key, label, is_pct in metric_names:
            user_val = result.metrics.get(key)
            base_val = baseline.metrics.get(key)
            qc_val = {"sharpe_ratio": -1.12}.get(key)

            def _fmt(v: float | int | None, pct: bool = False) -> str:
                if v is None:
                    return "--"
                if isinstance(v, int):
                    return str(v)
                if pct:
                    return f"{v:.2%}"
                return f"{v:.4f}"

            row: dict[str, str] = {
                "Metric": label,
                "Your Strategy": _fmt(user_val, is_pct),
                "EMA+RSI": _fmt(base_val, is_pct),
            }
            if qc_val is not None:
                row["QC Baseline"] = _fmt(qc_val, is_pct)
            cmp_rows.append(row)

        st.dataframe(
            pd.DataFrame(cmp_rows),
            use_container_width=True,
            hide_index=True,
        )

    # --- Trade log ---
    trades = result.trades
    if trades:
        with st.expander("Trade Log"):
            # Ticker filter
            all_tickers = sorted({t.ticker for t in trades})
            ticker_filter = st.selectbox(
                "Filter by Ticker",
                ["All"] + all_tickers,
                key="strat_trade_filter",
            )

            filtered = trades
            if ticker_filter != "All":
                filtered = [t for t in trades if t.ticker == ticker_filter]

            trade_rows = []
            for t in filtered:
                trade_rows.append({
                    "ID": t.trade_id[:8] if len(t.trade_id) > 8 else t.trade_id,
                    "Ticker": t.ticker,
                    "Signal": t.signal.value,
                    "Entry": f"${t.entry_price:.2f}",
                    "Exit": f"${t.exit_price:.2f}" if t.exit_price else "--",
                    "PnL": f"{t.pnl:.4f}" if t.pnl is not None else "--",
                    "Reason": t.exit_reason or "--",
                    "Bars": t.bars_held,
                })

            # Pagination
            page_size = 50
            total_rows = len(trade_rows)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)

            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="strat_trade_page",
            )
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)

            st.caption(f"Showing {start_idx + 1}–{end_idx} of {total_rows} trades")

            st.dataframe(
                pd.DataFrame(trade_rows[start_idx:end_idx]),
                use_container_width=True,
                hide_index=True,
            )

    # --- Download buttons ---
    st.markdown("---")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if trades:
            trade_csv_rows = []
            for t in trades:
                trade_csv_rows.append({
                    "trade_id": t.trade_id,
                    "ticker": t.ticker,
                    "signal": t.signal.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                    "bars_held": t.bars_held,
                    "size": t.size,
                    "commission": t.commission,
                    "slippage": t.slippage,
                })
            csv_buf = io.StringIO()
            pd.DataFrame(trade_csv_rows).to_csv(csv_buf, index=False)

            st.download_button(
                "📥 Download Trades CSV",
                data=csv_buf.getvalue(),
                file_name="trades.csv",
                mime="text/csv",
                use_container_width=True,
                key="strat_dl_trades",
            )

    with dl_col2:
        equity = result.equity_curve
        if len(equity) > 0:
            eq_buf = io.StringIO()
            eq_df = pd.DataFrame({
                "timestamp": equity.index,
                "portfolio_value": equity.values,
            })
            eq_df.to_csv(eq_buf, index=False)

            st.download_button(
                "📥 Download Equity CSV",
                data=eq_buf.getvalue(),
                file_name="equity_curve.csv",
                mime="text/csv",
                use_container_width=True,
                key="strat_dl_equity",
            )
