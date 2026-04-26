"""
Streamlit UI components for analytics verdict, attribution tables,
and trade markers on equity curves.

All components follow the existing ApexQuant dark-theme patterns
(plotly_dark, #0e1117 background, Outfit font).
"""

__all__ = [
    "render_verdict_card",
    "render_attribution_table",
    "render_equity_with_markers",
    "attribution_to_markdown",
]

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.verdict import Verdict, VerdictLevel


# ---------------------------------------------------------------------------
# Dark layout (matches 5_③_Backtest_Analysis.py)
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e"),
    xaxis=dict(gridcolor="#2d333b"),
    yaxis=dict(gridcolor="#2d333b"),
)

_VERDICT_COLORS = {
    VerdictLevel.GREEN: ("#3fb950", "#0d1117", "#238636"),
    VerdictLevel.YELLOW: ("#d29922", "#0d1117", "#9e6a03"),
    VerdictLevel.RED: ("#f85149", "#0d1117", "#da3633"),
}


# ---------------------------------------------------------------------------
# Verdict card
# ---------------------------------------------------------------------------

def render_verdict_card(verdict: Verdict) -> None:
    """Render a traffic-light verdict badge with details."""
    fg, bg, border = _VERDICT_COLORS[verdict.level]

    st.markdown(
        f"""
        <div style="
            background: {bg};
            border: 2px solid {border};
            border-radius: 10px;
            padding: 16px 24px;
            margin-bottom: 16px;
        ">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="
                    display:inline-block;
                    width:14px; height:14px;
                    border-radius:50%;
                    background:{fg};
                    box-shadow: 0 0 8px {fg}80;
                "></span>
                <span style="
                    font-family:'Outfit',sans-serif;
                    font-size:20px;
                    font-weight:700;
                    color:{fg};
                    letter-spacing:0.05em;
                ">{verdict.level.value} &mdash; {verdict.label}</span>
            </div>
            {"".join(
                f'<div style="color:#8b949e; font-size:13px; margin-top:6px; padding-left:26px;">&bull; {d}</div>'
                for d in verdict.details
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Attribution table
# ---------------------------------------------------------------------------

def render_attribution_table(
    df: pd.DataFrame,
    title: str = "Attribution",
    group_col: str | None = None,
) -> None:
    """Render an attribution DataFrame as a styled Streamlit table.

    Args:
        df: Output of ``by_exit_reason`` or ``by_conviction_tier``.
        title: Section title.
        group_col: Name of the grouping column (auto-detected if None).
    """
    if df.empty:
        st.info(f"No {title.lower()} data available.")
        return

    st.markdown(f"##### {title}")

    display = df.copy()

    # Format percentage columns for display
    for col in ("win_rate", "avg_pnl_pct", "contribution_pct"):
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.2%}")
    if "sum_pnl_pct" in display.columns:
        display["sum_pnl_pct"] = display["sum_pnl_pct"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display, use_container_width=True, hide_index=True)


def attribution_to_markdown(df: pd.DataFrame) -> str:
    """Convert an attribution DataFrame to a GitHub-flavored markdown table.

    Args:
        df: Output of ``by_exit_reason`` or ``by_conviction_tier``.

    Returns:
        GFM table string.
    """
    if df.empty:
        return ""

    formatted = df.copy()
    for col in ("win_rate", "avg_pnl_pct", "contribution_pct"):
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: f"{x:.2%}")
    if "sum_pnl_pct" in formatted.columns:
        formatted["sum_pnl_pct"] = formatted["sum_pnl_pct"].apply(lambda x: f"{x:.4f}")

    headers = formatted.columns.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"

    rows: list[str] = []
    for _, row in formatted.iterrows():
        row_str = "| " + " | ".join(str(row[h]) for h in headers) + " |"
        rows.append(row_str)

    return "\n".join([header_line, sep_line] + rows)


# ---------------------------------------------------------------------------
# Equity curve with trade markers
# ---------------------------------------------------------------------------

def render_equity_with_markers(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float = 100_000,
) -> None:
    """Render equity curve with entry/exit markers overlaid.

    Entry markers: triangle-up (BUY green) / triangle-down (SHORT red).
    Exit markers: diamond coloured by exit_reason.

    Args:
        equity_df: DataFrame with ``timestamp`` and ``portfolio_value``.
        trades_df: Trades DataFrame with ``timestamp``, ``signal``,
            ``exit_timestamp``, ``exit_reason``, ``pnl`` columns.
        initial_capital: For the reference line.
    """
    if equity_df is None or len(equity_df) == 0:
        st.info("No equity data available.")
        return

    ts_col = "timestamp" if "timestamp" in equity_df.columns else equity_df.columns[0]
    val_col = "portfolio_value" if "portfolio_value" in equity_df.columns else equity_df.columns[1]

    fig = go.Figure()

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=equity_df[ts_col],
            y=equity_df[val_col],
            mode="lines",
            name="Equity",
            line=dict(color="#3fb950", width=2),
            hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
        )
    )

    # Drawdown shading
    vals = equity_df[val_col].values.astype(float)
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
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Initial capital line
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color="#8b949e",
        line_width=1,
    )

    # Trade markers (only if we have trade data)
    if trades_df is not None and len(trades_df) > 0:
        _add_trade_markers(fig, equity_df, trades_df, ts_col, val_col)

    fig.update_layout(
        **_DARK_LAYOUT,
        height=500,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def _add_trade_markers(
    fig: go.Figure,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    ts_col: str,
    val_col: str,
) -> None:
    """Add entry/exit markers to the equity curve figure."""
    # Build a timestamp→equity lookup for y-positioning
    eq_ts = pd.to_datetime(equity_df[ts_col])
    eq_vals = equity_df[val_col].values.astype(float)

    # Entry markers
    if "timestamp" in trades_df.columns and "signal" in trades_df.columns:
        entries = trades_df[["timestamp", "signal"]].copy()
        entries["timestamp"] = pd.to_datetime(entries["timestamp"])

        buys = entries[entries["signal"].isin(["BUY", "COVER"])]
        shorts = entries[entries["signal"].isin(["SHORT", "SELL"])]

        if len(buys) > 0:
            buy_y = _nearest_equity(buys["timestamp"], eq_ts, eq_vals)
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buy_y,
                    mode="markers",
                    name="BUY entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=7,
                        color="#3fb950",
                        line=dict(width=0.5, color="#3fb950"),
                    ),
                    hovertemplate="BUY @ %{x}<extra></extra>",
                )
            )

        if len(shorts) > 0:
            short_y = _nearest_equity(shorts["timestamp"], eq_ts, eq_vals)
            fig.add_trace(
                go.Scatter(
                    x=shorts["timestamp"],
                    y=short_y,
                    mode="markers",
                    name="SHORT entry",
                    marker=dict(
                        symbol="triangle-down",
                        size=7,
                        color="#f85149",
                        line=dict(width=0.5, color="#f85149"),
                    ),
                    hovertemplate="SHORT @ %{x}<extra></extra>",
                )
            )

    # Exit markers
    if "exit_timestamp" in trades_df.columns and "exit_reason" in trades_df.columns:
        exits = trades_df[["exit_timestamp", "exit_reason", "pnl"]].dropna(subset=["exit_timestamp"]).copy()
        exits["exit_timestamp"] = pd.to_datetime(exits["exit_timestamp"])

        if len(exits) > 0:
            exit_y = _nearest_equity(exits["exit_timestamp"], eq_ts, eq_vals)
            colors = exits["pnl"].apply(
                lambda p: "#3fb950" if p is not None and p > 0 else "#f85149"
            )
            fig.add_trace(
                go.Scatter(
                    x=exits["exit_timestamp"],
                    y=exit_y,
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="diamond",
                        size=5,
                        color=colors,
                        line=dict(width=0.5, color="#8b949e"),
                    ),
                    text=exits["exit_reason"],
                    hovertemplate="Exit: %{text}<br>%{x}<extra></extra>",
                )
            )


def _nearest_equity(
    timestamps: pd.Series,
    eq_ts: pd.Series,
    eq_vals: np.ndarray,
) -> list[float]:
    """Map trade timestamps to nearest equity curve values via searchsorted."""
    eq_idx = eq_ts.values.astype("int64")
    trade_idx = timestamps.values.astype("int64")
    positions = np.searchsorted(eq_idx, trade_idx, side="right") - 1
    positions = np.clip(positions, 0, len(eq_vals) - 1)
    return [float(eq_vals[p]) for p in positions]
