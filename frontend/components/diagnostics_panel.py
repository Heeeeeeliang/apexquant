"""
Streamlit component for rendering diagnostics results.

Renders trade quality metrics, feature drift tables, and timing info
from a :class:`DiagnosticsReport`.

Usage::

    from frontend.components.diagnostics_panel import render_diagnostics_panel
    render_diagnostics_panel(diagnostics_report)
"""

__all__ = ["render_diagnostics_panel"]

import pandas as pd
import streamlit as st

from diagnostics.engine_hooks import DiagnosticsReport


def render_diagnostics_panel(report: DiagnosticsReport) -> None:
    """Render a full diagnostics panel from a DiagnosticsReport.

    Args:
        report: Output of :func:`diagnostics.engine_hooks.collect_diagnostics`.
    """
    st.caption(f"Diagnostics collected in {report.collection_time_ms:.1f}ms")

    _render_equity_scan(report)
    _render_trade_clustering(report)
    _render_pnl_autocorrelation(report)
    _render_trade_quality(report)
    _render_feature_drift(report)


def _render_trade_quality(report: DiagnosticsReport) -> None:
    """Render trade quality section."""
    tq = report.trade_quality
    if tq.total_trades == 0:
        st.info("No trade data for quality analysis.")
        return

    st.markdown("##### Streak Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Max Win Streak", tq.max_win_streak)
    with c2:
        st.metric("Avg Win Streak", f"{tq.avg_win_streak:.1f}")
    with c3:
        st.metric("Max Loss Streak", tq.max_loss_streak)
    with c4:
        st.metric("Avg Loss Streak", f"{tq.avg_loss_streak:.1f}")

    # Hourly PnL
    if tq.hourly_pnl:
        st.markdown("##### Hourly Edge")
        hours = sorted(tq.hourly_pnl.keys())
        vals = [tq.hourly_pnl[h] for h in hours]
        colors = ["#3fb950" if v > 0 else "#f85149" for v in vals]

        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=[f"{h}:00" for h in hours],
            y=[v * 100 for v in vals],  # convert to percentage
            marker_color=colors,
            hovertemplate="Hour: %{x}<br>Avg PnL: %{y:.3f}%<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            font=dict(color="#8b949e"),
            height=250,
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis_title="Hour",
            yaxis_title="Avg PnL (%)",
            yaxis=dict(gridcolor="#2d333b"),
        )
        st.plotly_chart(fig, use_container_width=True)

        if tq.best_hour is not None:
            st.caption(
                f"Best hour: {tq.best_hour}:00 "
                f"({tq.hourly_pnl[tq.best_hour]*100:.3f}%) | "
                f"Worst hour: {tq.worst_hour}:00 "
                f"({tq.hourly_pnl[tq.worst_hour]*100:.3f}%)"
            )

    # Ticker concentration
    if tq.ticker_trade_counts:
        st.markdown("##### Ticker Concentration")
        ticker_df = pd.DataFrame([
            {
                "Ticker": t,
                "Trades": tq.ticker_trade_counts[t],
                "Sum PnL %": f"{tq.ticker_pnl_sums.get(t, 0)*100:.2f}%",
            }
            for t in sorted(tq.ticker_trade_counts.keys())
        ])
        st.dataframe(ticker_df, use_container_width=True, hide_index=True)

    # Exit reason win rates
    if tq.exit_reason_win_rates:
        st.markdown("##### Exit Reason Win Rates")
        er_df = pd.DataFrame([
            {
                "Exit Reason": r,
                "Win Rate": f"{wr:.1%}",
            }
            for r, wr in sorted(
                tq.exit_reason_win_rates.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ])
        st.dataframe(er_df, use_container_width=True, hide_index=True)

    # Avg bars in losing trades
    if tq.avg_bars_to_recover > 0:
        st.caption(f"Avg bars held in losing trades: {tq.avg_bars_to_recover:.1f}")


def _render_equity_scan(report: DiagnosticsReport) -> None:
    """Render equity curve health section."""
    eq = report.equity_scan
    if eq.equity_points < 2:
        return

    st.markdown("##### Equity Curve Health")

    health_colors = {"good": "#3fb950", "caution": "#d29922", "poor": "#f85149"}
    color = health_colors.get(eq.health, "#8b949e")
    st.markdown(
        f'<span style="color:{color}; font-weight:600;">{eq.health.upper()}</span>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Time Underwater", f"{eq.pct_time_underwater:.0%}")
    with c2:
        st.metric("Max UW Days", eq.max_underwater_days)
    with c3:
        st.metric("Flat Days %", f"{eq.flat_pct:.0%}")
    with c4:
        st.metric("DD Recovery Days", eq.worst_dd_recovery_days)

    if eq.current_underwater_days > 0:
        st.caption(f"Currently underwater for {eq.current_underwater_days} days")

    st.markdown("")


def _render_trade_clustering(report: DiagnosticsReport) -> None:
    """Render trade clustering section."""
    cl = report.trade_clustering
    if cl.total_trades == 0:
        return

    st.markdown("##### Trade Clustering")

    if cl.has_clustering:
        st.markdown(
            '<span style="color:#d29922; font-weight:600;">CLUSTERING DETECTED</span>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Max/Day", cl.max_trades_per_day)
    with c2:
        st.metric("Avg/Day", f"{cl.avg_trades_per_day:.1f}")
    with c3:
        st.metric("Burst Days", f"{cl.burst_days} ({cl.burst_pct:.0%})")
    with c4:
        st.metric("Median Gap", f"{cl.median_gap_hours:.1f}h")

    if cl.busiest_day:
        st.caption(f"Busiest day: {cl.busiest_day} ({cl.max_trades_per_day} trades)")

    st.markdown("")


def _render_pnl_autocorrelation(report: DiagnosticsReport) -> None:
    """Render PnL autocorrelation section."""
    ac = report.pnl_autocorrelation
    if ac.n_trades < 10:
        return

    st.markdown("##### PnL Autocorrelation")

    regime_colors = {"momentum": "#3fb950", "mean_revert": "#d29922", "none": "#8b949e"}
    color = regime_colors.get(ac.regime_signal, "#8b949e")
    label = ac.regime_signal.upper().replace("_", " ")
    sig = " (significant)" if ac.is_significant else ""
    st.markdown(
        f'Lag-1: **{ac.lag1_autocorr:.3f}** — '
        f'<span style="color:{color};">{label}{sig}</span>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("P(W|W)", f"{ac.p_win_after_win:.1%}")
    with c2:
        st.metric("P(W|L)", f"{ac.p_win_after_loss:.1%}")
    with c3:
        st.metric("P(L|W)", f"{ac.p_loss_after_win:.1%}")
    with c4:
        st.metric("P(L|L)", f"{ac.p_loss_after_loss:.1%}")

    st.markdown("")


def _render_feature_drift(report: DiagnosticsReport) -> None:
    """Render feature drift section."""
    if not report.feature_drift:
        return

    st.markdown("##### Feature Drift")

    rows = []
    for d in report.feature_drift:
        status = "Aligned" if d.is_aligned else (
            f"{len(d.missing_in_runtime)} missing" if d.missing_in_runtime else
            f"{len(d.extra_in_runtime)} extra"
        )
        severity_color = {
            "none": "#3fb950",
            "warning": "#d29922",
            "error": "#f85149",
        }[d.drift_severity]

        rows.append({
            "Model": d.model_name,
            "Training": d.training_count,
            "Runtime": d.runtime_count,
            "Status": status,
            "Severity": d.drift_severity,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Show missing features for error-level drift
    for d in report.feature_drift:
        if d.missing_in_runtime:
            with st.expander(f"{d.model_name}: {len(d.missing_in_runtime)} missing features"):
                st.code(", ".join(d.missing_in_runtime[:20]))
                if len(d.missing_in_runtime) > 20:
                    st.caption(f"... and {len(d.missing_in_runtime) - 20} more")
