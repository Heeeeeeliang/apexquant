"""
Segmented health bar component for Streamlit.

Renders the result of :func:`analytics.health.run_preflight` as a
horizontal bar with one segment per check. Each segment is coloured
by its status (green / yellow / red) and is expandable to show details.

Usage::

    from analytics.health import run_preflight
    from frontend.components.health_bar import render_health_bar

    report = run_preflight(config)
    render_health_bar(report)
"""

__all__ = ["render_health_bar"]

import streamlit as st

from analytics.health import CheckStatus, HealthReport


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

_COLORS = {
    CheckStatus.GREEN: {"bg": "#238636", "fg": "#3fb950", "glow": "#3fb95060"},
    CheckStatus.YELLOW: {"bg": "#9e6a03", "fg": "#d29922", "glow": "#d2992260"},
    CheckStatus.RED: {"bg": "#da3633", "fg": "#f85149", "glow": "#f8514960"},
}

_STATUS_ICON = {
    CheckStatus.GREEN: "&#10003;",   # checkmark
    CheckStatus.YELLOW: "&#9888;",   # warning triangle
    CheckStatus.RED: "&#10007;",     # cross
}

_OVERALL_LABEL = {
    CheckStatus.GREEN: "All Systems Go",
    CheckStatus.YELLOW: "Runnable with Warnings",
    CheckStatus.RED: "Blocked — Fix Required",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_health_bar(report: HealthReport) -> None:
    """Render a segmented health bar from a preflight report.

    Each segment is a clickable expander showing check details.
    The overall status is shown as a header bar above the segments.

    Args:
        report: Result of :func:`analytics.health.run_preflight`.
    """
    overall = report.overall
    c = _COLORS[overall]

    # Overall status banner
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {c['bg']}40 0%, {c['bg']}20 100%);
            border: 1px solid {c['bg']};
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        ">
            <span style="
                display: inline-block;
                width: 12px; height: 12px;
                border-radius: 50%;
                background: {c['fg']};
                box-shadow: 0 0 8px {c['glow']};
            "></span>
            <span style="
                font-family: 'Outfit', sans-serif;
                font-size: 14px;
                font-weight: 600;
                color: {c['fg']};
                letter-spacing: 0.03em;
            ">PREFLIGHT: {_OVERALL_LABEL[overall]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Segmented bar (visual only)
    if report.segments:
        n = len(report.segments)
        bar_html = '<div style="display:flex; gap:4px; margin-bottom:12px;">'
        for seg in report.segments:
            sc = _COLORS[seg.status]
            pct = 100 / n
            bar_html += (
                f'<div style="'
                f"flex: 1;"
                f"height: 8px;"
                f"border-radius: 4px;"
                f"background: {sc['fg']};"
                f"box-shadow: 0 0 4px {sc['glow']};"
                f'"></div>'
            )
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

    # Segment details as expanders
    cols = st.columns(len(report.segments)) if report.segments else []
    for col, seg in zip(cols, report.segments):
        sc = _COLORS[seg.status]
        icon = _STATUS_ICON[seg.status]
        with col:
            with st.expander(f"{seg.name}", expanded=(seg.status != CheckStatus.GREEN)):
                st.markdown(
                    f'<span style="color:{sc["fg"]}; font-size:16px; font-weight:600;">'
                    f"{icon} {seg.status.upper()}</span>",
                    unsafe_allow_html=True,
                )
                for detail in seg.details:
                    st.caption(detail)
