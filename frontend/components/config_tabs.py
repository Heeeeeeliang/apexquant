"""
Reusable config tab rendering components for Streamlit.

Extracts the tab-body logic from the Setup page into standalone
functions so they can be composed and tested independently.

Usage::

    from frontend.components.config_tabs import render_guard_banner

    violations = validate_config(cfg)
    render_guard_banner(violations)
"""

__all__ = ["render_guard_banner", "render_active_preset_badge"]

from typing import Any

import streamlit as st

from config.schema import GuardViolation


def render_guard_banner(violations: list[GuardViolation]) -> None:
    """Show guard violations as a banner above config controls.

    Errors are shown as red callouts, warnings as yellow.
    If no violations, shows a green success indicator.

    Args:
        violations: Output of :func:`config.schema.validate_config`.
    """
    errors = [v for v in violations if v.level == "error"]
    warnings = [v for v in violations if v.level == "warning"]

    if not violations:
        st.markdown(
            '<div style="padding:8px 12px; background:#238636; border-radius:6px; '
            'color:#e6edf3; font-size:13px; margin-bottom:12px;">'
            '&#10003; Configuration valid</div>',
            unsafe_allow_html=True,
        )
        return

    if errors:
        error_html = "".join(
            f'<div style="color:#f85149; font-size:13px; padding:2px 0;">'
            f'&#10007; <code>{v.field}</code>: {v.message}</div>'
            for v in errors
        )
        st.markdown(
            f'<div style="padding:10px 14px; background:#da363320; '
            f'border:1px solid #da3633; border-radius:6px; margin-bottom:8px;">'
            f'<div style="color:#f85149; font-weight:600; margin-bottom:4px;">'
            f'Config Errors ({len(errors)})</div>{error_html}</div>',
            unsafe_allow_html=True,
        )

    if warnings:
        warn_html = "".join(
            f'<div style="color:#d29922; font-size:13px; padding:2px 0;">'
            f'&#9888; <code>{v.field}</code>: {v.message}</div>'
            for v in warnings
        )
        st.markdown(
            f'<div style="padding:10px 14px; background:#9e6a0320; '
            f'border:1px solid #9e6a03; border-radius:6px; margin-bottom:8px;">'
            f'<div style="color:#d29922; font-weight:600; margin-bottom:4px;">'
            f'Config Warnings ({len(warnings)})</div>{warn_html}</div>',
            unsafe_allow_html=True,
        )


def render_active_preset_badge(preset_name: str | None = None) -> None:
    """Show the currently active preset as a small badge.

    Args:
        preset_name: Name of the active preset. None = "Custom".
    """
    label = preset_name or "Custom"
    color = "#3fb950" if preset_name else "#8b949e"

    st.markdown(
        f'<span style="display:inline-block; padding:3px 10px; '
        f'border-radius:12px; border:1px solid {color}; color:{color}; '
        f'font-size:11px; font-weight:600; letter-spacing:0.04em; '
        f'font-family:\'Outfit\',sans-serif;">'
        f'{label}</span>',
        unsafe_allow_html=True,
    )
