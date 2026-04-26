"""
Preset picker component for Streamlit.

Renders a selectbox of built-in + user presets with description preview,
apply button, and save-as-preset dialog.

Usage::

    from frontend.components.preset_library import render_preset_picker

    render_preset_picker(cfg, on_apply_callback)
"""

__all__ = ["render_preset_picker"]

from copy import deepcopy
from typing import Any

import streamlit as st

from config.presets import PRESETS, apply_preset, list_presets
from config.preset_io import (
    delete_user_preset,
    list_user_presets,
    save_user_preset,
)


def render_preset_picker(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Render a preset picker with apply and save-as controls.

    Args:
        cfg: Current config dict (read-only — not mutated).

    Returns:
        New config dict if a preset was applied, else None.
    """
    # Build combined preset list
    builtin = list_presets()
    user = list_user_presets()

    all_presets = builtin + [
        {**u, "name": f"[User] {u['name']}", "description": "User-defined preset"}
        for u in user
    ]

    options = [p["id"] for p in all_presets]
    labels = [f"{p['name']}" for p in all_presets]

    if not options:
        st.info("No presets available.")
        return None

    col_select, col_apply = st.columns([4, 1])

    with col_select:
        selected_idx = st.selectbox(
            "Preset",
            range(len(options)),
            format_func=lambda i: labels[i],
            key="preset_picker_select",
        )

    selected_id = options[selected_idx]
    selected_meta = all_presets[selected_idx]

    # Description
    if selected_meta.get("description"):
        st.caption(selected_meta["description"])

    new_cfg = None

    with col_apply:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Apply", key="preset_apply", use_container_width=True):
            if selected_id in PRESETS:
                new_cfg = apply_preset(cfg, selected_id)
            else:
                # User preset — load and replace
                from config.preset_io import load_user_preset
                new_cfg = load_user_preset(selected_id)

    # Save current as user preset
    st.markdown("")
    save_col1, save_col2, del_col = st.columns([3, 1, 1])

    with save_col1:
        save_name = st.text_input(
            "Save current config as preset",
            placeholder="my_experiment",
            key="preset_save_name",
            label_visibility="collapsed",
        )

    with save_col2:
        if st.button("Save", key="preset_save_btn", use_container_width=True):
            name = save_name.strip()
            if not name:
                st.error("Enter a preset name.")
            else:
                try:
                    save_user_preset(name, cfg)
                    st.success(f"Saved preset '{name}'")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with del_col:
        if selected_id not in PRESETS:
            if st.button("Delete", key="preset_del_btn", use_container_width=True):
                delete_user_preset(selected_id)
                st.success(f"Deleted preset '{selected_id}'")
                st.rerun()

    return new_cfg
