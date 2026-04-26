"""
Configuration page -- view and edit system parameters.

Organised into tabs: Strategy, Models, Backtest, Compute, Data, Google Drive.
Supports named presets (built-in + user-defined) and guard validation.
Changes are stored in session_state and can be saved to disk.
"""

__all__: list[str] = []

import json
import shutil
from copy import deepcopy
from pathlib import Path

import sys, os
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from components.styles import inject_global_css

logger.info("Config page loaded")

# ---------------------------------------------------------------------------
# Ensure config in session state
# ---------------------------------------------------------------------------

if "config" not in st.session_state:
    from config.default import CONFIG

    st.session_state["config"] = deepcopy(CONFIG)

cfg = st.session_state["config"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

inject_global_css()
st.markdown("""
<div style="padding: 0 0 20px 0; border-bottom: 1px solid #21262d; margin-bottom: 24px;">
  <div style="font-family:'Outfit',sans-serif; font-size:11px; font-weight:600;
              letter-spacing:0.1em; text-transform:uppercase; color:#8b949e; margin-bottom:4px;">
    ① SETUP
  </div>
  <h1 style="margin:0; font-family:'Outfit',sans-serif;">Configuration</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Preset picker
# ---------------------------------------------------------------------------

from frontend.components.preset_library import render_preset_picker

with st.expander("Presets", expanded=False):
    new_cfg = render_preset_picker(cfg)
    if new_cfg is not None:
        st.session_state["config"] = new_cfg
        st.session_state["active_preset"] = st.session_state.get(
            "preset_picker_select", None
        )
        st.rerun()

# ---------------------------------------------------------------------------
# Guard validation banner
# ---------------------------------------------------------------------------

from config.schema import validate_config
from frontend.components.config_tabs import render_guard_banner, render_active_preset_badge

_violations = validate_config(cfg)
render_guard_banner(_violations)

# Show active preset badge if set
_active_preset = st.session_state.get("active_preset")
if _active_preset is not None:
    from config.presets import list_presets
    _preset_map = {p["id"]: p["name"] for p in list_presets()}
    render_active_preset_badge(_preset_map.get(str(_active_preset), None))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_strategy, tab_models, tab_backtest, tab_compute, tab_data, tab_drive = st.tabs(
    ["Strategy", "Models", "Backtest", "Compute", "Data", "Google Drive"]
)

# ======================== Strategy Tab ========================
with tab_strategy:
    st.subheader("Strategy Settings")

    strat_cfg = cfg.get("strategy", {})

    use_ai = st.toggle(
        "Use AI Signals",
        value=strat_cfg.get("use_ai", True),
        key="cfg_use_ai",
    )

    pred_cfg = cfg.get("predictors", {})
    enabled = pred_cfg.get("enabled", [])

    from predictors.registry import REGISTRY as _STRAT_REGISTRY

    _all_predictor_names = _STRAT_REGISTRY.list_all()
    if _all_predictor_names:
        st.markdown("**Enabled Predictors**")
        new_enabled = []
        for pname in _all_predictor_names:
            toggled = st.toggle(
                pname,
                value=pname in enabled,
                key=f"cfg_pred_{pname}",
            )
            if toggled:
                new_enabled.append(pname)
    else:
        st.info("No predictors registered.")
        new_enabled = []

    signal_mode = st.selectbox(
        "Signal Mode",
        options=["ai", "technical", "hybrid"],
        index=["ai", "technical", "hybrid"].index(
            strat_cfg.get("signal_mode", "ai")
        ),
        key="cfg_signal_mode",
    )

    # Write back
    cfg.setdefault("strategy", {})["use_ai"] = use_ai
    cfg["strategy"]["signal_mode"] = signal_mode
    cfg.setdefault("predictors", {})["enabled"] = new_enabled

# ======================== Models Tab ========================
with tab_models:
    st.subheader("Models")

    from predictors.registry import REGISTRY as _MODEL_REGISTRY

    _adapter_names = _MODEL_REGISTRY.list_all()

    if not _adapter_names:
        st.warning("No models loaded — sync from Google Drive first.")
    else:
        st.caption(
            f"{len(_adapter_names)} registered model(s). "
            "Thresholds are configured in the **Pipeline Editor** page."
        )
        for _aname in _adapter_names:
            st.markdown(f"- `{_aname}`")

    # --- Trade execution parameters (not model-specific) ---
    st.markdown("---")
    st.subheader("Trade Execution")

    model_cfg = cfg.get("model", {})

    col_tp, col_sl, col_mb = st.columns(3)
    with col_tp:
        meta_tp = st.number_input(
            "Take Profit (%)",
            min_value=0.1,
            max_value=5.0,
            value=float(model_cfg.get("meta_tp", 0.005)) * 100,
            step=0.1,
            format="%.1f",
            key="cfg_meta_tp",
        )
    with col_sl:
        meta_sl = st.number_input(
            "Stop Loss (%)",
            min_value=0.1,
            max_value=5.0,
            value=float(model_cfg.get("meta_sl", 0.003)) * 100,
            step=0.1,
            format="%.1f",
            key="cfg_meta_sl",
        )
    with col_mb:
        meta_mb = st.number_input(
            "Max Bars Held",
            min_value=1,
            max_value=200,
            value=int(model_cfg.get("meta_mb", 48)),
            step=1,
            key="cfg_meta_mb",
        )

    cfg.setdefault("model", {})["meta_tp"] = meta_tp / 100.0
    cfg["model"]["meta_sl"] = meta_sl / 100.0
    cfg["model"]["meta_mb"] = meta_mb

# ======================== Backtest Tab ========================
with tab_backtest:
    st.subheader("Backtest Parameters")

    bt_cfg = cfg.get("backtest", {})

    col1, col2 = st.columns(2)

    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1_000,
            max_value=10_000_000,
            value=int(bt_cfg.get("initial_capital", 100_000)),
            step=10_000,
            key="cfg_capital",
        )

        commission = st.slider(
            "Commission",
            min_value=0.0,
            max_value=0.005,
            value=float(bt_cfg.get("commission", 0.001)),
            step=0.0001,
            format="%.4f",
            key="cfg_commission",
        )

    with col2:
        slippage = st.slider(
            "Slippage",
            min_value=0.0,
            max_value=0.002,
            value=float(bt_cfg.get("slippage", 0.0005)),
            step=0.0001,
            format="%.4f",
            key="cfg_slippage",
        )

        position_size = st.slider(
            "Position Size",
            min_value=0.01,
            max_value=0.50,
            value=float(bt_cfg.get("position_size", 0.10)),
            step=0.01,
            format="%.2f",
            key="cfg_pos_size",
        )

    # Write back
    cfg.setdefault("backtest", {})["initial_capital"] = initial_capital
    cfg["backtest"]["commission"] = commission
    cfg["backtest"]["slippage"] = slippage
    cfg["backtest"]["position_size"] = position_size

    st.markdown("---")
    if st.button("🗑 Clear Backtest Cache", key="cfg_clear_bt_cache"):
        from backtest.runner import clear_backtest_cache
        _n_removed = clear_backtest_cache()
        st.success(f"Removed {_n_removed} cached backtest result(s)")

# ======================== Compute Tab ========================
with tab_compute:
    st.subheader("Compute Backend")

    compute_cfg = cfg.get("compute", {})

    backend = st.selectbox(
        "Backend",
        options=["local", "colab", "gcloud", "aws"],
        index=["local", "colab", "gcloud", "aws"].index(
            compute_cfg.get("backend", "local")
        ),
        key="cfg_backend",
    )
    cfg.setdefault("compute", {})["backend"] = backend

    # Conditional inputs
    if backend == "colab":
        colab_cfg = compute_cfg.get("colab", {})
        drive_path = st.text_input(
            "Google Drive Path",
            value=colab_cfg.get("drive_path", "/gdrive/MyDrive/apexquant/"),
            key="cfg_drive_path",
        )
        cfg["compute"].setdefault("colab", {})["drive_path"] = drive_path

    elif backend == "gcloud":
        gc_cfg = compute_cfg.get("gcloud", {})
        gc_project = st.text_input(
            "GCP Project",
            value=gc_cfg.get("project", ""),
            key="cfg_gc_project",
        )
        gc_region = st.text_input(
            "Region",
            value=gc_cfg.get("region", "us-central1"),
            key="cfg_gc_region",
        )
        cfg["compute"].setdefault("gcloud", {})["project"] = gc_project
        cfg["compute"]["gcloud"]["region"] = gc_region

    elif backend == "aws":
        aws_cfg = compute_cfg.get("aws", {})
        aws_instance = st.text_input(
            "Instance Type",
            value=aws_cfg.get("instance_type", "ml.p3.2xlarge"),
            key="cfg_aws_instance",
        )
        aws_role = st.text_input(
            "Role ARN",
            value=aws_cfg.get("role_arn", ""),
            key="cfg_aws_role",
        )
        cfg["compute"].setdefault("aws", {})["instance_type"] = aws_instance
        cfg["compute"]["aws"]["role_arn"] = aws_role

    # Test connection button
    if st.button("🔌 Test Connection", key="cfg_test_conn"):
        with st.spinner("Testing..."):
            try:
                from compute import get_backend

                be = get_backend(cfg)
                result = be.test_connection()
                if result.get("ok"):
                    st.success(
                        f"Connected ({result.get('latency_ms', '?')}ms) -- "
                        f"{result.get('message', '')}"
                    )
                else:
                    st.error(f"Connection failed: {result.get('message', '')}")
            except Exception as exc:
                st.error(f"Error: {exc}")

# ======================== Data Tab ========================
with tab_data:
    st.subheader("Data Settings")

    data_cfg = cfg.get("data", {})

    all_tickers = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "TSLA",
        "AMZN", "META", "SPY", "QQQ", "IWM", "DIA",
    ]
    current_tickers = data_cfg.get("tickers", ["AAPL"])

    tickers = st.multiselect(
        "Tickers",
        options=all_tickers,
        default=[t for t in current_tickers if t in all_tickers],
        key="cfg_tickers",
    )

    source = st.selectbox(
        "Data Source",
        options=["csv", "yfinance", "databento_api"],
        index=["csv", "yfinance", "databento_api"].index(
            data_cfg.get("source", "csv")
        ),
        key="cfg_source",
    )

    # Write back
    cfg.setdefault("data", {})["tickers"] = tickers
    cfg["data"]["source"] = source

# ======================== Google Drive Tab ========================
with tab_drive:
    st.subheader("Google Drive Integration")
    st.markdown(
        "Sync feature CSVs and trained model files from Google Drive. "
        "Add one row per Drive folder you want to sync."
    )

    import uuid as _uuid

    import pandas as pd
    from pathlib import Path as _Path

    drive_cfg = cfg.setdefault("drive", {})

    # ----------------------------------------------------------------
    # Helper: render an editable (local_path, drive_url) row list
    # ----------------------------------------------------------------

    def _ensure_rows(ss_key: str, config_map: dict[str, str]) -> list[dict]:
        """Initialise session-state row list from config if missing."""
        if ss_key not in st.session_state:
            if config_map:
                st.session_state[ss_key] = [
                    {"id": _uuid.uuid4().hex, "local": k, "url": v}
                    for k, v in config_map.items()
                ]
            else:
                st.session_state[ss_key] = []
        return st.session_state[ss_key]

    def _collect_mapping(ss_key: str) -> dict[str, str]:
        """Read current widget values and return {local: url} mapping."""
        rows = st.session_state.get(ss_key, [])
        mapping: dict[str, str] = {}
        for row in rows:
            rid = row["id"]
            local = st.session_state.get(f"local_{rid}", "").strip()
            url = st.session_state.get(f"url_{rid}", "").strip()
            if local and url:
                mapping[local] = url
        return mapping

    def _render_rows(
        ss_key: str,
        label_local: str,
        placeholder_local: str,
        add_label: str,
    ) -> None:
        """Render the dynamic row list for a sync section."""
        rows: list[dict] = st.session_state[ss_key]

        # Column headers
        if rows:
            h_local, h_url, h_del = st.columns([3, 6, 1])
            with h_local:
                st.caption(label_local)
            with h_url:
                st.caption("Drive URL")
            with h_del:
                st.caption("")

        # Rows — widget state is driven entirely by session_state keys.
        # We initialise the key once (from row data) then never pass
        # ``value=`` again, avoiding the Streamlit value/key conflict
        # that resets user input on rerun.
        delete_id: str | None = None
        for row in rows:
            rid = row["id"]

            # Seed widget keys from row data on first encounter only
            if f"local_{rid}" not in st.session_state:
                st.session_state[f"local_{rid}"] = row.get("local", "")
            if f"url_{rid}" not in st.session_state:
                st.session_state[f"url_{rid}"] = row.get("url", "")

            c_local, c_url, c_del = st.columns([3, 6, 1])
            with c_local:
                st.text_input(
                    label_local,
                    placeholder=placeholder_local,
                    key=f"local_{rid}",
                    label_visibility="collapsed",
                )
            with c_url:
                st.text_input(
                    "Drive URL",
                    placeholder="https://drive.google.com/drive/folders/...",
                    key=f"url_{rid}",
                    label_visibility="collapsed",
                )
            with c_del:
                if st.button("✕", key=f"del_{rid}"):
                    delete_id = rid

        # Handle deletion by id (not index)
        if delete_id is not None:
            removed = [
                r for r in st.session_state[ss_key]
                if r["id"] == delete_id
            ]
            # Clean up widget keys for deleted row
            for r in removed:
                st.session_state.pop(f"local_{r['id']}", None)
                st.session_state.pop(f"url_{r['id']}", None)
            st.session_state[ss_key] = [
                r for r in st.session_state[ss_key] if r["id"] != delete_id
            ]
            st.rerun()

        # Add button
        if st.button(add_label, key=f"{ss_key}_add"):
            st.session_state[ss_key].append(
                {"id": _uuid.uuid4().hex, "local": "", "url": ""}
            )
            st.rerun()

    def _show_sync_results(
        results: dict[str, dict],
        local_root: _Path,
    ) -> int:
        """Display per-row sync status. Returns count of successes."""
        ok = 0
        for local_path, res in results.items():
            status = res.get("status", "")
            if status == "ok":
                st.markdown(
                    f"<span style='color:#3fb950'>&#10003;</span> "
                    f"`{local_path}` — synced to `{local_root / local_path}`",
                    unsafe_allow_html=True,
                )
                if res.get("meta_generated"):
                    st.warning(
                        f"`{local_path}/meta.json` was auto-generated. "
                        "Review and edit `output_label` and `name` fields "
                        "to match your model."
                    )
                ok += 1
            else:
                msg = res.get("message", "unknown error")
                st.markdown(
                    f"<span style='color:#f85149'>&#10007;</span> "
                    f"`{local_path}` — {msg}",
                    unsafe_allow_html=True,
                )
        return ok

    # ================================================================
    # Models section — smart auto-detect
    # ================================================================

    st.markdown("##### Models")
    st.caption(
        "Paste a Google Drive URL for each weight file. The system "
        "auto-detects the model type (`.pt` → CNN, `.joblib` → LightGBM) "
        "and layer placement from the filename."
    )

    # Display stored sync results from previous rerun
    _msync = st.session_state.pop("_model_sync_result", None)
    if _msync is not None:
        if _msync.get("table"):
            st.dataframe(
                pd.DataFrame(_msync["table"]),
                use_container_width=True,
                hide_index=True,
            )
        if _msync.get("level") == "success":
            st.success(_msync["msg"])
        elif _msync.get("level") == "warning":
            st.warning(_msync["msg"])
        elif _msync.get("level") == "error":
            st.error(_msync["msg"])
        if _msync.get("reload_msg"):
            st.success(_msync["reload_msg"])
        if _msync.get("meta_hint"):
            st.info(_msync["meta_hint"])

    # Initialise URL rows in session_state
    if "_drive_model_urls" not in st.session_state:
        saved = drive_cfg.get("model_urls", [])
        st.session_state["_drive_model_urls"] = [
            {"id": _uuid.uuid4().hex, "url": u} for u in saved
        ] if saved else []

    _model_url_rows: list[dict] = st.session_state["_drive_model_urls"]

    # Render URL rows
    delete_model_id: str | None = None
    for row in _model_url_rows:
        rid = row["id"]
        if f"murl_{rid}" not in st.session_state:
            st.session_state[f"murl_{rid}"] = row.get("url", "")
        c_url, c_del = st.columns([9, 1])
        with c_url:
            st.text_input(
                "Drive URL",
                placeholder="https://drive.google.com/file/d/... or /drive/folders/...",
                key=f"murl_{rid}",
                label_visibility="collapsed",
            )
        with c_del:
            if st.button("✕", key=f"mdel_{rid}"):
                delete_model_id = rid

    if delete_model_id is not None:
        for r in [r for r in _model_url_rows if r["id"] == delete_model_id]:
            st.session_state.pop(f"murl_{r['id']}", None)
        st.session_state["_drive_model_urls"] = [
            r for r in _model_url_rows if r["id"] != delete_model_id
        ]
        st.rerun()

    if st.button("+ Add URL", key="_drive_model_urls_add"):
        st.session_state["_drive_model_urls"].append(
            {"id": _uuid.uuid4().hex, "url": ""}
        )
        st.rerun()

    if st.button("Sync All Models", use_container_width=True, key="cfg_drive_sync_models"):
        # Read current widget values back into row dicts
        for row in st.session_state["_drive_model_urls"]:
            row["url"] = st.session_state.get(f"murl_{row['id']}", "")

        urls = [
            row["url"].strip()
            for row in st.session_state["_drive_model_urls"]
            if row["url"].strip()
        ]
        # Persist for next load
        drive_cfg["model_urls"] = urls

        if not urls:
            st.warning("Add at least one Drive URL.")
        else:
            from services.drive_sync import DriveSync

            ds = DriveSync(
                data_dir=drive_cfg.get("cache_dir", "data/features"),
                models_dir=drive_cfg.get("models_dir", "models"),
            )
            with st.spinner(f"Syncing {len(urls)} URL(s)..."):
                all_results = ds.sync_smart_models(urls)

            # Build results table
            table_rows: list[dict[str, str]] = []
            ok_count = 0
            for url, res in zip(urls, all_results):
                short_url = url[:60] + "..." if len(url) > 60 else url
                if res.get("status") == "ok":
                    for m in res.get("models", []):
                        if m.get("status") == "error":
                            table_rows.append({
                                "URL": short_url,
                                "File": m.get("original_filename", "?"),
                                "Placed In": "—",
                                "Type": "—",
                                "Label": "—",
                                "Status": f"✗ {m.get('message', 'failed')}",
                            })
                        else:
                            table_rows.append({
                                "URL": short_url,
                                "File": m["original_filename"],
                                "Placed In": f"models/{m['inferred_path']}/",
                                "Type": m["adapter_type"],
                                "Label": m["output_label"],
                                "Status": "✓ synced" + (" (meta auto-generated)" if m.get("meta_generated") else ""),
                            })
                            ok_count += 1
                else:
                    debug = res.get("debug_files", [])
                    debug_str = ""
                    if debug:
                        debug_str = " | Files: " + ", ".join(
                            f"{d['name']} ({d['size']}B hdr={d['header'][:8]})"
                            for d in debug[:3]
                        )
                    table_rows.append({
                        "URL": short_url,
                        "File": "—",
                        "Placed In": "—",
                        "Type": "—",
                        "Label": "—",
                        "Status": f"✗ {res.get('message', 'failed')}{debug_str}",
                    })

            # Hot-reload registry so new models are available immediately
            reload_msg = ""
            if ok_count > 0:
                from predictors.registry import REGISTRY as _RELOAD_REG

                models_root = drive_cfg.get("models_dir", "models")
                n_new = _RELOAD_REG.reload(models_dir=models_root)
                if n_new:
                    reload_msg = (
                        f"Hot-reloaded {n_new} new adapter(s) into the "
                        "predictor registry (no restart needed)."
                    )

            # Build status message
            if ok_count == len(urls):
                level, msg = "success", f"All {ok_count} model(s) synced and placed automatically."
            elif ok_count > 0:
                level, msg = "warning", f"{ok_count}/{len(urls)} synced successfully."
            else:
                level, msg = "error", "No models were synced."

            # Hint about auto-generated meta.json
            any_auto = any(
                m.get("meta_generated")
                for r in all_results if r.get("status") == "ok"
                for m in r.get("models", [])
            )
            meta_hint = (
                "Some `meta.json` files were auto-generated. "
                "Review them under `models/` if the inferred "
                "`output_label` or `adapter` type needs adjustment."
            ) if any_auto else ""

            # Store results in session state and rerun so the page
            # (and cache overview) re-renders with updated registry
            st.session_state["_model_sync_result"] = {
                "table": table_rows,
                "level": level,
                "msg": msg,
                "reload_msg": reload_msg,
                "meta_hint": meta_hint,
            }
            st.rerun()
    else:
        # Persist URLs on every page load
        drive_cfg["model_urls"] = [
            st.session_state.get(f"murl_{row['id']}", "").strip()
            for row in st.session_state["_drive_model_urls"]
            if st.session_state.get(f"murl_{row['id']}", "").strip()
        ]

    # ================================================================
    # Data section — URL-only (same pattern as Models)
    # ================================================================

    st.markdown("---")
    st.markdown("##### Data Sources")
    st.caption(
        "Paste a Google Drive URL for each data file or folder. "
        "Files are downloaded into `data/features/`."
    )

    # Display stored data sync results from previous rerun
    _dsync = st.session_state.pop("_data_sync_result", None)
    if _dsync is not None:
        if _dsync.get("table"):
            st.dataframe(
                pd.DataFrame(_dsync["table"]),
                use_container_width=True,
                hide_index=True,
            )
        if _dsync.get("level") == "success":
            st.success(_dsync["msg"])
        elif _dsync.get("level") == "warning":
            st.warning(_dsync["msg"])
        elif _dsync.get("level") == "error":
            st.error(_dsync["msg"])

    if "_drive_data_urls" not in st.session_state:
        saved_data = drive_cfg.get("data_urls", [])
        st.session_state["_drive_data_urls"] = [
            {"id": _uuid.uuid4().hex, "url": u} for u in saved_data
        ] if saved_data else []

    _data_url_rows: list[dict] = st.session_state["_drive_data_urls"]

    delete_data_id: str | None = None
    for row in _data_url_rows:
        rid = row["id"]
        if f"durl_{rid}" not in st.session_state:
            st.session_state[f"durl_{rid}"] = row.get("url", "")
        c_url, c_del = st.columns([9, 1])
        with c_url:
            st.text_input(
                "Drive URL",
                placeholder="https://drive.google.com/file/d/... or /drive/folders/...",
                key=f"durl_{rid}",
                label_visibility="collapsed",
            )
        with c_del:
            if st.button("✕", key=f"ddel_{rid}"):
                delete_data_id = rid

    if delete_data_id is not None:
        for r in [r for r in _data_url_rows if r["id"] == delete_data_id]:
            st.session_state.pop(f"durl_{r['id']}", None)
        st.session_state["_drive_data_urls"] = [
            r for r in _data_url_rows if r["id"] != delete_data_id
        ]
        st.rerun()

    if st.button("+ Add URL", key="_drive_data_urls_add"):
        st.session_state["_drive_data_urls"].append(
            {"id": _uuid.uuid4().hex, "url": ""}
        )
        st.rerun()

    if st.button("Sync All Data", use_container_width=True, key="cfg_drive_sync_data"):
        for row in st.session_state["_drive_data_urls"]:
            row["url"] = st.session_state.get(f"durl_{row['id']}", "")

        data_urls = [
            row["url"].strip()
            for row in st.session_state["_drive_data_urls"]
            if row["url"].strip()
        ]
        drive_cfg["data_urls"] = data_urls

        if not data_urls:
            st.warning("Add at least one Drive URL.")
        else:
            from services.drive_sync import DriveSync

            ds = DriveSync(
                data_dir=drive_cfg.get("cache_dir", "data/features"),
                models_dir=drive_cfg.get("models_dir", "models"),
            )
            with st.spinner(f"Syncing {len(data_urls)} data URL(s)..."):
                data_results = ds.sync_smart_data(data_urls)

            d_table: list[dict[str, str]] = []
            d_ok = 0
            for url, res in zip(data_urls, data_results):
                short = url[:60] + "..." if len(url) > 60 else url
                if res.get("status") == "ok":
                    n_files = res.get("files", "?")
                    fname = res.get("filename", "")
                    detail = f"{n_files} file(s)" + (f" ({fname})" if fname else "")
                    d_table.append({"URL": short, "Status": f"✓ {detail}"})
                    d_ok += 1
                else:
                    d_table.append({"URL": short, "Status": f"✗ {res.get('message', 'failed')}"})

            if d_ok == len(data_urls):
                level, msg = "success", f"All {d_ok} data source(s) synced."
            elif d_ok > 0:
                level, msg = "warning", f"{d_ok}/{len(data_urls)} synced."
            else:
                level, msg = "error", "No data sources were synced."

            st.session_state["_data_sync_result"] = {
                "table": d_table,
                "level": level,
                "msg": msg,
            }
            st.rerun()
    else:
        drive_cfg["data_urls"] = [
            st.session_state.get(f"durl_{row['id']}", "").strip()
            for row in st.session_state["_drive_data_urls"]
            if st.session_state.get(f"durl_{row['id']}", "").strip()
        ]

    # ================================================================
    # Cache overview + clear
    # ================================================================

    st.markdown("---")
    st.markdown("##### Cached Files")

    from services.drive_sync import DriveSync as _DS

    _cache = _DS(
        data_dir=drive_cfg.get("cache_dir", "data/features"),
        models_dir=drive_cfg.get("models_dir", "models"),
    ).get_cache_status()

    # ----------------------------------------------------------
    # Data Files with per-file delete
    # ----------------------------------------------------------
    if _cache["data_files"]:
        st.markdown("**Data Files**")
        _data_dir = _Path(_cache["data_dir"])
        for _df_entry in _cache["data_files"]:
            _df_name = _df_entry["name"]
            _df_cols = st.columns([5, 2, 2, 1])
            with _df_cols[0]:
                st.text(_df_name)
            with _df_cols[1]:
                st.text(f"{_df_entry['size_kb']} KB")
            with _df_cols[2]:
                st.text(_df_entry["modified"])
            with _df_cols[3]:
                if st.button("🗑", key=f"del_data_{_df_name}"):
                    (_data_dir / _df_name).unlink(missing_ok=True)
                    st.rerun()
    else:
        st.info(f"No data files cached in `{_cache['data_dir']}`")

    # ----------------------------------------------------------
    # Model Files with per-model-folder delete (with confirm)
    # ----------------------------------------------------------

    # Collect unique model folders (parent of each model file)
    def _model_folders(
        model_files: list[dict], models_root: _Path
    ) -> dict[str, dict]:
        """Group model files by their top-level model directory."""
        folders: dict[str, dict] = {}
        for mf in model_files:
            rel = _Path(mf["name"])
            # Model folder is the directory containing the file.
            # For paths like layer1/vol/lgb_v1/weights.joblib the model
            # folder is the immediate parent of the file.
            model_dir = rel.parent
            key = str(model_dir)
            if key not in folders:
                folders[key] = {
                    "path": key,
                    "abs": models_root / model_dir,
                    "files": [],
                    "total_kb": 0.0,
                }
            folders[key]["files"].append(mf["name"])
            folders[key]["total_kb"] += mf["size_kb"]
        return folders

    if _cache["model_files"]:
        st.markdown("**Model Files**")
        _models_root = _Path(_cache["models_dir"])
        _mfolders = _model_folders(_cache["model_files"], _models_root)

        for _mkey, _minfo in _mfolders.items():
            _confirm_key = f"confirm_delete_model_{_mkey}"
            _mcols = st.columns([5, 2, 1])
            with _mcols[0]:
                st.text(
                    f"{_mkey}/ ({len(_minfo['files'])} file(s))"
                )
            with _mcols[1]:
                st.text(f"{_minfo['total_kb']:.1f} KB")
            with _mcols[2]:
                if st.button("🗑 Delete", key=f"del_model_{_mkey}"):
                    st.session_state[_confirm_key] = True

            # Confirmation row
            if st.session_state.get(_confirm_key, False):
                st.warning(
                    f"Delete **{_mkey}/** and all its contents? "
                    "This cannot be undone."
                )
                _cc1, _cc2, _ = st.columns([1, 1, 4])
                with _cc1:
                    if st.button(
                        "Yes, delete",
                        key=f"confirm_yes_{_mkey}",
                        type="primary",
                    ):
                        shutil.rmtree(_minfo["abs"], ignore_errors=True)
                        st.session_state.pop(_confirm_key, None)
                        from predictors.registry import REGISTRY as _DEL_REG
                        _del_models_root = drive_cfg.get(
                            "models_dir", "models"
                        )
                        # Clear + re-scan so deleted model disappears
                        _DEL_REG.reload(models_dir=_del_models_root)
                        # Purge stale cfg_pred_* toggle keys and
                        # remove deleted names from enabled list
                        _live = set(_DEL_REG.list_all())
                        _stale_keys = [
                            k for k in list(st.session_state)
                            if k.startswith("cfg_pred_")
                            and k[len("cfg_pred_"):] not in _live
                        ]
                        for _sk in _stale_keys:
                            st.session_state.pop(_sk, None)
                        _pred_cfg = cfg.setdefault("predictors", {})
                        _pred_cfg["enabled"] = [
                            n for n in _pred_cfg.get("enabled", [])
                            if n in _live
                        ]
                        st.rerun()
                with _cc2:
                    if st.button("Cancel", key=f"confirm_no_{_mkey}"):
                        st.session_state.pop(_confirm_key, None)
                        st.rerun()
    else:
        st.info(f"No model files cached in `{_cache['models_dir']}`")

    if st.button("Clear Cache", key="cfg_drive_clear"):
        from services.drive_sync import DriveSync as _DS2

        result = _DS2(
            data_dir=drive_cfg.get("cache_dir", "data/features"),
            models_dir=drive_cfg.get("models_dir", "models"),
        ).clear_cache()
        # Re-scan registry (models dir may now be empty)
        from predictors.registry import REGISTRY as _CLR_REG
        _CLR_REG.reload(
            models_dir=drive_cfg.get("models_dir", "models")
        )
        # Purge stale predictor toggle keys and enabled list
        _clr_live = set(_CLR_REG.list_all())
        for _ck in [
            k for k in list(st.session_state)
            if k.startswith("cfg_pred_")
            and k[len("cfg_pred_"):] not in _clr_live
        ]:
            st.session_state.pop(_ck, None)
        _clr_pred_cfg = cfg.setdefault("predictors", {})
        _clr_pred_cfg["enabled"] = [
            n for n in _clr_pred_cfg.get("enabled", [])
            if n in _clr_live
        ]
        st.success(f"Removed {result['count']} files")
        st.rerun()

# ---------------------------------------------------------------------------
# Save button
# ---------------------------------------------------------------------------

st.markdown("---")

col_save, col_reset, _ = st.columns([1, 1, 3])

with col_save:
    if st.button("💾 Save Configuration", use_container_width=True, key="cfg_save"):
        _save_violations = validate_config(cfg)
        _save_errors = [v for v in _save_violations if v.level == "error"]
        if _save_errors:
            st.error(
                f"Cannot save — {len(_save_errors)} guard error(s). "
                "Fix the issues shown above first."
            )
        else:
            save_path = Path("config/local_override.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, default=str)
            st.success(f"Configuration saved to `{save_path}`")
            logger.info("Configuration saved to {}", save_path)

with col_reset:
    if st.button("🔄 Reset to Defaults", use_container_width=True, key="cfg_reset"):
        from config.default import CONFIG

        st.session_state["config"] = deepcopy(CONFIG)
        st.success("Configuration reset to defaults")
        st.rerun()

# ---------------------------------------------------------------------------
# Raw config viewer
# ---------------------------------------------------------------------------

with st.expander("View Raw Config JSON"):
    st.json(json.loads(json.dumps(cfg, default=str)))
