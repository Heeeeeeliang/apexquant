"""
Visual Pipeline Editor — drag-and-drop strategy builder.

Users connect model nodes with threshold-gated edges to define a
trading strategy with zero code.  The pipeline is saved to
``config/pipeline.json`` and automatically used by the backtest engine.

State management
----------------
``st.session_state`` keys used (all prefixed ``_pipe_``):

- ``_pipe_nodes``     : list[StreamlitFlowNode]  — single source of truth for nodes
- ``_pipe_edges``     : list[StreamlitFlowEdge]  — single source of truth for edges
- ``_pipe_node_meta`` : dict[str, dict]           — per-node metadata (model_key, label)
- ``_pipe_edge_meta`` : dict[str, dict]           — per-edge metadata (condition, threshold)

The ``streamlit_flow`` component return is only used to *update* the
lists above when the user interacts on the canvas.  Button handlers
mutate the lists directly and call ``st.rerun()``.
"""

__all__: list[str] = []

import json
import re
import uuid
from copy import deepcopy
from pathlib import Path

import sys, os
import streamlit as st
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from components.styles import inject_global_css

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

logger.info("Pipeline editor page loaded")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_PATH = Path("config/pipeline.json")

_NODE_STYLES = {
    "START": {
        "background": "#ffffff",
        "border": "2px solid #8b949e",
        "color": "#000000",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "fontWeight": "bold",
    },
    "BUY": {
        "background": "#238636",
        "border": "2px solid #2ea043",
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "fontWeight": "bold",
    },
    "SHORT": {
        "background": "#da3633",
        "border": "2px solid #f85149",
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "fontWeight": "bold",
    },
    "CLOSE": {
        "background": "#d29922",
        "border": "2px solid #e3b341",
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "fontWeight": "bold",
    },
    "SELL": {
        "background": "#d29922",
        "border": "2px solid #e3b341",
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "fontWeight": "bold",
    },
    "SKIP": {
        "background": "#484f58",
        "border": "2px solid #6e7681",
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "fontWeight": "bold",
    },
    "MODEL": {
        "background": "#1f3a5f",
        "border": "2px solid #388bfd",
        "color": "#e6edf3",
        "borderRadius": "8px",
        "padding": "8px 16px",
    },
}

_EDGE_ABOVE_STYLE = {"stroke": "#3fb950", "strokeWidth": 2}
_EDGE_BELOW_STYLE = {"stroke": "#f85149", "strokeWidth": 2}
_EDGE_UNCONDITIONAL_STYLE = {"stroke": "#8b949e", "strokeWidth": 2}

# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def _get_registry_names() -> list[str]:
    try:
        from predictors.registry import REGISTRY
        return REGISTRY.list_all()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Node / Edge builders
# ---------------------------------------------------------------------------


def _style_for(model_key: str) -> dict:
    return deepcopy(_NODE_STYLES.get(model_key, _NODE_STYLES["MODEL"]))


def _node_type(model_key: str) -> str:
    if model_key == "START":
        return "input"
    if model_key in ("BUY", "SHORT", "CLOSE", "SELL", "SKIP"):
        return "output"
    return "default"


def _make_flow_node(
    nid: str, model_key: str, label: str,
    x: float, y: float,
) -> StreamlitFlowNode:
    """Build a StreamlitFlowNode from metadata."""
    return StreamlitFlowNode(
        id=nid,
        pos=(x, y),
        data={"content": label},
        node_type=_node_type(model_key),
        style=_style_for(model_key),
        source_position="right",
        target_position="left",
        draggable=True,
        selectable=True,
        deletable=model_key != "START",
        connectable=True,
    )


def _edge_label(condition: str, threshold: float | None) -> str:
    """Human-readable edge label."""
    if condition == "unconditional":
        return ""
    if threshold is None:
        threshold = 0.5
    if condition == "above":
        return f">= {threshold:.2f}"
    return f"< {threshold:.2f}"


def _edge_style(condition: str) -> dict:
    """Return stroke style dict for edge condition."""
    if condition == "above":
        return deepcopy(_EDGE_ABOVE_STYLE)
    if condition == "below":
        return deepcopy(_EDGE_BELOW_STYLE)
    return deepcopy(_EDGE_UNCONDITIONAL_STYLE)


def _make_flow_edge(
    eid: str, source: str, target: str,
    condition: str = "above", threshold: float | None = 0.5,
) -> StreamlitFlowEdge:
    """Build a StreamlitFlowEdge from metadata."""
    return StreamlitFlowEdge(
        id=eid,
        source=source,
        target=target,
        edge_type="smoothstep",
        animated=True,
        label=_edge_label(condition, threshold),
        style=_edge_style(condition),
        marker_end={"type": "arrowclosed"},
        deletable=True,
    )


def _source_is_start(source_id: str) -> bool:
    """Check whether a source node is the START node."""
    meta = st.session_state.get("_pipe_node_meta", {}).get(source_id, {})
    return meta.get("model_key") == "START"


# ---------------------------------------------------------------------------
# Load / save pipeline JSON
# ---------------------------------------------------------------------------


def _default_pipeline() -> dict:
    return {
        "nodes": [
            {
                "id": "start",
                "model_key": "START",
                "label": "Start",
                "position": {"x": 50, "y": 200},
            }
        ],
        "edges": [],
        "version": 2,
    }


def _load_pipeline() -> dict:
    if PIPELINE_PATH.exists():
        try:
            return json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load pipeline: {}", exc)
    return _default_pipeline()


def _bump_timestamp() -> None:
    """Bump the flow-state timestamp so the React component picks up our changes."""
    from datetime import datetime
    st.session_state["_pipe_timestamp"] = int(datetime.now().timestamp() * 1000)


def _init_state_from_pipeline(pipe: dict) -> None:
    """Populate session_state lists from a pipeline dict.

    Handles backward-compatible loading from v1 format:
    - Edges without ``threshold`` inherit from source node's threshold.
    - Edges from START without ``"unconditional"`` are upgraded.
    """
    # Build a node-threshold lookup for backward compat (v1 format)
    _node_thresh: dict[str, float] = {}
    _node_keys: dict[str, str] = {}
    for n in pipe.get("nodes", []):
        _node_thresh[n["id"]] = n.get("threshold", 0.5) or 0.5
        _node_keys[n["id"]] = n.get("model_key", "")

    node_meta: dict[str, dict] = {}
    nodes: list[StreamlitFlowNode] = []
    for n in pipe.get("nodes", []):
        nid = n["id"]
        mk = n["model_key"]
        lbl = n["label"]
        pos = n.get("position", {"x": 0, "y": 0})
        node_meta[nid] = {"model_key": mk, "label": lbl}
        nodes.append(_make_flow_node(nid, mk, lbl, pos["x"], pos["y"]))

    edge_meta: dict[str, dict] = {}
    edges: list[StreamlitFlowEdge] = []
    for e in pipe.get("edges", []):
        eid = e["id"]
        src = e.get("source", "")
        cond = e.get("condition", "above")
        thr = e.get("threshold")

        # Migrate v1 → v2: START edges become unconditional
        if _node_keys.get(src) == "START" and cond != "unconditional":
            cond = "unconditional"
            thr = None

        # For conditional edges, inherit node threshold if missing
        if cond in ("above", "below") and thr is None:
            thr = _node_thresh.get(src, 0.5)

        # Unconditional edges never have a threshold
        if cond == "unconditional":
            thr = None

        edge_meta[eid] = {"condition": cond, "threshold": thr}
        edges.append(_make_flow_edge(eid, src, e["target"], cond, thr))

    st.session_state["_pipe_nodes"] = nodes
    st.session_state["_pipe_edges"] = edges
    st.session_state["_pipe_node_meta"] = node_meta
    st.session_state["_pipe_edge_meta"] = edge_meta
    _bump_timestamp()


def _export_pipeline_dict() -> dict:
    """Build a pipeline JSON dict from current session_state."""
    nodes_out = []
    for fn in st.session_state["_pipe_nodes"]:
        meta = st.session_state["_pipe_node_meta"].get(fn.id, {})
        nodes_out.append({
            "id": fn.id,
            "model_key": meta.get("model_key", "SKIP"),
            "label": meta.get("label", fn.data.get("content", "")),
            "position": {"x": fn.position["x"], "y": fn.position["y"]},
        })

    edges_out = []
    for fe in st.session_state["_pipe_edges"]:
        emeta = st.session_state["_pipe_edge_meta"].get(fe.id, {})
        cond = emeta.get("condition", "above")
        thr = emeta.get("threshold")
        edges_out.append({
            "id": fe.id,
            "source": fe.source,
            "target": fe.target,
            "condition": cond,
            "threshold": thr,
            "label": _edge_label(cond, thr),
        })

    return {"nodes": nodes_out, "edges": edges_out, "version": 2}


# ---------------------------------------------------------------------------
# Initialise session state (once)
# ---------------------------------------------------------------------------

if "_pipe_nodes" not in st.session_state:
    _init_state_from_pipeline(_load_pipeline())

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
  <h1 style="margin:0; font-family:'Outfit',sans-serif;">Pipeline Editor</h1>
</div>
""", unsafe_allow_html=True)
st.caption(
    "Build a trading strategy by connecting model nodes with edges. "
    "Green edges (above) fire when model output >= threshold. "
    "Red edges (below) fire when output < threshold. "
    "Grey edges from START are unconditional."
)

# ---------------------------------------------------------------------------
# Code generation / parsing helpers
# ---------------------------------------------------------------------------


def _generate_code_from_pipeline() -> str:
    """Generate a Python code representation of the current pipeline."""
    node_meta = st.session_state.get("_pipe_node_meta", {})
    edges = st.session_state.get("_pipe_edges", [])
    edge_meta = st.session_state.get("_pipe_edge_meta", {})

    # Build adjacency: source_id -> list of (target_id, condition, threshold)
    adjacency: dict[str, list[tuple[str, str, float | None]]] = {}
    for e in edges:
        emeta = edge_meta.get(e.id, {})
        cond = emeta.get("condition", "above")
        thr = emeta.get("threshold")
        adjacency.setdefault(e.source, []).append((e.target, cond, thr))

    lines = [
        '"""',
        "Auto-generated pipeline code.",
        "Edit thresholds below, then click 'Apply' to sync back to the visual editor.",
        '"""',
        "",
        "",
        "def pipeline(bar):",
        '    """Evaluate the pipeline for a single bar."""',
    ]

    # Find START node
    start_id = None
    for nid, meta in node_meta.items():
        if meta.get("model_key") == "START":
            start_id = nid
            break

    if start_id is None:
        lines.append("    return 'SKIP'  # No START node found")
        return "\n".join(lines)

    # Walk from START through model nodes to action outputs
    start_targets = adjacency.get(start_id, [])
    if not start_targets:
        lines.append("    return 'SKIP'  # No edges from START")
        return "\n".join(lines)

    # For each model connected from START, generate decision branches
    for i, (model_id, _, _) in enumerate(start_targets):
        model_meta = node_meta.get(model_id, {})
        model_key = model_meta.get("model_key", "unknown")
        var_name = model_key.replace("-", "_").replace(" ", "_").lower()

        lines.append(f"    # Model: {model_key}")
        lines.append(f"    {var_name} = bar.predictions.get('{model_key}', 0.0)")
        lines.append("")

        # Get edges from this model to action nodes
        model_targets = adjacency.get(model_id, [])
        if not model_targets:
            lines.append(f"    # No edges from {model_key}")
            lines.append("")
            continue

        first = True
        for target_id, cond, thr in model_targets:
            target_meta = node_meta.get(target_id, {})
            target_key = target_meta.get("model_key", "SKIP")
            target_label = target_meta.get("label", target_key)

            if cond == "unconditional":
                keyword = "if" if first else "elif"
                lines.append(f"    {keyword} True:  # unconditional → {target_label}")
                lines.append(f"        return '{target_key}'")
            elif cond == "above":
                keyword = "if" if first else "elif"
                thr_val = thr if thr is not None else 0.5
                lines.append(f"    {keyword} {var_name} >= {thr_val:.2f}:  # → {target_label}")
                lines.append(f"        return '{target_key}'")
            elif cond == "below":
                keyword = "if" if first else "elif"
                thr_val = thr if thr is not None else 0.5
                lines.append(f"    {keyword} {var_name} < {thr_val:.2f}:  # → {target_label}")
                lines.append(f"        return '{target_key}'")
            first = False

        lines.append("")

    lines.append("    return 'SKIP'  # Default fallback")
    return "\n".join(lines)


def _apply_code_to_pipeline(code: str) -> tuple[bool, str]:
    """Parse threshold values from code and update edge metadata.

    Returns (success, message).
    """
    edge_meta = st.session_state.get("_pipe_edge_meta", {})
    node_meta = st.session_state.get("_pipe_node_meta", {})
    edges = st.session_state.get("_pipe_edges", [])

    # Build lookup: model_key -> node_id
    key_to_nid: dict[str, str] = {}
    for nid, meta in node_meta.items():
        mk = meta.get("model_key", "")
        if mk not in ("START", "BUY", "SHORT", "CLOSE", "SELL", "SKIP"):
            key_to_nid[mk] = nid

    # Parse threshold patterns from code:
    #   var >= 0.50   → above
    #   var < 0.50    → below
    pattern = re.compile(
        r"(\w+)\s*(>=?|<)\s*([0-9]*\.?[0-9]+)\s*:\s*#\s*.*?→\s*(\w+)"
    )

    updates = 0
    errors = []

    for match in pattern.finditer(code):
        var_name = match.group(1)
        operator = match.group(2)
        threshold = float(match.group(3))
        target_action = match.group(4)

        cond = "above" if ">=" in operator or ">" in operator else "below"

        # Find matching edge(s)
        for e in edges:
            emeta = edge_meta.get(e.id, {})
            src_meta = node_meta.get(e.source, {})
            tgt_meta = node_meta.get(e.target, {})
            src_key = src_meta.get("model_key", "")
            tgt_key = tgt_meta.get("model_key", "")

            # Match by: source model var name matches AND target action matches
            src_var = src_key.replace("-", "_").replace(" ", "_").lower()
            if src_var == var_name and tgt_key == target_action:
                old_cond = emeta.get("condition")
                old_thr = emeta.get("threshold")
                if old_cond != cond or old_thr != threshold:
                    emeta["condition"] = cond
                    emeta["threshold"] = threshold
                    # Rebuild the flow edge
                    for i, fe in enumerate(st.session_state["_pipe_edges"]):
                        if fe.id == e.id:
                            st.session_state["_pipe_edges"][i] = _make_flow_edge(
                                e.id, e.source, e.target, cond, threshold,
                            )
                            break
                    updates += 1

    if updates > 0:
        _bump_timestamp()
        return True, f"Applied {updates} threshold update(s) to the visual pipeline."
    elif errors:
        return False, "Errors: " + "; ".join(errors)
    else:
        return True, "No threshold changes detected — pipeline unchanged."


# ---------------------------------------------------------------------------
# Layout: tabs (Visual / Code)
# ---------------------------------------------------------------------------

tab_visual, tab_code = st.tabs(["Visual", "Code"])

# ===========================================================================
# Tab 1: Visual — toolbox sidebar + canvas
# ===========================================================================

with tab_visual:
    col_toolbox, col_canvas = st.columns([1, 4])

    # ======================== Toolbox ========================
    with col_toolbox:
        st.markdown("**Add Nodes**")

        registry_names = _get_registry_names()

        if not registry_names:
            st.info("No models loaded — sync from Google Drive first.")
        else:
            st.markdown("*Models:*")
            for model_name in registry_names:
                if st.button(
                    f"+ {model_name}",
                    key=f"_pipe_add_{model_name}",
                    use_container_width=True,
                ):
                    nid = f"{model_name}_{uuid.uuid4().hex[:6]}"
                    st.session_state["_pipe_node_meta"][nid] = {
                        "model_key": model_name,
                        "label": model_name,
                    }
                    st.session_state["_pipe_nodes"].append(
                        _make_flow_node(nid, model_name, model_name, 300, 200)
                    )
                    _bump_timestamp()
                    st.rerun()

        st.markdown("---")
        st.markdown("*Actions:*")
        for action in ("BUY", "SHORT", "CLOSE", "SKIP"):
            if st.button(
                f"+ {action}",
                key=f"_pipe_add_{action}",
                use_container_width=True,
            ):
                nid = f"{action.lower()}_{uuid.uuid4().hex[:6]}"
                st.session_state["_pipe_node_meta"][nid] = {
                    "model_key": action,
                    "label": action,
                }
                st.session_state["_pipe_nodes"].append(
                    _make_flow_node(nid, action, action, 700, 200)
                )
                _bump_timestamp()
                st.rerun()

        st.markdown("---")
        st.markdown("*New Edge Defaults:*")
        st.caption(
            "Condition and threshold for new edges drawn on canvas. "
            "Edges from START are always unconditional."
        )
        new_edge_condition = st.radio(
            "Condition",
            options=["above", "below"],
            index=0,
            key="_pipe_new_edge_cond",
            horizontal=True,
        )
        new_edge_threshold = st.number_input(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            format="%.2f",
            key="_pipe_new_edge_thresh",
        )

    # ======================== Canvas ========================
    with col_canvas:
        curr_state = StreamlitFlowState(
            st.session_state["_pipe_nodes"],
            st.session_state["_pipe_edges"],
            timestamp=st.session_state["_pipe_timestamp"],
        )

        result = streamlit_flow(
            "pipeline_canvas",
            curr_state,
            height=600,
            fit_view=False,
            style={"width": "100%"},
            show_controls=True,
            show_minimap=True,
            pan_on_drag=True,
            allow_zoom=True,
            allow_new_edges=True,
            animate_new_edges=True,
            get_node_on_click=True,
            get_edge_on_click=True,
            hide_watermark=True,
        )

        # --- Merge canvas changes back into session_state ---
        if result is not None and result.timestamp != st.session_state["_pipe_timestamp"]:
            st.session_state["_pipe_timestamp"] = result.timestamp

            old_node_ids = {n.id for n in st.session_state["_pipe_nodes"]}
            old_edge_ids = {e.id for e in st.session_state["_pipe_edges"]}

            # --- Update nodes (positions, deletions) ---
            if result.nodes is not None and len(result.nodes) > 0:
                result_node_ids = {n.id for n in result.nodes}

                pos_map = {n.id: n.position for n in result.nodes}
                for n in st.session_state["_pipe_nodes"]:
                    if n.id in pos_map:
                        n.position = pos_map[n.id]

                deleted_nodes = old_node_ids - result_node_ids
                if deleted_nodes:
                    st.session_state["_pipe_nodes"] = [
                        n for n in st.session_state["_pipe_nodes"]
                        if n.id not in deleted_nodes
                    ]
                    for nid in deleted_nodes:
                        st.session_state["_pipe_node_meta"].pop(nid, None)
                    remaining_ids = {n.id for n in st.session_state["_pipe_nodes"]}
                    st.session_state["_pipe_edges"] = [
                        e for e in st.session_state["_pipe_edges"]
                        if e.source in remaining_ids and e.target in remaining_ids
                    ]
                    valid_edge_ids = {e.id for e in st.session_state["_pipe_edges"]}
                    for eid in list(st.session_state["_pipe_edge_meta"]):
                        if eid not in valid_edge_ids:
                            st.session_state["_pipe_edge_meta"].pop(eid, None)

            # --- Update edges (new edges, deletions) ---
            if result.edges is not None and len(result.edges) > 0:
                result_edge_ids = {e.id for e in result.edges}

                for re_edge in result.edges:
                    if re_edge.id not in old_edge_ids:
                        # Determine condition based on source node type
                        if _source_is_start(re_edge.source):
                            cond = "unconditional"
                            thr = None
                        else:
                            cond = st.session_state.get("_pipe_new_edge_cond", "above")
                            thr = st.session_state.get("_pipe_new_edge_thresh", 0.5)

                        st.session_state["_pipe_edge_meta"][re_edge.id] = {
                            "condition": cond,
                            "threshold": thr,
                        }
                        st.session_state["_pipe_edges"].append(
                            _make_flow_edge(
                                re_edge.id, re_edge.source, re_edge.target,
                                cond, thr,
                            )
                        )
                        logger.info(
                            "New edge {} ({} -> {}) condition={} threshold={}",
                            re_edge.id, re_edge.source, re_edge.target, cond, thr,
                        )

                deleted_edges = old_edge_ids - result_edge_ids
                if deleted_edges:
                    st.session_state["_pipe_edges"] = [
                        e for e in st.session_state["_pipe_edges"]
                        if e.id not in deleted_edges
                    ]
                    for eid in deleted_edges:
                        st.session_state["_pipe_edge_meta"].pop(eid, None)

            elif result.edges is not None and len(result.edges) == 0:
                if result.nodes is not None and len(result.nodes) > 0:
                    if len(st.session_state["_pipe_edges"]) > 0:
                        st.session_state["_pipe_edges"] = []
                        st.session_state["_pipe_edge_meta"] = {}

    # -------------------------------------------------------------------
    # Selected Node/Edge Panel
    # -------------------------------------------------------------------

    st.markdown("---")

    selected_id = result.selected_id if result is not None else None

    if selected_id and selected_id in st.session_state["_pipe_node_meta"]:
        meta = st.session_state["_pipe_node_meta"][selected_id]
        model_key = meta["model_key"]

        st.subheader(f"Node: {meta['label']}")

        col_info, col_del = st.columns([4, 1])

        with col_info:
            st.text_input(
                "Model Key",
                value=model_key,
                disabled=True,
                key="_pipe_sel_model_key",
            )

        with col_del:
            if model_key != "START":
                if st.button("Delete Node", key="_pipe_del_node", type="primary"):
                    st.session_state["_pipe_nodes"] = [
                        n for n in st.session_state["_pipe_nodes"]
                        if n.id != selected_id
                    ]
                    st.session_state["_pipe_node_meta"].pop(selected_id, None)
                    st.session_state["_pipe_edges"] = [
                        e for e in st.session_state["_pipe_edges"]
                        if e.source != selected_id and e.target != selected_id
                    ]
                    valid_eids = {e.id for e in st.session_state["_pipe_edges"]}
                    for eid in list(st.session_state["_pipe_edge_meta"]):
                        if eid not in valid_eids:
                            st.session_state["_pipe_edge_meta"].pop(eid, None)
                    _bump_timestamp()
                    st.rerun()
            else:
                st.caption("Cannot delete START.")

    elif selected_id and selected_id in st.session_state.get("_pipe_edge_meta", {}):
        edge_meta = st.session_state["_pipe_edge_meta"][selected_id]

        # Find the source node id for this edge
        _sel_edge_obj = None
        for _e in st.session_state["_pipe_edges"]:
            if _e.id == selected_id:
                _sel_edge_obj = _e
                break

        _is_start_edge = _sel_edge_obj is not None and _source_is_start(_sel_edge_obj.source)

        if _is_start_edge:
            # ---- Unconditional edge (from START) ----
            st.subheader("Edge Settings")
            st.caption("Edges from START are unconditional — always taken.")

            # Force unconditional if not already
            if edge_meta.get("condition") != "unconditional":
                edge_meta["condition"] = "unconditional"
                edge_meta["threshold"] = None
                if _sel_edge_obj is not None:
                    for i, e in enumerate(st.session_state["_pipe_edges"]):
                        if e.id == selected_id:
                            st.session_state["_pipe_edges"][i] = _make_flow_edge(
                                selected_id, e.source, e.target,
                                "unconditional", None,
                            )
                            break
                    _bump_timestamp()
                    st.rerun()

            if st.button("Delete Edge", key="_pipe_del_edge", type="primary"):
                st.session_state["_pipe_edges"] = [
                    e for e in st.session_state["_pipe_edges"]
                    if e.id != selected_id
                ]
                st.session_state["_pipe_edge_meta"].pop(selected_id, None)
                _bump_timestamp()
                st.rerun()

        else:
            # ---- Conditional edge (from model node) ----
            st.subheader("Edge Settings")

            col_cond, col_thresh, col_del_e = st.columns([2, 2, 1])

            _cur_cond = edge_meta.get("condition", "above")
            _cur_thresh = edge_meta.get("threshold")
            if _cur_thresh is None:
                _cur_thresh = 0.5
            _cur_thresh = float(_cur_thresh)

            # Map condition to radio index — treat unconditional/unknown as above
            _cond_idx = 0 if _cur_cond != "below" else 1

            with col_cond:
                new_cond = st.radio(
                    "Condition",
                    options=["above", "below"],
                    index=_cond_idx,
                    key="_pipe_edge_cond_edit",
                    horizontal=True,
                )

            with col_thresh:
                new_thresh = st.number_input(
                    "Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=_cur_thresh,
                    step=0.01,
                    format="%.2f",
                    key="_pipe_edge_thresh_edit",
                )

            # Persist changes
            cond_changed = new_cond != edge_meta.get("condition")
            thresh_changed = new_thresh != edge_meta.get("threshold")
            if cond_changed or thresh_changed:
                edge_meta["condition"] = new_cond
                edge_meta["threshold"] = new_thresh
                for i, e in enumerate(st.session_state["_pipe_edges"]):
                    if e.id == selected_id:
                        st.session_state["_pipe_edges"][i] = _make_flow_edge(
                            selected_id, e.source, e.target, new_cond, new_thresh,
                        )
                        break
                _bump_timestamp()
                st.rerun()

            with col_del_e:
                if st.button("Delete Edge", key="_pipe_del_edge", type="primary"):
                    st.session_state["_pipe_edges"] = [
                        e for e in st.session_state["_pipe_edges"]
                        if e.id != selected_id
                    ]
                    st.session_state["_pipe_edge_meta"].pop(selected_id, None)
                    _bump_timestamp()
                    st.rerun()
    else:
        st.caption("Click a node or edge on the canvas to edit it.")

    # -------------------------------------------------------------------
    # Validation + Save
    # -------------------------------------------------------------------

    st.markdown("---")

    col_validate, col_save, col_reset, _ = st.columns([1, 1, 1, 2])

    with col_validate:
        if st.button("Validate", use_container_width=True, key="_pipe_validate"):
            from pipeline.schema import Pipeline as PipelineModel
            from pipeline.validator import validate_pipeline

            pipe_dict = _export_pipeline_dict()
            try:
                pipeline_obj = PipelineModel.model_validate(pipe_dict)
                errors = validate_pipeline(
                    pipeline_obj, _get_registry_names() or None
                )
                if not errors:
                    st.success("Pipeline is valid.")
                else:
                    for err in errors:
                        if err.level == "error":
                            st.error(err.message)
                        else:
                            st.warning(err.message)
            except Exception as exc:
                st.error(f"Validation error: {exc}")

    with col_save:
        if st.button("Save Pipeline", use_container_width=True, key="_pipe_save"):
            from pipeline.schema import Pipeline as PipelineModel
            from pipeline.validator import validate_pipeline

            pipe_dict = _export_pipeline_dict()
            try:
                pipeline_obj = PipelineModel.model_validate(pipe_dict)
                errors = [
                    e for e in validate_pipeline(
                        pipeline_obj, _get_registry_names() or None
                    )
                    if e.level == "error"
                ]
                if errors:
                    for err in errors:
                        st.error(err.message)
                    st.error("Fix errors before saving.")
                else:
                    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    PIPELINE_PATH.write_text(
                        json.dumps(pipe_dict, indent=2), encoding="utf-8"
                    )
                    st.toast("Pipeline saved — will be used in next backtest.")
                    logger.info("Pipeline saved to {}", PIPELINE_PATH)
            except Exception as exc:
                st.error(f"Save error: {exc}")

    with col_reset:
        if st.button("Reset", use_container_width=True, key="_pipe_reset"):
            _init_state_from_pipeline(_default_pipeline())
            st.rerun()

    # -------------------------------------------------------------------
    # Raw JSON viewer
    # -------------------------------------------------------------------

    with st.expander("View Pipeline JSON"):
        st.json(_export_pipeline_dict())

# ===========================================================================
# Tab 2: Code — Python code editor with bidirectional sync
# ===========================================================================

with tab_code:
    st.caption(
        "Python representation of the pipeline. "
        "Edit thresholds in the code, then click **Apply** to sync changes "
        "back to the visual editor."
    )

    # Generate code from current pipeline state
    generated_code = _generate_code_from_pipeline()

    # Try streamlit-ace, fallback to text_area
    _use_ace = False
    try:
        from streamlit_ace import st_ace
        _use_ace = True
    except ImportError:
        pass

    if _use_ace:
        edited_code = st_ace(
            value=generated_code,
            language="python",
            theme="monokai",
            height=500,
            key="_pipe_code_editor",
            auto_update=False,
        )
    else:
        edited_code = st.text_area(
            "Pipeline Code",
            value=generated_code,
            height=500,
            key="_pipe_code_editor_fallback",
        )

    col_apply, col_refresh, _ = st.columns([1, 1, 3])

    with col_apply:
        if st.button("Apply to Visual", use_container_width=True, key="_pipe_code_apply"):
            if edited_code:
                ok, msg = _apply_code_to_pipeline(edited_code)
                if ok:
                    st.success(msg)
                    if "update" in msg.lower():
                        st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("No code to apply.")

    with col_refresh:
        if st.button("Refresh from Visual", use_container_width=True, key="_pipe_code_refresh"):
            st.rerun()
