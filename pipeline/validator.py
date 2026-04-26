"""
Pipeline graph validator.

Checks structural integrity before save / run:

1. Exactly one START node.
2. At least one BUY, SHORT, or CLOSE terminal.
3. No cycles (DAG check).
4. START outgoing edges must be "unconditional".
5. Model nodes need at least one conditional edge (above or below).
6. All model_key values exist in the registry or are special keys.
7. All edge source/target reference valid node IDs.
"""

__all__ = ["validate_pipeline", "ValidationError"]

from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipeline.schema import SPECIAL_KEYS, TERMINAL_KEYS, Pipeline


@dataclass
class ValidationError:
    """A single validation issue."""

    level: str  # "error" | "warning"
    message: str


def validate_pipeline(
    pipeline: Pipeline,
    registry_names: list[str] | None = None,
) -> list[ValidationError]:
    """Validate a pipeline and return a list of issues.

    Args:
        pipeline: The pipeline to validate.
        registry_names: Known model registry keys.  If ``None``,
            model-key existence checks are skipped.

    Returns:
        List of :class:`ValidationError`.  Empty means valid.
    """
    errors: list[ValidationError] = []
    node_map = {n.id: n for n in pipeline.nodes}
    node_ids = set(node_map.keys())

    # 1. Exactly one START
    starts = [n for n in pipeline.nodes if n.is_start]
    if len(starts) == 0:
        errors.append(ValidationError("error", "Pipeline must have a START node."))
    elif len(starts) > 1:
        errors.append(ValidationError("error", f"Multiple START nodes found ({len(starts)})."))

    # 2. At least one actionable terminal (BUY, SHORT, CLOSE, or SELL)
    terminals = [n for n in pipeline.nodes if n.model_key in ("BUY", "SHORT", "CLOSE", "SELL")]
    if not terminals:
        errors.append(ValidationError("error", "Pipeline must have at least one BUY, SHORT, or CLOSE node."))

    # 3. Edge references valid nodes
    for edge in pipeline.edges:
        if edge.source not in node_ids:
            errors.append(ValidationError("error", f"Edge '{edge.id}' source '{edge.source}' not found."))
        if edge.target not in node_ids:
            errors.append(ValidationError("error", f"Edge '{edge.id}' target '{edge.target}' not found."))

    # 4. Build adjacency for cycle check
    # adjacency maps node_id -> list of target node_ids
    adj: dict[str, list[str]] = {}
    for edge in pipeline.edges:
        adj.setdefault(edge.source, []).append(edge.target)

    if _has_cycle(adj, node_ids):
        errors.append(ValidationError("error", "Pipeline contains a cycle."))

    # 5. Per-node edge checks
    # Build per-node edge condition sets
    node_edge_conds: dict[str, set[str]] = {}
    for edge in pipeline.edges:
        node_edge_conds.setdefault(edge.source, set()).add(edge.condition)

    for node in pipeline.nodes:
        if node.is_terminal:
            continue

        conds = node_edge_conds.get(node.id, set())

        if node.is_start:
            if not conds:
                errors.append(ValidationError("error", "START node has no outgoing edge."))
            continue

        # Model nodes need at least one conditional edge
        has_above = "above" in conds
        has_below = "below" in conds
        if not has_above and not has_below:
            errors.append(ValidationError(
                "warning",
                f"Node '{node.label}' ({node.id}) has no conditional edges — "
                f"pipeline may get stuck.",
            ))
        elif not has_above:
            errors.append(ValidationError(
                "warning",
                f"Node '{node.label}' ({node.id}) missing 'above' edge — "
                f"pipeline may get stuck.",
            ))
        elif not has_below:
            errors.append(ValidationError(
                "warning",
                f"Node '{node.label}' ({node.id}) missing 'below' edge — "
                f"pipeline may get stuck.",
            ))

    # 6. Model keys exist in registry
    if registry_names is not None:
        known = set(registry_names) | SPECIAL_KEYS
        for node in pipeline.nodes:
            if node.model_key not in known:
                errors.append(ValidationError(
                    "warning",
                    f"Node '{node.label}' uses model_key '{node.model_key}' "
                    f"which is not in the registry.",
                ))

    return errors


def _has_cycle(adj: dict[str, list[str]], all_nodes: set[str]) -> bool:
    """DFS cycle detection on the adjacency graph."""
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in all_nodes}

    def dfs(node: str) -> bool:
        color[node] = GREY
        for target in adj.get(node, []):
            if target not in color:
                continue
            if color[target] == GREY:
                return True
            if color[target] == WHITE and dfs(target):
                return True
        color[node] = BLACK
        return False

    for node in all_nodes:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False
