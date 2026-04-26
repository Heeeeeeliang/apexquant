"""
Pipeline executor — traverses the visual pipeline graph for a single bar.

Given a :class:`Pipeline` graph and a bar with signals, the executor
walks from the START node, following edges based on their condition:

* ``"unconditional"`` — always taken (START outgoing edges).
* ``"above"`` — taken when model output >= edge threshold.
* ``"below"`` — taken when model output < edge threshold.

The traversal continues until it reaches a terminal node
(BUY / SHORT / CLOSE / SKIP).

Usage::

    from pipeline import Pipeline, PipelineExecutor

    pipeline = Pipeline.model_validate_json(path.read_text())
    executor = PipelineExecutor(pipeline)
    action = executor.run(bar, bar.signals)
    # action is "BUY", "SHORT", "CLOSE", or "SKIP"
"""

__all__ = ["PipelineExecutor"]

from collections import defaultdict, deque
from typing import Any, Literal

from loguru import logger

from pipeline.schema import Pipeline, PipelineEdge, PipelineNode


class PipelineExecutor:
    """Traverses a pipeline graph to produce a trade action.

    Edge-driven design: each edge carries its own condition and threshold.
    At each node the executor evaluates outgoing edges to find the first
    whose condition is satisfied.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        trend_bypass_config: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._node_map: dict[str, PipelineNode] = {
            n.id: n for n in pipeline.nodes
        }
        # Adjacency: node_id -> list[PipelineEdge]
        self._outgoing: dict[str, list[PipelineEdge]] = {}
        for edge in pipeline.edges:
            self._outgoing.setdefault(edge.source, []).append(edge)

        # Trend bypass: detect vol-gate nodes by model_key containing "vol"
        cfg = trend_bypass_config or {}
        self._tb_period: int = cfg.get("trend_bypass_period", 20)
        self._tb_pct: float = cfg.get("trend_bypass_pct", 0.03)
        self._tb_min_vol: float = cfg.get("trend_bypass_min_vol", 0.35)
        self._tb_enabled: bool = cfg.get("enabled", True)
        self._close_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._tb_period + 1)
        )

    def run(
        self,
        bar: object,
        signals: object | None = None,
    ) -> Literal["BUY", "SHORT", "CLOSE", "SELL", "SKIP"]:
        """Traverse the pipeline for a single bar.

        Returns:
            Terminal action string.
        """
        # Track close history for trend bypass
        ticker = getattr(bar, "ticker", "")
        close = getattr(bar, "close", None)
        if ticker and close is not None:
            self._close_history[ticker].append(close)

        start_nodes = [n for n in self.pipeline.nodes if n.is_start]
        if not start_nodes:
            logger.warning("No START node in pipeline, returning SKIP")
            return "SKIP"

        current = start_nodes[0].id
        visited: set[str] = set()

        while True:
            if current in visited:
                logger.warning("Cycle detected at node '{}', returning SKIP", current)
                return "SKIP"
            visited.add(current)

            node = self._node_map.get(current)
            if node is None:
                logger.warning("Node '{}' not found, returning SKIP", current)
                return "SKIP"

            # Terminal nodes
            if node.model_key in ("BUY", "SHORT", "CLOSE", "SELL", "SKIP"):
                return node.model_key  # type: ignore[return-value]

            # Non-terminal: evaluate outgoing edges
            edges = self._outgoing.get(current, [])
            if not edges:
                logger.warning("Node '{}' has no outgoing edges, returning SKIP", node.label)
                return "SKIP"

            # For START or any node with an unconditional edge, take it
            # without evaluating model output.
            unconditional = [e for e in edges if e.condition == "unconditional"]
            if unconditional:
                current = unconditional[0].target
                continue

            # Model node: get prediction, evaluate conditional edges
            prob = self._get_prob(node.model_key, signals, bar)

            # Trend bypass: for vol-gate nodes, boost prob when in a
            # strong trend so the gate doesn't block trending markets.
            if self._tb_enabled and self._is_vol_node(node) and ticker:
                prob = self._apply_trend_bypass(prob, ticker)

            next_id = self._pick_edge(edges, prob)

            if next_id is None:
                logger.warning(
                    "Node '{}' has no matching edge for prob={:.4f}, returning SKIP",
                    node.label, prob,
                )
                return "SKIP"

            current = next_id

    @staticmethod
    def _is_vol_node(node: PipelineNode) -> bool:
        """Check if a node is a volatility gate."""
        key = (node.model_key or "").lower()
        return "vol" in key

    def _apply_trend_bypass(self, prob: float, ticker: str) -> float:
        """If a strong trend is detected and prob exceeds the minimum
        floor, boost the effective probability above the gate threshold."""
        if prob > self._tb_min_vol:
            history = self._close_history.get(ticker)
            if history and len(history) >= self._tb_period:
                past_close = history[-self._tb_period]
                current_close = history[-1]
                if past_close > 0:
                    change = abs(current_close - past_close) / past_close
                    if change > self._tb_pct:
                        # Boost to 1.0 so "above" edges always fire
                        return 1.0
        return prob

    @staticmethod
    def _pick_edge(edges: list[PipelineEdge], prob: float) -> str | None:
        """Return target of the first edge whose condition is satisfied."""
        for edge in edges:
            thr = edge.threshold if edge.threshold is not None else 0.5
            if edge.condition == "above" and prob >= thr:
                return edge.target
            if edge.condition == "below" and prob < thr:
                return edge.target
            if edge.condition == "unconditional":
                return edge.target
        return None

    @staticmethod
    def _get_prob(
        model_key: str,
        signals: object | None,
        bar: object,
    ) -> float:
        """Resolve prediction probability for a model key.

        Tries multiple lookup strategies:
        1. signals.get(model_key) — direct registry name lookup
        2. bar.predictions[model_key] — direct label lookup
        3. bar.predictions values scan for matching label suffix
        """
        # Try SignalsProxy (path-based or label-based)
        if signals is not None and hasattr(signals, "get"):
            val = signals.get(model_key, None)
            if val is not None:
                return float(val)

        # Try bar.predictions dict
        preds = getattr(bar, "predictions", None)
        if isinstance(preds, dict):
            if model_key in preds:
                v = preds[model_key]
                return float(v.prob) if hasattr(v, "prob") else float(v)
            for key, v in preds.items():
                if model_key in key or key in model_key:
                    return float(v.prob) if hasattr(v, "prob") else float(v)

        available = list(preds.keys()) if isinstance(preds, dict) else []
        raise KeyError(
            f"Model '{model_key}' not found in predictions. "
            f"Available keys: {available}. "
            f"Check pipeline configuration."
        )
