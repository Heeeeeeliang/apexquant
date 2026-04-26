"""
Pydantic models for the visual pipeline graph.

The pipeline is a directed acyclic graph where each node is either a
model predictor or a terminal action (BUY / SHORT / CLOSE / SKIP).
Edges carry all decision logic: a condition (``"above"`` / ``"below"``
/ ``"unconditional"``) and an optional threshold.  ``SELL`` is kept as
a backward-compatible alias for ``CLOSE``.

Serialised to / loaded from ``config/pipeline.json``.
"""

__all__ = ["PipelineNode", "PipelineEdge", "Pipeline"]

from typing import Literal

from pydantic import BaseModel, Field


SPECIAL_KEYS = frozenset({"START", "BUY", "SHORT", "CLOSE", "SELL", "SKIP"})
TERMINAL_KEYS = frozenset({"BUY", "SHORT", "CLOSE", "SELL", "SKIP"})


class PipelineNode(BaseModel):
    """A single node in the pipeline graph."""

    id: str
    model_key: str
    label: str
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})

    # Legacy field — kept so old pipeline.json files still parse.
    # Ignored by the executor (threshold lives on edges now).
    threshold: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.model_key in TERMINAL_KEYS

    @property
    def is_start(self) -> bool:
        return self.model_key == "START"

    @property
    def is_model(self) -> bool:
        return self.model_key not in SPECIAL_KEYS


class PipelineEdge(BaseModel):
    """A threshold-gated edge between two nodes.

    Each edge carries the full decision logic:

    * ``"unconditional"`` — always taken (used for START outgoing edges).
      ``threshold`` must be ``None``.
    * ``"above"`` — taken when the source model's output >= threshold.
    * ``"below"`` — taken when the source model's output < threshold.
    """

    id: str
    source: str
    target: str
    condition: Literal["above", "below", "unconditional"]
    threshold: float | None = None
    label: str = ""


class Pipeline(BaseModel):
    """Complete pipeline graph."""

    nodes: list[PipelineNode]
    edges: list[PipelineEdge]
    version: int = 1

    def get_node(self, node_id: str) -> PipelineNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}
