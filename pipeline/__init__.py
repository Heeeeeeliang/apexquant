"""
ApexQuant Pipeline Module
=========================

Visual drag-and-drop pipeline editor backend.  Users connect model
nodes with threshold-gated edges to define a trading strategy with
zero code.

Core types:

- :class:`Pipeline` — full pipeline graph (nodes + edges)
- :class:`PipelineNode` — single model / action node
- :class:`PipelineEdge` — threshold-gated connection between nodes
- :class:`PipelineExecutor` — traverses the graph for a single bar
- :func:`validate_pipeline` — checks graph integrity before save

Usage::

    from pipeline import Pipeline, PipelineExecutor, validate_pipeline
"""

__all__ = [
    "Pipeline",
    "PipelineNode",
    "PipelineEdge",
    "PipelineExecutor",
    "validate_pipeline",
]

from pipeline.schema import Pipeline, PipelineEdge, PipelineNode
from pipeline.executor import PipelineExecutor
from pipeline.validator import validate_pipeline
