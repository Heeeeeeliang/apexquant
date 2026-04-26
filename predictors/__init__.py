"""
ApexQuant Predictors Module
============================

Three-layer task decomposition framework:

- **Layer 1 — Volatility prediction** (LightGBM, DA=83.6%)
- **Layer 2 — Turning-point detection** (Multi-Scale CNN, AUC=0.826)
- **Layer 3 — Trade filtering** (LightGBM meta-label, WR=64.5%)

Core types:

- :class:`Predictor` — new per-bar prediction interface
- :class:`BasePredictor` — legacy batch interface (backward compat)
- :class:`PredictionResult` — single-predictor output carrier
- :class:`AggregatedSignal` — combined aggregator output
- :class:`Context` — ambient runtime state for predictors
- :class:`ProbabilityCalibrator` — calibrate raw scores to probabilities
- :class:`FactorLayer` — normalise predictions into feature vectors
- :class:`LearnedAggregator` — learn optimal signal weights from validation data

All predictors register themselves via ``REGISTRY.register()`` so that
the aggregator can discover them without hard-coded imports.

Usage::

    from predictors import REGISTRY, Predictor, PredictionResult, Context
    from predictors import ProbabilityCalibrator, FactorLayer, LearnedAggregator
"""

__all__ = [
    "REGISTRY",
    "BasePredictor",
    "Predictor",
    "Context",
    "PredictionResult",
    "AggregatedSignal",
    "ProbabilityCalibrator",
    "FactorLayer",
    "LearnedAggregator",
]

from predictors.result import AggregatedSignal, PredictionResult
from predictors.base import BasePredictor, Context, Predictor
from predictors.registry import REGISTRY
from predictors.calibrator import ProbabilityCalibrator
from predictors.factor_layer import FactorLayer
from predictors.aggregator import LearnedAggregator

from loguru import logger


def _discover_and_register(models_dir: str = "models") -> None:
    """Scan ``models/`` for ``meta.json`` files and register adapters.

    Each ``meta.json`` must contain at minimum::

        {
            "adapter": "lightgbm" | "multiscale_cnn",
            "output_label": "vol_prob",
            "name": "vol_lgb"         // optional, derived from path
        }

    The adapter type is resolved from ``meta["adapter"]``:

    - ``"lightgbm"`` with layer1 path → :class:`VolAdapter`
    - ``"lightgbm"`` with layer3 path → :class:`MetaAdapter`
    - ``"multiscale_cnn"``            → :class:`CnnAdapter`
    """
    import json
    from pathlib import Path

    root = Path(models_dir)
    if not root.exists():
        logger.debug("models/ directory not found, skipping auto-discovery")
        return

    # Import adapter classes (fail-safe per class)
    _adapter_classes: dict[str, type] = {}
    try:
        from predictors.adapters.vol_adapter import VolAdapter
        _adapter_classes["vol"] = VolAdapter
    except Exception as exc:
        logger.warning("Could not import VolAdapter: {}", exc)
    try:
        from predictors.adapters.meta_adapter import MetaAdapter
        _adapter_classes["meta"] = MetaAdapter
    except Exception as exc:
        logger.warning("Could not import MetaAdapter: {}", exc)
    try:
        from predictors.adapters.cnn_adapter import CnnAdapter
        _adapter_classes["cnn"] = CnnAdapter
    except Exception as exc:
        logger.warning("Could not import CnnAdapter: {}", exc)

    for meta_path in sorted(root.rglob("meta.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            logger.warning("Bad meta.json at {}: {}", meta_path, exc)
            continue

        adapter_type = meta.get("adapter", "")
        output_label = meta.get("output_label") or meta.get("output") or ""
        if not output_label:
            continue

        model_dir = meta_path.parent
        rel_path = model_dir.relative_to(root)
        parts = rel_path.parts  # e.g. ("layer1", "volatility", "lightgbm_v3")
        layer = parts[0] if parts else ""

        # Derive registry name from meta or path
        name = meta.get("name", str(rel_path).replace("/", "_").replace("\\", "_"))

        # Already registered (e.g. from config-based registration)?
        if name in REGISTRY:
            continue

        # Resolve adapter class
        cls = None
        if adapter_type == "lightgbm":
            # Classifiers (predict_proba) → MetaAdapter;
            # Regressors (predict + sigmoid) → VolAdapter.
            # Detect classifiers by: layer3 path, output="probability", or
            # presence of a task field (bottom/top).
            is_classifier = (
                layer == "layer3"
                or meta.get("output") == "probability"
                or meta.get("task") in ("bottom", "top")
            )
            if is_classifier:
                cls = _adapter_classes.get("meta")
            else:
                cls = _adapter_classes.get("vol")
        elif adapter_type == "multiscale_cnn":
            cls = _adapter_classes.get("cnn")

        if cls is None:
            logger.debug(
                "No adapter class for type='{}' at {}, skipping",
                adapter_type, meta_path,
            )
            continue

        try:
            adapter = cls(
                name=name,
                output_label=output_label,
                model_dir=str(model_dir),
            )
            REGISTRY.register(adapter)
            logger.info("Auto-discovered adapter '{}' from {}", name, meta_path)
        except Exception as exc:
            logger.warning(
                "Could not register auto-discovered '{}': {}", name, exc
            )


_discover_and_register()
