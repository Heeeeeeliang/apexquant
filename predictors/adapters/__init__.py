"""
Adapter layer — loads pre-trained model weights into the Predictor interface.

Each adapter wraps a trained model file from ``models/`` and exposes it
through the :class:`~predictors.base.Predictor` interface so that the
:class:`~predictors.registry.PredictorRegistry` can execute them via
``run_all``.

Adapters:

- :class:`VolAdapter` — LightGBM regressor for volatility (sigmoid output).
- :class:`MetaAdapter` — LightGBM classifier for meta-label filtering.
- :class:`CnnAdapter` — Dual-branch MultiScaleCNN for turning-point detection.

Imports are lazy so that a broken torch installation does not prevent
the LightGBM-based adapters from loading.
"""

__all__ = ["VolAdapter", "MetaAdapter", "CnnAdapter"]


def __getattr__(name: str):
    """Lazy import so a broken torch does not block vol/meta adapters."""
    if name == "VolAdapter":
        from predictors.adapters.vol_adapter import VolAdapter
        return VolAdapter
    if name == "MetaAdapter":
        from predictors.adapters.meta_adapter import MetaAdapter
        return MetaAdapter
    if name == "CnnAdapter":
        from predictors.adapters.cnn_adapter import CnnAdapter
        return CnnAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
