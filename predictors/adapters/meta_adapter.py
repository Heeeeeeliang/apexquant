"""
Meta-label adapter — wraps a pre-trained LightGBM *classifier* for Layer 3.

The model outputs ``predict_proba(X)[:, 1]`` directly as the
probability that a proposed trade will be profitable.

Weight layout::

    models/layer3/trade_filter/lgb_bottom_v1/
        weights.joblib   — sklearn-style LGBMClassifier
        meta.json        — {"n_features": 70, "threshold": 0.45, ...}
"""

__all__ = ["MetaAdapter"]

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from config.default import get
from data.bar import Bar
from predictors.base import Context, Predictor
from predictors.result import PredictionResult

# Streamlit-aware model cache: avoids reloading weights on page reruns.
try:
    import streamlit as _st_cache

    @_st_cache.cache_resource(show_spinner=False)
    def _cached_load_joblib(path_str: str, _mtime: float):
        import joblib
        return joblib.load(path_str)
except Exception:
    def _cached_load_joblib(path_str: str, _mtime: float):
        import joblib
        return joblib.load(path_str)


def _align_features(
    features: list[float], expected: int | None, name: str
) -> list[float]:
    """Pad or truncate a feature vector to match the model's expected size."""
    if expected is None or len(features) == expected:
        return features
    if not hasattr(_align_features, "_warned"):
        _align_features._warned = set()
    key = (name, len(features), expected)
    if key not in _align_features._warned:
        _align_features._warned.add(key)
        logger.warning(
            "{}: feature vector has {} elements but model expects {}; {}",
            name,
            len(features),
            expected,
            "padding with zeros" if len(features) < expected else "truncating",
        )
    if len(features) < expected:
        return features + [0.0] * (expected - len(features))
    return features[:expected]


class MetaAdapter(Predictor):
    """LightGBM classifier adapter for meta-label trade filtering.

    Attributes:
        name: Registry key (e.g. ``"meta_lgb_bottom"``).
        output_label: Label written to ``bar.predictions``.
        _model_dir: Path to the weight directory.
        _meta: Metadata loaded from ``meta.json``.
    """

    update_freq: str = "bar"
    default_validity: int = 3600

    def __init__(
        self,
        name: str,
        output_label: str,
        model_dir: str | Path,
    ) -> None:
        self.name = name
        self.output_label = output_label
        self._model_dir = Path(model_dir)
        self._meta: dict[str, Any] = {}
        super().__init__()

    def load(self) -> None:
        """Load LGBMClassifier from ``weights.joblib`` and metadata."""
        weights_path = self._model_dir / "weights.joblib"
        meta_path = self._model_dir / "meta.json"

        self._model = _cached_load_joblib(
            str(weights_path), weights_path.stat().st_mtime
        )
        logger.info(
            "Loaded meta model from {}: type={}, n_features_in_={}, "
            "has predict_proba={}",
            weights_path,
            type(self._model).__name__,
            getattr(self._model, "n_features_in_", "N/A"),
            hasattr(self._model, "predict_proba"),
        )

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            logger.info("Meta meta: {}", self._meta)

    def is_ready(self) -> bool:
        """Check that weight file exists on disk."""
        return (self._model_dir / "weights.joblib").exists()

    def get_version(self) -> str:
        return self._meta.get("version", f"{self.name}_v1")

    def predict(self, bar: Bar, context: Context) -> PredictionResult:
        """Run inference on a single bar.

        Builds a feature vector that includes upstream predictor outputs
        (vol_prob, tp_score) concatenated with the bar's own features,
        then calls ``model.predict_proba``.
        """
        if self._model is None:
            self.load()

        features = self._extract_features(bar, context)
        expected = getattr(self._model, "n_features_", None) or getattr(
            self._model, "n_features_in_", None
        )
        features = _align_features(features, expected, self.name)
        X = np.array([features])
        prob = float(self._model.predict_proba(X)[:, 1][0])

        return self._make_result(
            prob=prob,
            raw_score=prob,
            confidence=abs(prob - 0.5) * 2.0,
        )

    def _extract_features(self, bar: Bar, context: Context) -> list[float]:
        """Build feature vector with upstream predictions prepended."""
        upstream: list[float] = []
        for label in ["vol_prob", "vol_prob_flat", "tp_bottom", "tp_top"]:
            pred = context.predictions.get(label)
            if pred is not None:
                upstream.append(pred.prob if hasattr(pred, "prob") else float(pred))
            else:
                upstream.append(bar.get_prob(label, 0.5))

        if bar.features:
            base = list(bar.features.values())
        else:
            base = [bar.open, bar.high, bar.low, bar.close, bar.volume]
            for attr in [
                "ema_8", "ema_21", "ema_50", "rsi_14", "macd",
                "macd_signal", "macd_hist", "atr_14", "bb_upper",
                "bb_lower", "bb_mid", "volume_ratio", "vwap", "obv",
                "adx_14", "stoch_k", "stoch_d", "willr_14", "cci_20", "mfi_14",
            ]:
                v = getattr(bar, attr, None)
                base.append(v if v is not None else 0.0)

        return upstream + base
