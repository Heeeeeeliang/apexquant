"""
Abstract base class for all ApexQuant predictors.

Defines two layers of abstraction:

- :class:`Predictor` — the **new** per-bar prediction interface used by
  the aggregator.  Each subclass overrides ``predict(bar, context)`` to
  return a :class:`~predictors.result.PredictionResult`.
- :class:`BasePredictor` — the **legacy** batch interface retained for
  backward compatibility with the existing ``p01``/``p02``/``p03``
  predictor implementations.

New predictors should subclass :class:`Predictor`.

Usage::

    from predictors.base import Predictor, Context
    from predictors.result import PredictionResult

    class MyPredictor(Predictor):
        name = "my_pred"
        output_label = "my_score"
        update_freq = "bar"
        default_validity = 3600

        def predict(self, bar, context):
            score = self._model.infer(bar.features)
            return PredictionResult(
                label=self.output_label,
                prob=score,
                raw_score=score,
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(seconds=self.default_validity),
                model_version=self.get_version(),
            )
"""

__all__ = ["Predictor", "BasePredictor", "Context"]

import abc
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from config.default import get
from data.bar import Bar
from predictors.result import PredictionResult


# ---------------------------------------------------------------------------
# Context — ambient state available to every predict() call
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Ambient runtime context passed to every predictor.

    Carries recent bar history, current predictions from upstream
    layers, and metadata so that each predictor can make stateful
    decisions without maintaining its own buffers.

    Attributes:
        current_time: Wall-clock time of the prediction request.
        history: Recent bars in chronological order (newest last).
        predictions: Upstream predictor results keyed by label.
        regime: Current detected market-regime string.
        meta: Arbitrary key-value metadata.
    """

    current_time: datetime = field(default_factory=datetime.now)
    history: list[Bar] = field(default_factory=list)
    predictions: dict[str, PredictionResult] = field(default_factory=dict)
    regime: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predictor — new per-bar interface
# ---------------------------------------------------------------------------

class Predictor(abc.ABC):
    """Abstract per-bar predictor that all new layers must subclass.

    Class attributes (must be overridden by subclasses):

    - ``name``              — unique predictor identifier
    - ``output_label``      — key written into ``bar.predictions``
    - ``update_freq``       — ``"bar"`` | ``"1h"`` | ``"6h"`` | ``"1d"`` | ``"on_event"``
    - ``default_validity``  — seconds a prediction stays valid

    Attributes:
        _model: The underlying trained model object (loaded via :meth:`load`).
    """

    # -- class attributes (override in subclasses) --------------------------
    name: str = ""
    output_label: str = ""
    update_freq: str = "bar"
    default_validity: int = 3600

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty 'name' class attribute"
            )
        if not self.output_label:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty 'output_label' class attribute"
            )
        self._model: Any = None
        logger.info(
            "Initialised predictor '{}' (label={}, freq={}, validity={}s)",
            self.name,
            self.output_label,
            self.update_freq,
            self.default_validity,
        )

    # -- abstract methods ---------------------------------------------------

    @abc.abstractmethod
    def predict(self, bar: Bar, context: Context) -> PredictionResult:
        """Produce a prediction for a single bar.

        Args:
            bar: The current OHLCV bar with features populated.
            context: Ambient runtime context (history, upstream preds, etc.).

        Returns:
            A :class:`~predictors.result.PredictionResult` for this bar.
        """

    # -- concrete methods ---------------------------------------------------

    def get_version(self) -> str:
        """Return a version string for this predictor.

        The default implementation returns ``"{name}_v1"``.  Override
        this when the model artefact changes to enable degradation
        detection across runs.

        Returns:
            Version identifier string.
        """
        return f"{self.name}_v1"

    def is_ready(self) -> bool:
        """Check whether the predictor is ready to serve predictions.

        The default implementation returns ``True``.  Override to
        verify that model files exist on disk or that required
        resources are available.

        Returns:
            ``True`` if the predictor can serve predictions.
        """
        return True

    def load(self) -> None:
        """Load model artefacts from disk.

        The default implementation is a no-op.  Override to
        deserialise model weights, calibration parameters, etc.
        """
        logger.debug("Predictor '{}' load() — no-op (override in subclass)", self.name)

    def _make_result(
        self,
        prob: float,
        raw_score: float = 0.0,
        confidence: float = 1.0,
        feature_hash: str = "",
        validity_seconds: int | None = None,
    ) -> PredictionResult:
        """Convenience factory for building a :class:`PredictionResult`.

        Fills in ``label``, ``model_version``, and timing fields
        automatically so that subclasses can focus on the score.

        Args:
            prob: Calibrated probability ``[0, 1]``.
            raw_score: Uncalibrated model output.
            confidence: Model certainty ``[0, 1]``.
            feature_hash: Hash of the input features.
            validity_seconds: Override for ``default_validity``.

        Returns:
            Fully populated :class:`PredictionResult`.
        """
        now = datetime.now()
        ttl = validity_seconds if validity_seconds is not None else self.default_validity
        return PredictionResult(
            label=self.output_label,
            prob=prob,
            confidence=confidence,
            raw_score=raw_score,
            generated_at=now,
            valid_until=now + timedelta(seconds=ttl),
            model_version=self.get_version(),
            feature_hash=feature_hash,
        )

    def __repr__(self) -> str:
        ready = "ready" if self.is_ready() else "NOT ready"
        return f"{type(self).__name__}(name={self.name!r}, {ready})"


# ---------------------------------------------------------------------------
# BasePredictor — legacy batch interface (backward compat)
# ---------------------------------------------------------------------------

class BasePredictor(abc.ABC):
    """Legacy batch-prediction interface.

    Retained for backward compatibility with the existing
    ``p01_volatility``, ``p02_turning_point``, and ``p03_meta_label``
    implementations.  New predictors should subclass :class:`Predictor`
    instead.

    Attributes:
        name: Unique predictor identifier (used as key in Bar.predictions).
        config: Predictor-specific config sub-dict.
        model: The underlying trained model object.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.config: dict[str, Any] = get(f"predictors.{self.name}", {})
        self.model: Any = None
        logger.info("Initialised legacy predictor: {}", self.name)

    @abc.abstractmethod
    def train(self, train_bars: list[Bar], val_bars: list[Bar]) -> dict[str, float]:
        """Train the predictor on training data, tune on validation data.

        Args:
            train_bars: Training set (chronological).
            val_bars: Validation set (chronological, for threshold tuning).

        Returns:
            Dict of training metrics.
        """

    @abc.abstractmethod
    def predict(self, bars: list[Bar]) -> list[Bar]:
        """Run inference and write results into ``bar.predictions[self.name]``.

        Args:
            bars: Bars to predict on (mutated in-place).

        Returns:
            The same list of bars with predictions attached.
        """

    def save(self, path: Path | None = None) -> Path:
        """Persist the trained model to disk.

        Args:
            path: Optional override directory. Defaults to
                ``results_dir/predictors/``.

        Returns:
            The path the model was saved to.
        """
        if path is None:
            path = Path(get("system.results_dir", "results/runs")) / "predictors"
        path.mkdir(parents=True, exist_ok=True)
        save_path = path / f"{self.name}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)

        logger.info("Saved {} model to {}", self.name, save_path)
        return save_path

    def load(self, path: Path | None = None) -> None:
        """Load a previously saved model from disk.

        Args:
            path: Path to the pickle file. Defaults to standard location.
        """
        if path is None:
            path = (
                Path(get("system.results_dir", "results/runs"))
                / "predictors"
                / f"{self.name}.pkl"
            )

        with open(path, "rb") as f:
            self.model = pickle.load(f)

        logger.info("Loaded {} model from {}", self.name, path)
