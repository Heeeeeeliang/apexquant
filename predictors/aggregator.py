"""
Learned signal aggregator for ApexQuant.

**Status: Future Work hook — not used in the live system.**

The live three-layer cascade is currently hardcoded in
``strategies.builtin.ai_strategy.AIStrategy.on_bar()`` as sequential
threshold checks on vol_prob, tp_top, and tp_bottom predictions.
This module provides the *learned* replacement that would combine
predictor outputs via a fitted model instead of fixed thresholds.

Supports two model backends:

- ``"lightgbm"`` — :class:`LGBMClassifier` (default, captures non-linear
  interactions between predictor signals)
- ``"logistic"`` — :class:`LogisticRegression` (interpretable linear baseline)

The aggregator is trained **only** on the validation set.  The test set
is touched exactly once for final evaluation.

Usage::

    from predictors.aggregator import LearnedAggregator
    from predictors.factor_layer import FactorLayer
    from config.default import CONFIG

    agg = LearnedAggregator(CONFIG)
    fl  = FactorLayer(labels=["vol_prob", "tp_score", "meta_prob"])

    # Fit on validation data
    agg.fit(val_bars, val_returns, fl)

    # Tune decision threshold
    threshold = agg.fit_threshold(val_bars, fl, metric="sharpe")

    # Inference — single bar
    signal = agg.aggregate(predictions, fl)
    print(signal)  # AggregatedSignal(BULL, dir=+0.62, str=0.62, ...)

    # Persist / restore
    agg.save("results/runs/aggregator.joblib")
    agg.load("results/runs/aggregator.joblib")
"""

__all__ = ["LearnedAggregator"]

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from data.bar import Bar
from predictors.factor_layer import FactorLayer
from predictors.result import AggregatedSignal, PredictionResult

# Minimum trades required for a threshold candidate to be valid
_MIN_TRADES: int = 50


class LearnedAggregator:
    """Learns optimal signal weights from validation data.

    Trains a classifier on normalised predictor outputs (via
    :class:`FactorLayer`) to predict whether a bar's forward return
    is positive.  At inference time, the predicted probability is
    mapped to an :class:`AggregatedSignal` with direction, strength,
    and confidence.

    Attributes:
        model_type: ``"lightgbm"`` or ``"logistic"``.
        model: The fitted sklearn / LightGBM classifier.
        is_fitted: Whether :meth:`fit` has been called.
        feature_names: Label order used during training.
        threshold: Decision threshold (tuned by :meth:`fit_threshold`).
    """

    _SUPPORTED_TYPES = {"lightgbm", "logistic"}

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the aggregator from CONFIG.

        Args:
            config: Full CONFIG dict.  Reads
                ``config["aggregator"]["model_type"]``.

        Raises:
            ValueError: If the configured model_type is not supported.
        """
        self.model_type: str = (
            config.get("aggregator", {}).get("model_type", "lightgbm")
        )
        if self.model_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported aggregator model_type '{self.model_type}'. "
                f"Valid: {sorted(self._SUPPORTED_TYPES)}"
            )

        self.model: Any = self._build_model()
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self.threshold: float = 0.50

        logger.info("LearnedAggregator initialised (model_type={})", self.model_type)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        val_bars: list[Bar],
        val_returns: list[float],
        factor_layer: FactorLayer,
    ) -> None:
        """Train the aggregator on the validation set.

        Args:
            val_bars: Validation bars with ``predictions`` dicts already
                populated by all upstream predictors.
            val_returns: Forward returns for each bar (same length as
                *val_bars*).  Positive means the trade was profitable.
            factor_layer: Shared :class:`FactorLayer` used to normalise
                predictor outputs into feature vectors.

        Raises:
            ValueError: If *val_bars* and *val_returns* have different lengths.
        """
        if len(val_bars) != len(val_returns):
            raise ValueError(
                f"Length mismatch: val_bars={len(val_bars)}, "
                f"val_returns={len(val_returns)}"
            )

        # --- Build feature matrix ---
        X = self._bars_to_features(val_bars, factor_layer)
        y = np.array(
            [1 if r > 0 else 0 for r in val_returns], dtype=np.int32
        )

        logger.info(
            "Fitting aggregator on {} samples ({} positive, {} negative)",
            len(y),
            int(y.sum()),
            int((y == 0).sum()),
        )

        self.model.fit(X, y)
        self.feature_names = factor_layer.get_feature_names()
        self.is_fitted = True

        # --- Log feature importances (LightGBM only) ---
        if self.model_type == "lightgbm":
            importances = self.model.feature_importances_
            for name, imp in sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            ):
                logger.info("  Feature importance: {} = {}", name, imp)

        logger.info("Aggregator fitting complete (model_type={})", self.model_type)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def aggregate(
        self,
        predictions: dict[str, PredictionResult],
        factor_layer: FactorLayer,
    ) -> AggregatedSignal:
        """Produce an aggregated signal from predictor outputs.

        If the aggregator has not been fitted, falls back to a simple
        average of non-stale prediction probabilities.

        Args:
            predictions: Dict mapping label to :class:`PredictionResult`.
            factor_layer: Shared :class:`FactorLayer` for normalisation.

        Returns:
            :class:`AggregatedSignal` with direction, strength, and
            confidence derived from the model's predicted probability.
        """
        contributing = [
            label for label, r in predictions.items() if not r.is_stale
        ]

        if not self.is_fitted:
            logger.debug("Aggregator not fitted, using simple average fallback")
            prob = self._simple_average_fallback(predictions)
        else:
            features = factor_layer.normalize(predictions)
            prob = self._predict_proba(features)

        direction = (prob - 0.5) * 2.0
        strength = abs(prob - 0.5) * 2.0
        confidence = max(prob, 1.0 - prob)

        signal = AggregatedSignal(
            direction=float(np.clip(direction, -1.0, 1.0)),
            strength=float(np.clip(strength, 0.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            contributing_predictors=contributing,
        )

        logger.debug(
            "Aggregated: prob={:.4f} -> dir={:+.3f}, str={:.3f}, "
            "conf={:.3f}, contributors={}",
            prob,
            signal.direction,
            signal.strength,
            signal.confidence,
            contributing,
        )
        return signal

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def fit_threshold(
        self,
        val_bars: list[Bar],
        factor_layer: FactorLayer,
        val_returns: list[float] | None = None,
        metric: str = "sharpe",
    ) -> float:
        """Find the optimal decision threshold on the validation set.

        Sweeps thresholds from 0.40 to 0.70 (inclusive, step 0.01) and
        selects the one that maximises the chosen metric.  A threshold
        candidate is only valid if it produces at least
        :data:`_MIN_TRADES` trades.

        Args:
            val_bars: Validation bars with predictions populated.
            factor_layer: Shared :class:`FactorLayer`.
            val_returns: Forward returns (same length as *val_bars*).
                If ``None``, computed from bar close prices.
            metric: ``"sharpe"`` or ``"win_rate"``.

        Returns:
            The optimal threshold value.  Also stored in
            ``self.threshold``.

        Raises:
            ValueError: If *metric* is not ``"sharpe"`` or ``"win_rate"``.
            RuntimeError: If the aggregator has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot fit threshold before fit() — call fit() first"
            )
        if metric not in ("sharpe", "win_rate"):
            raise ValueError(
                f"Unknown metric '{metric}'. Valid: 'sharpe', 'win_rate'"
            )

        # --- Compute probabilities for every validation bar ---
        X = self._bars_to_features(val_bars, factor_layer)
        probs = self._predict_proba_batch(X)

        # --- Resolve forward returns ---
        if val_returns is not None:
            returns = np.asarray(val_returns, dtype=np.float64)
        else:
            closes = np.array([b.close for b in val_bars], dtype=np.float64)
            returns = np.append(np.diff(closes) / closes[:-1], 0.0)

        # --- Sweep thresholds ---
        best_score = -np.inf
        best_threshold = 0.50
        thresholds = np.arange(0.40, 0.71, 0.01)

        for t in thresholds:
            trade_mask = probs >= t
            n_trades = int(trade_mask.sum())

            if n_trades < _MIN_TRADES:
                continue

            trade_returns = returns[trade_mask]

            if metric == "sharpe":
                score = self._sharpe(trade_returns)
            else:  # win_rate
                score = float((trade_returns > 0).sum()) / n_trades

            logger.debug(
                "Threshold {:.2f}: trades={}, {}={:.4f}",
                t,
                n_trades,
                metric,
                score,
            )

            if score > best_score:
                best_score = score
                best_threshold = round(float(t), 2)

        self.threshold = best_threshold
        logger.info(
            "Optimal threshold: {:.2f} ({}={:.4f})",
            best_threshold,
            metric,
            best_score,
        )
        return best_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> Path:
        """Persist the fitted aggregator to disk via joblib.

        Args:
            path: Destination file path.

        Returns:
            Resolved path of the written file.
        """
        import joblib

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_type": self.model_type,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
        }
        joblib.dump(state, file_path)
        logger.info("Aggregator saved to {}", file_path)
        return file_path.resolve()

    def load(self, path: str) -> None:
        """Load a previously saved aggregator from disk.

        Args:
            path: Path to a joblib file written by :meth:`save`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        import joblib

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Aggregator file not found: {file_path}")

        state: dict[str, Any] = joblib.load(file_path)
        self.model_type = state["model_type"]
        self.model = state["model"]
        self.is_fitted = state["is_fitted"]
        self.feature_names = state["feature_names"]
        self.threshold = state["threshold"]

        logger.info(
            "Aggregator loaded from {} (model_type={}, threshold={:.2f})",
            file_path,
            self.model_type,
            self.threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Construct the underlying classifier.

        Returns:
            An unfitted sklearn-compatible classifier.
        """
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
        else:  # logistic
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def _bars_to_features(
        self, bars: list[Bar], factor_layer: FactorLayer
    ) -> np.ndarray:
        """Convert a list of bars into a feature matrix.

        Each bar's ``predictions`` dict is converted to a
        :class:`PredictionResult` dict (wrapping raw floats if needed),
        then normalised via the factor layer.

        Args:
            bars: List of bars with predictions populated.
            factor_layer: Shared normaliser.

        Returns:
            2-D numpy array of shape ``(n_bars, n_features)``.
        """
        rows: list[np.ndarray] = []
        labels = factor_layer.get_feature_names()

        for bar in bars:
            pred_dict = self._bar_predictions_to_results(bar, labels)
            features = factor_layer.normalize(pred_dict)
            rows.append(features)

        return np.vstack(rows)

    @staticmethod
    def _bar_predictions_to_results(
        bar: Bar, labels: list[str]
    ) -> dict[str, PredictionResult]:
        """Convert a bar's predictions dict to PredictionResult objects.

        Bar.predictions may contain raw floats (legacy) or
        PredictionResult instances (new interface).  This method
        normalises both into PredictionResult.

        Args:
            bar: A bar with predictions populated.
            labels: Expected label names.

        Returns:
            Dict mapping label to :class:`PredictionResult`.
        """
        from datetime import datetime, timedelta

        result: dict[str, PredictionResult] = {}
        now = datetime.now()
        valid_until = now + timedelta(hours=1)

        for label in labels:
            value = bar.predictions.get(label)
            if value is None:
                continue

            if isinstance(value, PredictionResult):
                result[label] = value
            else:
                # Legacy: raw float in bar.predictions
                result[label] = PredictionResult(
                    label=label,
                    prob=float(value),
                    raw_score=float(value),
                    generated_at=now,
                    valid_until=valid_until,
                    model_version="legacy",
                )

        return result

    def _predict_proba(self, features: np.ndarray) -> float:
        """Predict positive-class probability for a single feature vector.

        Args:
            features: 1-D feature array.

        Returns:
            Probability in ``[0, 1]``.
        """
        X = features.reshape(1, -1)
        prob = float(self.model.predict_proba(X)[0, 1])
        return prob

    def _predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for a feature matrix.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of probabilities.
        """
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    @staticmethod
    def _simple_average_fallback(
        predictions: dict[str, PredictionResult],
    ) -> float:
        """Compute a simple average of non-stale prediction probabilities.

        Used as a fallback when the aggregator has not been fitted.

        Args:
            predictions: Current prediction results.

        Returns:
            Average probability, or 0.5 if no valid predictions.
        """
        valid_probs = [
            r.prob for r in predictions.values() if not r.is_stale
        ]
        if not valid_probs:
            return 0.5
        return float(np.mean(valid_probs))

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Compute a simplified Sharpe ratio (annualised, daily bars).

        Args:
            returns: Array of trade returns.

        Returns:
            Annualised Sharpe ratio, or 0.0 if std is near zero.
        """
        if len(returns) < 2:
            return 0.0
        mean = float(returns.mean())
        std = float(returns.std())
        if std < 1e-10:
            return 0.0
        return (mean / std) * np.sqrt(252)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "NOT fitted"
        return (
            f"LearnedAggregator(model_type={self.model_type!r}, "
            f"{status}, threshold={self.threshold:.2f})"
        )
