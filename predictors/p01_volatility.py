"""
Layer 1 — Volatility Prediction (LightGBM).

Predicts whether the next-bar realised volatility will exceed a
threshold derived from the training distribution.  The threshold
quantile is tuned on the validation set.

Thesis result: Directional Accuracy = 83.6%.

Usage::

    from predictors.registry import REGISTRY
    vol = REGISTRY.get("volatility")
    metrics = vol.train(train_bars, val_bars)
    bars = vol.predict(test_bars)
    print(bars[0].predictions["volatility"])  # e.g. 0.87
"""

__all__ = ["VolatilityPredictor"]

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score

from data.bar import Bar
from predictors.base import BasePredictor
from predictors.registry import REGISTRY


@REGISTRY.register
class VolatilityPredictor(BasePredictor):
    """LightGBM classifier for high/low volatility regimes.

    Attributes:
        name: ``"volatility"``.
        threshold: Realised-volatility level separating high/low regimes.
    """

    name: str = "volatility"

    def __init__(self) -> None:
        super().__init__()
        self.threshold: float = 0.0

    def train(self, train_bars: list[Bar], val_bars: list[Bar]) -> dict[str, float]:
        """Train the LightGBM volatility classifier.

        Args:
            train_bars: Training bars with features already computed.
            val_bars: Validation bars for threshold selection.

        Returns:
            Dict with ``train_acc``, ``val_acc``, ``val_auc``, and ``threshold``.
        """
        X_train, y_train = self._prepare_xy(train_bars, fit_threshold=True)
        X_val, y_val = self._prepare_xy(val_bars, fit_threshold=False)

        logger.info(
            "Training volatility predictor: {} train, {} val samples",
            len(X_train),
            len(X_val),
        )

        params = {
            "n_estimators": self.config.get("n_estimators", 500),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 6),
            "num_leaves": self.config.get("num_leaves", 31),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "random_state": 42,
            "verbose": -1,
        }

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        train_pred = self.model.predict(X_train)
        val_prob = self.model.predict_proba(X_val)[:, 1]
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_acc": float(accuracy_score(y_train, train_pred)),
            "val_acc": float(accuracy_score(y_val, val_pred)),
            "val_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.0,
            "threshold": self.threshold,
        }
        logger.info("Volatility training complete: {}", metrics)
        return metrics

    def predict(self, bars: list[Bar]) -> list[Bar]:
        """Predict volatility regime probability for each bar.

        Args:
            bars: Bars with features populated.

        Returns:
            Bars with ``predictions["volatility"]`` set to P(high_vol).
        """
        X, _ = self._prepare_xy(bars, fit_threshold=False)
        probs = self.model.predict_proba(X)[:, 1]

        for bar, prob in zip(bars, probs):
            bar.predictions["volatility"] = float(prob)
            logger.debug("Bar {} vol_prob={:.3f}", bar.timestamp.date(), prob)

        logger.info("Volatility prediction complete for {} bars", len(bars))
        return bars

    def _prepare_xy(
        self, bars: list[Bar], fit_threshold: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Extract feature matrix and binary volatility label.

        Args:
            bars: Input bars.
            fit_threshold: If True, compute threshold from these bars.

        Returns:
            Tuple of (feature_matrix, labels).
        """
        feature_dicts = [b.features for b in bars]
        X = pd.DataFrame(feature_dicts)

        # Target: forward realised vol (1-bar) exceeds threshold
        closes = np.array([b.close for b in bars])
        log_returns = np.log(closes[1:] / closes[:-1])
        # Pad first bar with 0
        forward_vol = np.abs(np.append(log_returns, 0.0))

        if fit_threshold:
            quantile = self.config.get("threshold_quantile", 0.67)
            self.threshold = float(np.quantile(forward_vol[:-1], quantile))
            logger.info("Volatility threshold set at {:.6f} (q={})", self.threshold, quantile)

        y = (forward_vol > self.threshold).astype(int)
        return X, y
