"""
Layer 3 — Meta-Label Trade Filter (LightGBM).

Takes the outputs of Layer 1 (volatility) and Layer 2 (turning point)
as additional features and predicts whether a candidate trade will be
profitable.  This is the "meta-labelling" approach from de Prado.

Thesis result: Win Rate = 64.5%.

Usage::

    from predictors.registry import REGISTRY
    ml = REGISTRY.get("meta_label")
    metrics = ml.train(train_bars, val_bars)  # bars need L1+L2 predictions
    bars = ml.predict(test_bars)
    print(bars[0].predictions["meta_label"])  # e.g. 0.71
"""

__all__ = ["MetaLabelPredictor"]

import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from data.bar import Bar
from predictors.base import BasePredictor
from predictors.registry import REGISTRY


@REGISTRY.register
class MetaLabelPredictor(BasePredictor):
    """LightGBM meta-label classifier for trade filtering.

    Attributes:
        name: ``"meta_label"``.
        probability_threshold: Minimum P(win) to pass a trade.
    """

    name: str = "meta_label"

    def __init__(self) -> None:
        super().__init__()
        self.probability_threshold: float = self.config.get("probability_threshold", 0.55)

    def train(self, train_bars: list[Bar], val_bars: list[Bar]) -> dict[str, float]:
        """Train the meta-label classifier.

        Requires that bars already have predictions from Layer 1 and Layer 2.

        Args:
            train_bars: Training bars with L1/L2 predictions and features.
            val_bars: Validation bars for threshold tuning.

        Returns:
            Dict with ``train_acc``, ``val_acc``, ``val_precision``, ``val_auc``,
            ``probability_threshold``.
        """
        X_train, y_train = self._prepare_xy(train_bars)
        X_val, y_val = self._prepare_xy(val_bars)

        logger.info(
            "Training meta-label predictor: {} train, {} val samples",
            len(X_train),
            len(X_val),
        )

        params = {
            "n_estimators": self.config.get("n_estimators", 300),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 4),
            "num_leaves": self.config.get("num_leaves", 15),
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
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        # Tune threshold on validation set
        val_prob = self.model.predict_proba(X_val)[:, 1]
        self._tune_threshold(val_prob, y_val)

        train_pred = self.model.predict(X_train)
        val_pred = (val_prob >= self.probability_threshold).astype(int)

        metrics = {
            "train_acc": float(accuracy_score(y_train, train_pred)),
            "val_acc": float(accuracy_score(y_val, val_pred)),
            "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "val_auc": float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.0,
            "probability_threshold": self.probability_threshold,
        }
        logger.info("Meta-label training complete: {}", metrics)
        return metrics

    def predict(self, bars: list[Bar]) -> list[Bar]:
        """Predict trade-win probability for each bar.

        Args:
            bars: Bars with features and L1/L2 predictions.

        Returns:
            Bars with ``predictions["meta_label"]`` set to P(win).
        """
        X, _ = self._prepare_xy(bars)
        probs = self.model.predict_proba(X)[:, 1]

        for bar, prob in zip(bars, probs):
            bar.predictions["meta_label"] = float(prob)
            logger.debug("Bar {} meta_prob={:.3f}", bar.timestamp.date(), prob)

        logger.info("Meta-label prediction complete for {} bars", len(bars))
        return bars

    def _prepare_xy(self, bars: list[Bar]) -> tuple[pd.DataFrame, np.ndarray]:
        """Build feature matrix including L1/L2 predictions and trade outcome labels.

        Args:
            bars: Input bars.

        Returns:
            Tuple of (feature_matrix, labels).
        """
        rows: list[dict[str, float]] = []
        for b in bars:
            row = dict(b.features)
            # Append upstream predictor outputs as features
            row["pred_volatility"] = b.predictions.get("volatility", 0.0)
            row["pred_turning_point"] = b.predictions.get("turning_point", 0.0)
            rows.append(row)

        X = pd.DataFrame(rows)

        # Label: was the next-bar return positive? (1 = profitable trade)
        closes = np.array([b.close for b in bars])
        forward_returns = np.append(np.diff(closes) / closes[:-1], 0.0)
        y = (forward_returns > 0).astype(int)

        return X, y

    def _tune_threshold(self, probs: np.ndarray, y: np.ndarray) -> None:
        """Tune probability threshold on validation set to maximise precision
        while maintaining reasonable recall.

        Args:
            probs: Predicted probabilities.
            y: True labels.
        """
        best_score = 0.0
        best_t = 0.55

        for t in np.arange(0.45, 0.75, 0.01):
            preds = (probs >= t).astype(int)
            if preds.sum() == 0:
                continue
            prec = float(precision_score(y, preds, zero_division=0))
            # Require at least 20% of bars to have signals
            coverage = preds.mean()
            if coverage < 0.1:
                continue
            score = prec * np.sqrt(coverage)  # Balance precision and coverage
            if score > best_score:
                best_score = score
                best_t = float(t)

        self.probability_threshold = best_t
        logger.info(
            "Tuned meta-label threshold: {:.2f} (score={:.3f})",
            best_t,
            best_score,
        )
