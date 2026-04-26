"""
Probability calibration for ApexQuant predictors.

**Status: Future Work hook — not called in the live system.**

Maps raw model outputs to well-calibrated probabilities using either
**Platt scaling** (logistic regression) or **isotonic regression**.
Calibrators are fitted per predictor label on the validation set and
then applied at inference time.

Usage::

    from predictors.calibrator import ProbabilityCalibrator
    import numpy as np

    cal = ProbabilityCalibrator()

    # Fit on validation predictions
    cal.fit("vol_prob", preds=np.array([0.2, 0.8, 0.6]), labels=np.array([0, 1, 1]))

    # Apply at inference time
    calibrated = cal.calibrate("vol_prob", raw_score=0.73)  # e.g. 0.81

    # Convenience: fit all labels from validation set at once
    cal.fit_from_val_set(
        val_predictions={"vol_prob": [0.2, 0.8], "tp_score": [0.1, 0.9]},
        val_labels=[0, 1],
    )

    # Persist / restore
    cal.save("results/runs/calibrators.joblib")
    cal.load("results/runs/calibrators.joblib")
"""

__all__ = ["ProbabilityCalibrator"]

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Maps raw predictor scores to calibrated probabilities.

    Maintains one fitted calibrator per predictor label.  Supports two
    calibration strategies:

    - ``"platt"`` — Platt scaling via :class:`LogisticRegression`.
    - ``"isotonic"`` — monotone :class:`IsotonicRegression`.

    Attributes:
        _calibrators: Internal mapping from label to fitted calibrator.
        _methods: Internal mapping from label to method name.
    """

    _VALID_METHODS = {"platt", "isotonic"}

    def __init__(self) -> None:
        self._calibrators: dict[str, Any] = {}
        self._methods: dict[str, str] = {}
        logger.info("ProbabilityCalibrator initialised")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        label: str,
        preds: np.ndarray,
        labels: np.ndarray,
        method: str = "isotonic",
    ) -> None:
        """Fit a calibrator for a single predictor label.

        Args:
            label: Predictor output label (e.g. ``"vol_prob"``).
            preds: Raw model predictions, shape ``(n_samples,)``.
            labels: Binary ground-truth labels, shape ``(n_samples,)``.
            method: ``"platt"`` for Platt scaling (logistic regression)
                or ``"isotonic"`` for isotonic regression.

        Raises:
            ValueError: If *method* is not ``"platt"`` or ``"isotonic"``.
            ValueError: If *preds* and *labels* have mismatched lengths.
        """
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown calibration method '{method}'. "
                f"Valid: {sorted(self._VALID_METHODS)}"
            )
        if len(preds) != len(labels):
            raise ValueError(
                f"Length mismatch: preds={len(preds)}, labels={len(labels)}"
            )

        preds = np.asarray(preds, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        if method == "platt":
            calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            calibrator.fit(preds.reshape(-1, 1), labels)
        else:  # isotonic
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(preds, labels)

        self._calibrators[label] = calibrator
        self._methods[label] = method
        logger.info(
            "Fitted '{}' calibrator for '{}' on {} samples",
            method,
            label,
            len(preds),
        )

    def fit_from_val_set(
        self,
        val_predictions: dict[str, list[float]],
        val_labels: list[int],
        method: str = "isotonic",
    ) -> None:
        """Convenience: fit calibrators for all labels from validation data.

        Args:
            val_predictions: Dict mapping label name to list of raw
                predictions (one per validation sample).
            val_labels: Binary ground-truth labels shared across all
                predictors (same ordering).
            method: Calibration method applied to all labels.
        """
        labels_arr = np.asarray(val_labels, dtype=np.float64)

        for label, preds_list in val_predictions.items():
            preds_arr = np.asarray(preds_list, dtype=np.float64)
            if len(preds_arr) != len(labels_arr):
                logger.warning(
                    "Skipping '{}': {} predictions vs {} labels",
                    label,
                    len(preds_arr),
                    len(labels_arr),
                )
                continue
            self.fit(label, preds_arr, labels_arr, method=method)

        logger.info(
            "fit_from_val_set complete: {} labels calibrated", len(val_predictions)
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, label: str, raw_score: float) -> float:
        """Apply calibration to a raw model score.

        If no calibrator has been fitted for *label*, the raw score is
        returned unchanged.

        Args:
            label: Predictor output label.
            raw_score: Uncalibrated model output.

        Returns:
            Calibrated probability in ``[0, 1]``.
        """
        if label not in self._calibrators:
            logger.debug(
                "No calibrator for '{}', returning raw score {:.4f}",
                label,
                raw_score,
            )
            return raw_score

        calibrator = self._calibrators[label]
        method = self._methods[label]

        if method == "platt":
            prob = float(
                calibrator.predict_proba(
                    np.array([[raw_score]])
                )[0, 1]
            )
        else:  # isotonic
            prob = float(calibrator.predict(np.array([raw_score]))[0])

        prob = float(np.clip(prob, 0.0, 1.0))
        logger.debug(
            "Calibrated '{}': {:.4f} -> {:.4f} ({})",
            label,
            raw_score,
            prob,
            method,
        )
        return prob

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> Path:
        """Persist all fitted calibrators to disk via joblib.

        Args:
            path: Destination file path.

        Returns:
            Resolved path of the written file.
        """
        import joblib

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "calibrators": self._calibrators,
            "methods": self._methods,
        }
        joblib.dump(state, file_path)
        logger.info(
            "Saved {} calibrators to {}", len(self._calibrators), file_path
        )
        return file_path.resolve()

    def load(self, path: str) -> None:
        """Load previously saved calibrators from disk.

        Args:
            path: Path to a joblib file written by :meth:`save`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        import joblib

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Calibrator file not found: {file_path}")

        state: dict[str, Any] = joblib.load(file_path)
        self._calibrators = state["calibrators"]
        self._methods = state["methods"]
        logger.info(
            "Loaded {} calibrators from {}", len(self._calibrators), file_path
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def fitted_labels(self) -> list[str]:
        """Return labels for which a calibrator has been fitted.

        Returns:
            Sorted list of label strings.
        """
        return sorted(self._calibrators.keys())

    def __repr__(self) -> str:
        return (
            f"ProbabilityCalibrator(fitted={self.fitted_labels()})"
        )
