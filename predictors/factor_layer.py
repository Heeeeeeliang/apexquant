"""
Factor layer — normalises predictor outputs into a consistent feature vector.

**Status: Future Work hook — consumed only by LearnedAggregator (also
Future Work). Not called in the live system.**

Applies freshness-weighting, stale-prediction handling, and running
z-score standardisation so that downstream consumers (aggregator,
strategies) receive a stable, zero-mean, unit-variance input regardless
of individual predictor scale or timing.

Usage::

    from predictors.factor_layer import FactorLayer
    from predictors.result import PredictionResult
    from datetime import datetime, timedelta

    fl = FactorLayer(labels=["vol_prob", "tp_score", "meta_prob"])

    results = {
        "vol_prob": PredictionResult(
            label="vol_prob", prob=0.83, raw_score=1.47,
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=1),
            model_version="v1",
        ),
        "tp_score": PredictionResult(
            label="tp_score", prob=0.62, raw_score=0.9,
            generated_at=datetime.now(),
            valid_until=datetime.now() + timedelta(hours=1),
            model_version="v1",
        ),
    }

    vec = fl.normalize(results)   # np.ndarray of shape (3,)
    names = fl.get_feature_names()  # ["meta_prob", "tp_score", "vol_prob"]
"""

__all__ = ["FactorLayer"]

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from predictors.result import PredictionResult

# Neutral fill for stale / missing predictions (0.5 = maximum-entropy prior)
_NEUTRAL: float = 0.5


class FactorLayer:
    """Converts a dict of :class:`PredictionResult` into a normalised
    numpy feature vector.

    The processing pipeline for each label is:

    1. **Freshness weighting** — ``prob * freshness``, where freshness
       decays linearly from 1.0 to 0.0 over the prediction's validity
       window.
    2. **Stale fill** — if ``is_stale`` or the label is missing, the
       value is replaced with 0.5 (neutral).
    3. **Running standardisation** — Welford's online algorithm tracks
       per-label mean and variance; the final vector is z-scored.

    The label order is fixed (sorted alphabetically) so that the
    returned array is always in a deterministic order.

    Attributes:
        _labels: Sorted list of expected label names.
        _n: Number of observations seen (for running stats).
        _mean: Per-label running mean.
        _m2: Per-label running sum of squared deviations (Welford).
    """

    def __init__(self, labels: list[str]) -> None:
        """Initialise the factor layer.

        Args:
            labels: List of expected predictor output labels.  Order
                is determined by sorting; duplicates are removed.
        """
        self._labels: list[str] = sorted(set(labels))
        k = len(self._labels)
        self._n: int = 0
        self._mean: np.ndarray = np.full(k, _NEUTRAL, dtype=np.float64)
        self._m2: np.ndarray = np.zeros(k, dtype=np.float64)
        logger.info("FactorLayer initialised with {} labels: {}", k, self._labels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, results: dict[str, PredictionResult]) -> np.ndarray:
        """Build a normalised feature vector from prediction results.

        Args:
            results: Dict mapping label to :class:`PredictionResult`.
                Labels not present in *results* or whose result is stale
                are filled with 0.5 (neutral).

        Returns:
            1-D numpy array of shape ``(n_labels,)``, z-scored using
            running statistics.
        """
        raw = self._extract_weighted(results)
        self.update_stats_from_raw(raw)
        normalised = self._standardise(raw)
        logger.debug(
            "Normalised {} features: raw_range=[{:.3f}, {:.3f}], "
            "norm_range=[{:.3f}, {:.3f}]",
            len(normalised),
            float(raw.min()),
            float(raw.max()),
            float(normalised.min()),
            float(normalised.max()),
        )
        return normalised

    def get_feature_names(self) -> list[str]:
        """Return labels in the same order as :meth:`normalize` output.

        Returns:
            Sorted list of label strings.
        """
        return list(self._labels)

    def update_stats(self, results: dict[str, PredictionResult]) -> None:
        """Update running mean/std with a new observation.

        This is called automatically by :meth:`normalize`.  Call it
        manually if you want to update statistics without producing a
        normalised vector (e.g. during a warm-up phase).

        Args:
            results: Dict mapping label to :class:`PredictionResult`.
        """
        raw = self._extract_weighted(results)
        self.update_stats_from_raw(raw)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> Path:
        """Persist normaliser statistics to disk.

        Args:
            path: Destination ``.npz`` file path.

        Returns:
            Resolved path of the written file.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            file_path,
            labels=np.array(self._labels),
            n=np.array([self._n]),
            mean=self._mean,
            m2=self._m2,
        )
        logger.info("FactorLayer stats saved to {} (n={})", file_path, self._n)
        return file_path.resolve()

    def load(self, path: str) -> None:
        """Load normaliser statistics from disk.

        Args:
            path: Path to an ``.npz`` file written by :meth:`save`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the saved labels do not match the current ones.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"FactorLayer stats not found: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        saved_labels = list(data["labels"])

        if saved_labels != self._labels:
            raise ValueError(
                f"Label mismatch: saved={saved_labels}, current={self._labels}"
            )

        self._n = int(data["n"][0])
        self._mean = data["mean"].astype(np.float64)
        self._m2 = data["m2"].astype(np.float64)
        logger.info(
            "FactorLayer stats loaded from {} (n={})", file_path, self._n
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_weighted(
        self, results: dict[str, PredictionResult]
    ) -> np.ndarray:
        """Extract freshness-weighted probabilities, filling stale/missing
        labels with neutral (0.5).

        Args:
            results: Current prediction results.

        Returns:
            Raw (pre-standardisation) array of shape ``(n_labels,)``.
        """
        raw = np.full(len(self._labels), _NEUTRAL, dtype=np.float64)

        for i, label in enumerate(self._labels):
            if label not in results:
                logger.debug("Label '{}' missing, using neutral {}", label, _NEUTRAL)
                continue

            pred = results[label]

            if pred.is_stale:
                logger.debug(
                    "Label '{}' is stale (age={:.0f}s), using neutral {}",
                    label,
                    pred.age_seconds,
                    _NEUTRAL,
                )
                continue

            raw[i] = pred.prob * pred.freshness

        return raw

    def update_stats_from_raw(self, raw: np.ndarray) -> None:
        """Update running mean/variance using Welford's online algorithm.

        Args:
            raw: Pre-standardisation array from :meth:`_extract_weighted`.
        """
        self._n += 1
        delta = raw - self._mean
        self._mean += delta / self._n
        delta2 = raw - self._mean
        self._m2 += delta * delta2

    def _standardise(self, raw: np.ndarray) -> np.ndarray:
        """Apply z-score standardisation using running statistics.

        If fewer than 2 observations have been seen, the raw values
        are returned (centred by mean but not scaled) to avoid division
        by zero.

        Args:
            raw: Pre-standardisation array.

        Returns:
            Standardised array.
        """
        if self._n < 2:
            return raw - self._mean

        variance = self._m2 / (self._n - 1)
        std = np.sqrt(np.maximum(variance, 1e-10))
        return (raw - self._mean) / std

    def __repr__(self) -> str:
        return (
            f"FactorLayer(labels={self._labels}, observations={self._n})"
        )
