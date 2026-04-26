"""
Core prediction result types for ApexQuant.

Defines two data carriers that flow between predictors, the
aggregator, and strategies:

- :class:`PredictionResult` — output of a single predictor for one bar.
- :class:`AggregatedSignal` — combined output of the aggregator layer.

Both are immutable-by-convention dataclasses with rich properties
for staleness detection, freshness decay, and serialisation.

Usage::

    from predictors.result import PredictionResult, AggregatedSignal
    from datetime import datetime, timedelta

    pred = PredictionResult(
        label="vol_prob",
        prob=0.83,
        confidence=0.91,
        raw_score=1.47,
        generated_at=datetime.now(),
        valid_until=datetime.now() + timedelta(hours=1),
        model_version="vol_lgb_v2",
    )
    print(pred.is_stale)       # False  (just created)
    print(pred.freshness)      # ~1.0   (brand new)
    print(pred.age_seconds)    # ~0.0

    sig = AggregatedSignal(
        direction=0.72,
        strength=0.85,
        confidence=0.90,
        contributing_predictors=["vol_lgb", "cnn_turning", "meta_lgb"],
        regime="trending_up",
    )
    print(sig.is_bullish)      # True
    print(sig.is_actionable)   # True
"""

__all__ = ["PredictionResult", "AggregatedSignal"]

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Output of a single predictor for one bar.

    Carries the calibrated probability, raw model score, timing
    metadata for staleness detection, and a feature hash for
    reproducibility audits.

    Attributes:
        label: Signal identifier, e.g. ``"vol_prob"``, ``"tp_score"``.
        prob: Calibrated probability in ``[0, 1]``.
        confidence: Model certainty in ``[0, 1]``.
        raw_score: Uncalibrated raw model output.
        generated_at: Timestamp when this prediction was created.
        valid_until: Expiry timestamp beyond which the prediction is stale.
        model_version: Version string for model-degradation detection.
        feature_hash: Hash of the input feature vector for reproducibility.
    """

    label: str
    prob: float
    confidence: float = 1.0
    raw_score: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)
    valid_until: datetime = field(default_factory=datetime.now)
    model_version: str = ""
    feature_hash: str = ""

    # -- properties ----------------------------------------------------------

    @property
    def is_stale(self) -> bool:
        """Whether the current time has passed ``valid_until``.

        Returns:
            ``True`` if the prediction has expired.
        """
        return datetime.now() > self.valid_until

    @property
    def freshness(self) -> float:
        """Linear decay from 1.0 (at ``generated_at``) to 0.0 (at ``valid_until``).

        Values are clamped to ``[0.0, 1.0]``.  If ``generated_at ==
        valid_until`` (zero-length window), returns ``0.0`` once the
        instant has passed, or ``1.0`` if it has not.

        Returns:
            Freshness score in ``[0.0, 1.0]``.
        """
        total = (self.valid_until - self.generated_at).total_seconds()
        if total <= 0:
            return 0.0 if datetime.now() >= self.valid_until else 1.0
        elapsed = (datetime.now() - self.generated_at).total_seconds()
        remaining = max(0.0, min(1.0, 1.0 - elapsed / total))
        return remaining

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since ``generated_at``.

        Returns:
            Non-negative float.
        """
        return max(0.0, (datetime.now() - self.generated_at).total_seconds())

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary.

        Datetime fields are converted to ISO-8601 strings.

        Returns:
            Plain dict suitable for ``json.dump``.
        """
        return {
            "label": self.label,
            "prob": self.prob,
            "confidence": self.confidence,
            "raw_score": self.raw_score,
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "model_version": self.model_version,
            "feature_hash": self.feature_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PredictionResult":
        """Deserialise from a dictionary.

        Accepts both ISO-8601 strings and ``datetime`` objects for
        timestamp fields.

        Args:
            d: Dictionary (typically from :meth:`to_dict` or JSON).

        Returns:
            Reconstructed ``PredictionResult``.
        """
        def _parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(str(val))

        return cls(
            label=str(d["label"]),
            prob=float(d["prob"]),
            confidence=float(d.get("confidence", 1.0)),
            raw_score=float(d.get("raw_score", 0.0)),
            generated_at=_parse_dt(d["generated_at"]),
            valid_until=_parse_dt(d["valid_until"]),
            model_version=str(d.get("model_version", "")),
            feature_hash=str(d.get("feature_hash", "")),
        )

    def __repr__(self) -> str:
        stale_tag = " STALE" if self.is_stale else ""
        return (
            f"PredictionResult({self.label}, p={self.prob:.3f}, "
            f"conf={self.confidence:.2f}, v={self.model_version}{stale_tag})"
        )


# ---------------------------------------------------------------------------
# AggregatedSignal
# ---------------------------------------------------------------------------

@dataclass
class AggregatedSignal:
    """Combined output produced by the aggregator layer.

    Strategies consume this object and should never need to inspect
    individual predictor outputs.

    Attributes:
        direction: Signed direction in ``[-1.0, 1.0]``.
            Positive = bullish, negative = bearish.
        strength: Signal magnitude in ``[0.0, 1.0]``.
        confidence: Aggregation confidence in ``[0.0, 1.0]``.
        contributing_predictors: Names of predictors that contributed.
        regime: Detected market regime label (e.g. ``"trending_up"``).
    """

    direction: float
    strength: float
    confidence: float
    contributing_predictors: list[str] = field(default_factory=list)
    regime: str = "unknown"

    # -- properties ----------------------------------------------------------

    @property
    def is_bullish(self) -> bool:
        """Whether the signal direction is positive.

        Returns:
            ``True`` if ``direction > 0``.
        """
        return self.direction > 0

    @property
    def is_bearish(self) -> bool:
        """Whether the signal direction is negative.

        Returns:
            ``True`` if ``direction < 0``.
        """
        return self.direction < 0

    @property
    def is_actionable(self) -> bool:
        """Whether the signal is strong enough to act on.

        Returns:
            ``True`` if ``strength > 0.5``.
        """
        return self.strength > 0.5

    def __repr__(self) -> str:
        arrow = "BULL" if self.is_bullish else ("BEAR" if self.is_bearish else "FLAT")
        return (
            f"AggregatedSignal({arrow}, dir={self.direction:+.3f}, "
            f"str={self.strength:.2f}, conf={self.confidence:.2f}, "
            f"regime={self.regime})"
        )
