"""
Feature drift detection between training-time and runtime feature sets.

Reuses :func:`backtest.inference.load_feature_names_from_dir` for loading
``feature_names.json`` — no duplication.

Usage::

    from diagnostics.feature_drift import check_feature_drift
    report = check_feature_drift(runtime_columns, model_dir)
"""

__all__ = ["FeatureDriftReport", "check_feature_drift"]

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FeatureDriftReport:
    """Result of a feature drift check for one model."""

    model_name: str
    training_count: int
    runtime_count: int
    missing_in_runtime: list[str] = field(default_factory=list)
    extra_in_runtime: list[str] = field(default_factory=list)

    @property
    def is_aligned(self) -> bool:
        return not self.missing_in_runtime and not self.extra_in_runtime

    @property
    def drift_severity(self) -> str:
        """'none', 'warning' (extras only), 'error' (missing features)."""
        if self.is_aligned:
            return "none"
        if self.missing_in_runtime:
            return "error"
        return "warning"


def check_feature_drift(
    runtime_columns: list[str],
    model_dir: Path | str,
    model_name: str | None = None,
) -> FeatureDriftReport:
    """Compare runtime feature columns against a model's training features.

    Args:
        runtime_columns: Feature column names available at inference time.
        model_dir: Path to the model directory (contains feature_names.json).
        model_name: Display name for the report. Defaults to directory name.

    Returns:
        A :class:`FeatureDriftReport`.
    """
    from backtest.inference import load_feature_names_from_dir

    model_dir = Path(model_dir)
    name = model_name or model_dir.name

    training_names = load_feature_names_from_dir(model_dir)
    if training_names is None:
        return FeatureDriftReport(
            model_name=name,
            training_count=0,
            runtime_count=len(runtime_columns),
            missing_in_runtime=[],
            extra_in_runtime=[],
        )

    training_set = set(training_names)
    runtime_set = set(runtime_columns)

    missing = sorted(training_set - runtime_set)
    extra = sorted(runtime_set - training_set)

    return FeatureDriftReport(
        model_name=name,
        training_count=len(training_names),
        runtime_count=len(runtime_columns),
        missing_in_runtime=missing,
        extra_in_runtime=extra,
    )
