"""
Preflight health checks for ApexQuant backtest readiness.

Runs a battery of checks grouped into segments (models, data, config,
dependencies) and returns structured results that the UI renders as a
segmented health bar.

All check functions are pure — they inspect the filesystem and config
but perform no mutations.

Usage::

    from analytics.health import run_preflight, HealthReport

    report = run_preflight(config)
    print(report.overall)          # "green" | "yellow" | "red"
    for seg in report.segments:
        print(seg.name, seg.status, seg.details)
"""

__all__ = [
    "CheckStatus",
    "Segment",
    "HealthReport",
    "run_preflight",
    "check_models",
    "check_data",
    "check_config",
    "check_dependencies",
    "check_feature_alignment",
    "check_vol_prob_distribution",
]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class CheckStatus:
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class Segment:
    """Result of a single preflight check segment."""

    name: str
    status: str  # CheckStatus value
    details: list[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Aggregated preflight report across all segments."""

    segments: list[Segment] = field(default_factory=list)

    @property
    def overall(self) -> str:
        """Worst status across all segments."""
        if not self.segments:
            return CheckStatus.RED
        statuses = [s.status for s in self.segments]
        if CheckStatus.RED in statuses:
            return CheckStatus.RED
        if CheckStatus.YELLOW in statuses:
            return CheckStatus.YELLOW
        return CheckStatus.GREEN

    @property
    def can_run(self) -> bool:
        """Whether backtest can proceed (no RED segments)."""
        return all(s.status != CheckStatus.RED for s in self.segments)

    def get_segment(self, name: str) -> Segment | None:
        for s in self.segments:
            if s.name == name:
                return s
        return None


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_models(config: dict[str, Any]) -> Segment:
    """Check that registered model adapters have weights on disk.

    GREEN: all registered models have weights
    YELLOW: some models missing weights (backtest runs with degraded predictions)
    RED: zero models registered or all weights missing
    """
    details: list[str] = []

    try:
        from predictors.registry import REGISTRY

        names = REGISTRY.list_all()
    except Exception as exc:
        return Segment(
            name="Models",
            status=CheckStatus.RED,
            details=[f"Cannot load predictor registry: {exc}"],
        )

    if not names:
        return Segment(
            name="Models",
            status=CheckStatus.RED,
            details=["No models registered (models/ directory empty or missing)"],
        )

    ready_count = 0
    for name in names:
        try:
            pred = REGISTRY._predictors[name]
            if hasattr(pred, "is_ready") and pred.is_ready():
                ready_count += 1
                details.append(f"{name}: ready")
            else:
                details.append(f"{name}: weights missing")
        except Exception:
            details.append(f"{name}: error checking status")

    if ready_count == 0:
        return Segment(name="Models", status=CheckStatus.RED, details=details)
    if ready_count < len(names):
        return Segment(name="Models", status=CheckStatus.YELLOW, details=details)
    return Segment(name="Models", status=CheckStatus.GREEN, details=details)


def check_data(config: dict[str, Any]) -> Segment:
    """Check that feature CSV files exist for configured tickers.

    GREEN: all tickers have data files
    YELLOW: some tickers missing (backtest runs on available subset)
    RED: no data files found at all
    """
    details: list[str] = []
    tickers = config.get("data", {}).get("tickers", [])
    features_dir = Path(config.get("data", {}).get("features_dir", "data/features"))

    if not tickers:
        return Segment(
            name="Data",
            status=CheckStatus.RED,
            details=["No tickers configured in config['data']['tickers']"],
        )

    if not features_dir.exists():
        return Segment(
            name="Data",
            status=CheckStatus.RED,
            details=[f"Features directory does not exist: {features_dir}"],
        )

    found = 0
    for ticker in tickers:
        # Try common naming patterns
        patterns = [
            features_dir / f"{ticker}_1hour.csv",
            features_dir / f"{ticker}_1h.csv",
            features_dir / f"{ticker}_15min.csv",
            features_dir / f"{ticker}_1hour_features.csv",
        ]
        has_file = any(p.exists() for p in patterns)
        if has_file:
            found += 1
            details.append(f"{ticker}: found")
        else:
            details.append(f"{ticker}: missing")

    if found == 0:
        return Segment(name="Data", status=CheckStatus.RED, details=details)
    if found < len(tickers):
        return Segment(name="Data", status=CheckStatus.YELLOW, details=details)
    return Segment(name="Data", status=CheckStatus.GREEN, details=details)


def check_config(config: dict[str, Any]) -> Segment:
    """Validate critical config sections.

    GREEN: all checks pass
    YELLOW: non-critical issues (e.g. unusual parameter values)
    RED: missing required sections or invalid values
    """
    details: list[str] = []
    issues_red: list[str] = []
    issues_yellow: list[str] = []

    # Required top-level sections
    for section in ("data", "strategy", "backtest"):
        if section not in config:
            issues_red.append(f"Missing config section: {section}")

    # Data split ratios
    data_cfg = config.get("data", {})
    train = data_cfg.get("train_ratio", 0)
    val = data_cfg.get("val_ratio", 0)
    test = data_cfg.get("test_ratio", 0)
    ratio_sum = train + val + test
    if abs(ratio_sum - 1.0) > 0.05:
        issues_red.append(
            f"Data split ratios sum to {ratio_sum:.2f} (expected ~1.0)"
        )
    else:
        details.append(f"Data splits: {train}/{val}/{test} (sum={ratio_sum:.2f})")

    # Signal mode
    signal_mode = config.get("strategy", {}).get("signal_mode", "")
    valid_modes = ("ai", "technical", "hybrid")
    if signal_mode not in valid_modes:
        issues_red.append(
            f"Invalid signal_mode '{signal_mode}' (expected one of {valid_modes})"
        )
    else:
        details.append(f"Signal mode: {signal_mode}")

    # Initial capital
    capital = config.get("backtest", {}).get("initial_capital", 0)
    if capital <= 0:
        issues_red.append(f"Initial capital must be > 0 (got {capital})")
    else:
        details.append(f"Initial capital: ${capital:,.0f}")

    # Position size sanity
    pos_size = config.get("backtest", {}).get("position_size", 0)
    if pos_size <= 0 or pos_size > 1.0:
        issues_yellow.append(
            f"Position size {pos_size} outside (0, 1] range"
        )

    if issues_red:
        return Segment(
            name="Config",
            status=CheckStatus.RED,
            details=issues_red + issues_yellow + details,
        )
    if issues_yellow:
        return Segment(
            name="Config",
            status=CheckStatus.YELLOW,
            details=issues_yellow + details,
        )
    return Segment(name="Config", status=CheckStatus.GREEN, details=details)


def check_dependencies() -> Segment:
    """Check that critical Python packages are importable.

    GREEN: all imports succeed
    YELLOW: optional packages missing (e.g. torch for CNN)
    RED: core packages missing (pandas, numpy, lightgbm)
    """
    details: list[str] = []
    missing_core: list[str] = []
    missing_optional: list[str] = []

    # Core (RED if missing)
    for pkg in ("pandas", "numpy", "lightgbm", "joblib", "loguru"):
        try:
            __import__(pkg)
            details.append(f"{pkg}: ok")
        except ImportError:
            missing_core.append(pkg)
            details.append(f"{pkg}: MISSING")

    # Optional (YELLOW if missing)
    for pkg in ("torch", "plotly", "streamlit"):
        try:
            __import__(pkg)
            details.append(f"{pkg}: ok")
        except ImportError:
            missing_optional.append(pkg)
            details.append(f"{pkg}: missing (optional)")

    if missing_core:
        return Segment(
            name="Dependencies",
            status=CheckStatus.RED,
            details=[f"Core packages missing: {', '.join(missing_core)}"] + details,
        )
    if missing_optional:
        return Segment(
            name="Dependencies",
            status=CheckStatus.YELLOW,
            details=[f"Optional packages missing: {', '.join(missing_optional)}"] + details,
        )
    return Segment(name="Dependencies", status=CheckStatus.GREEN, details=details)


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def check_feature_alignment(config: dict[str, Any]) -> Segment:
    """Check that each model's feature_names.json exists and aligns with
    the features the pipeline would produce.

    Scans all ``models/*/feature_names.json`` files.  For vol models
    (lightgbm_v3, lightgbm_v3_flat), cross-references against the
    canonical ``_FC_LIST`` in ``data/vol_features.py``.

    GREEN: all feature_names.json found, no missing features
    YELLOW: feature_names.json found but minor mismatches (<=3 missing)
    RED: feature_names.json missing for a model, or >3 features missing
    """
    details: list[str] = []
    worst = CheckStatus.GREEN

    models_dir = Path("models")
    if not models_dir.exists():
        return Segment(
            name="FeatureAlignment",
            status=CheckStatus.RED,
            details=["models/ directory does not exist"],
        )

    # Collect all model dirs that have meta.json (= registered models)
    model_dirs: list[Path] = []
    for meta_path in models_dir.rglob("meta.json"):
        model_dirs.append(meta_path.parent)

    if not model_dirs:
        return Segment(
            name="FeatureAlignment",
            status=CheckStatus.YELLOW,
            details=["No model directories with meta.json found"],
        )

    # Try to load the canonical feature list from vol_features.py
    canonical_vol_features: list[str] | None = None
    try:
        from data.vol_features import _FC_LIST
        canonical_vol_features = list(_FC_LIST)
    except Exception:
        pass

    for mdir in model_dirs:
        fn_path = mdir / "feature_names.json"
        model_name = mdir.name

        if not fn_path.exists():
            # CNN models don't have feature_names.json — that's expected
            meta_path = mdir / "meta.json"
            adapter_type = ""
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        adapter_type = json.load(f).get("adapter", "")
                except Exception:
                    pass

            if adapter_type == "multiscale_cnn":
                details.append(f"{model_name}: CNN model (no feature_names.json expected)")
                continue

            details.append(f"{model_name}: feature_names.json MISSING")
            worst = CheckStatus.RED
            continue

        # Load and validate
        try:
            with open(fn_path, "r", encoding="utf-8") as f:
                fn_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            details.append(f"{model_name}: feature_names.json unreadable ({exc})")
            worst = CheckStatus.RED
            continue

        features = fn_data.get("features", fn_data) if isinstance(fn_data, dict) else fn_data
        n_expected = fn_data.get("n_features", len(features)) if isinstance(fn_data, dict) else len(features)

        if len(features) != n_expected:
            details.append(
                f"{model_name}: count mismatch (list={len(features)}, declared={n_expected})"
            )
            worst = _worse(worst, CheckStatus.RED)
            continue

        # For vol models, check alignment with canonical _FC_LIST
        if canonical_vol_features is not None and _is_vol_model(features):
            # Vol model features contain tech_{name} entries derived from _FC_LIST
            expected_tech = {f"tech_{f}" for f in canonical_vol_features}
            actual_tech = {f for f in features if f.startswith("tech_")}
            missing = expected_tech - actual_tech
            if missing:
                n_miss = len(missing)
                details.append(
                    f"{model_name}: {n_miss} tech features missing "
                    f"({', '.join(sorted(missing)[:5])}{'...' if n_miss > 5 else ''})"
                )
                worst = _worse(worst, CheckStatus.RED if n_miss > 3 else CheckStatus.YELLOW)
            else:
                details.append(f"{model_name}: {len(features)} features aligned")
        else:
            details.append(f"{model_name}: {len(features)} features (OK)")

    return Segment(name="FeatureAlignment", status=worst, details=details)


def _is_vol_model(features: list[str]) -> bool:
    """Heuristic: vol models have block features (b0_rv, b1_rv, ...) and tech_ prefixes."""
    return any(f.startswith("b0_") for f in features) and any(f.startswith("tech_") for f in features)


def _worse(a: str, b: str) -> str:
    """Return the worse of two statuses."""
    order = {CheckStatus.GREEN: 0, CheckStatus.YELLOW: 1, CheckStatus.RED: 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


# ---------------------------------------------------------------------------
# Vol prob distribution
# ---------------------------------------------------------------------------

# Thresholds for healthy vol_prob distribution
_VP_STD_MIN = 0.05       # std < 0.05 means collapsed
_VP_MEDIAN_LOW = 0.20    # median below 0.20 is suspiciously low
_VP_MEDIAN_HIGH = 0.80   # median above 0.80 is suspiciously high


def check_vol_prob_distribution(
    vol_probs: np.ndarray | None = None,
) -> Segment:
    """Check that vol_prob predictions have a healthy distribution.

    A collapsed distribution (std near 0, or all values clustered at one
    extreme) indicates a broken model or stale features.

    Args:
        vol_probs: Array of vol_prob values from a recent inference run.
            If None, attempts to load from the latest predictions CSV.

    GREEN: std >= 0.05 and median in [0.20, 0.80]
    YELLOW: marginal std (0.03-0.05) or median slightly outside range
    RED: std < 0.03 (collapsed) or no data available
    """
    details: list[str] = []

    if vol_probs is None:
        vol_probs = _load_latest_vol_probs()

    if vol_probs is None or len(vol_probs) == 0:
        return Segment(
            name="VolProbDistribution",
            status=CheckStatus.YELLOW,
            details=["No vol_prob data available (run inference first)"],
        )

    arr = np.asarray(vol_probs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return Segment(
            name="VolProbDistribution",
            status=CheckStatus.RED,
            details=["All vol_prob values are NaN"],
        )

    std = float(np.std(arr))
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))

    details.append(f"N={len(arr)}, median={median:.3f}, mean={mean:.3f}, std={std:.3f}")
    details.append(f"P10={p10:.3f}, P90={p90:.3f}")

    if std < 0.03:
        details.append(f"COLLAPSED: std={std:.4f} < 0.03")
        return Segment(name="VolProbDistribution", status=CheckStatus.RED, details=details)

    issues: list[str] = []

    if std < _VP_STD_MIN:
        issues.append(f"Low spread: std={std:.4f} < {_VP_STD_MIN}")

    if median < _VP_MEDIAN_LOW:
        issues.append(f"Median too low: {median:.3f} < {_VP_MEDIAN_LOW}")
    elif median > _VP_MEDIAN_HIGH:
        issues.append(f"Median too high: {median:.3f} > {_VP_MEDIAN_HIGH}")

    if issues:
        details.extend(issues)
        return Segment(name="VolProbDistribution", status=CheckStatus.YELLOW, details=details)

    details.append("Distribution looks healthy")
    return Segment(name="VolProbDistribution", status=CheckStatus.GREEN, details=details)


def _load_latest_vol_probs() -> np.ndarray | None:
    """Try to load vol_prob values from the latest prediction output."""
    import pandas as pd

    candidates = [
        Path("results/predictions/AAPL_predictions.csv"),
        Path("results/predictions/SPY_predictions.csv"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                for col in ("vol_prob", "vol_prob_flat"):
                    if col in df.columns:
                        return df[col].dropna().values
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Aggregated preflight
# ---------------------------------------------------------------------------

def run_preflight(
    config: dict[str, Any],
    vol_probs: np.ndarray | None = None,
) -> HealthReport:
    """Run all preflight checks and return an aggregated report.

    Args:
        config: Full CONFIG dict.
        vol_probs: Optional vol_prob array for distribution check.
            If None, attempts to load from saved predictions.

    Returns:
        A :class:`HealthReport` with one :class:`Segment` per check.
    """
    return HealthReport(
        segments=[
            check_models(config),
            check_data(config),
            check_config(config),
            check_dependencies(),
            check_feature_alignment(config),
            check_vol_prob_distribution(vol_probs),
        ]
    )
