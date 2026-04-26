"""
Typed configuration schema with guard validation.

Defines the canonical set of config keys, their types, and valid ranges.
Guards are pure functions that return a list of violation strings — an
empty list means the config is valid.

Usage::

    from config.schema import validate_config, GuardViolation

    violations = validate_config(config)
    if violations:
        for v in violations:
            print(v.level, v.field, v.message)
"""

__all__ = [
    "KNOWN_TICKERS",
    "GuardViolation",
    "validate_config",
]

from dataclasses import dataclass
from typing import Any


# The 8 tickers with data files on disk.
# GOOGL and GOOG are different Alphabet share classes ��� both are valid.
KNOWN_TICKERS = ("AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "TSLA", "SPY", "QQQ")

VALID_SIGNAL_MODES = ("ai", "technical", "hybrid")
VALID_DATA_SOURCES = ("csv", "yfinance", "databento_api")
VALID_BACKENDS = ("local", "colab", "gcloud", "aws")


@dataclass(frozen=True)
class GuardViolation:
    """A single config guard violation."""

    level: str  # "error" or "warning"
    field: str  # dot-path to the offending key
    message: str


def validate_config(config: dict[str, Any]) -> list[GuardViolation]:
    """Run all config guards and return violations.

    Args:
        config: Full CONFIG dict.

    Returns:
        List of :class:`GuardViolation`. Empty means valid.
    """
    vs: list[GuardViolation] = []
    vs.extend(_guard_data(config))
    vs.extend(_guard_strategy(config))
    vs.extend(_guard_backtest(config))
    vs.extend(_guard_model(config))
    return vs


# ---------------------------------------------------------------------------
# Individual guards
# ---------------------------------------------------------------------------

def _guard_data(config: dict[str, Any]) -> list[GuardViolation]:
    vs: list[GuardViolation] = []
    data = config.get("data", {})

    tickers = data.get("tickers", [])
    if not tickers:
        vs.append(GuardViolation("error", "data.tickers", "No tickers configured"))

    source = data.get("source", "")
    if source not in VALID_DATA_SOURCES:
        vs.append(GuardViolation(
            "error", "data.source",
            f"Invalid source '{source}' (expected one of {VALID_DATA_SOURCES})",
        ))

    train = data.get("train_ratio", 0)
    val = data.get("val_ratio", 0)
    test = data.get("test_ratio", 0)
    ratio_sum = train + val + test
    if abs(ratio_sum - 1.0) > 0.05:
        vs.append(GuardViolation(
            "error", "data.train_ratio",
            f"Split ratios sum to {ratio_sum:.2f} (expected ~1.0)",
        ))

    return vs


def _guard_strategy(config: dict[str, Any]) -> list[GuardViolation]:
    vs: list[GuardViolation] = []
    strat = config.get("strategy", {})

    mode = strat.get("signal_mode", "")
    if mode not in VALID_SIGNAL_MODES:
        vs.append(GuardViolation(
            "error", "strategy.signal_mode",
            f"Invalid signal_mode '{mode}' (expected one of {VALID_SIGNAL_MODES})",
        ))

    vol_thresh = strat.get("vol_threshold", 0.5)
    if not (0.0 <= vol_thresh <= 1.0):
        vs.append(GuardViolation(
            "warning", "strategy.vol_threshold",
            f"vol_threshold={vol_thresh} outside [0, 1]",
        ))

    return vs


def _guard_backtest(config: dict[str, Any]) -> list[GuardViolation]:
    vs: list[GuardViolation] = []
    bt = config.get("backtest", {})

    capital = bt.get("initial_capital", 0)
    if capital <= 0:
        vs.append(GuardViolation(
            "error", "backtest.initial_capital",
            f"Initial capital must be > 0 (got {capital})",
        ))

    pos = bt.get("position_size", 0)
    if pos <= 0 or pos > 1.0:
        vs.append(GuardViolation(
            "warning", "backtest.position_size",
            f"Position size {pos} outside (0, 1]",
        ))

    commission = bt.get("commission", 0)
    if commission < 0:
        vs.append(GuardViolation(
            "error", "backtest.commission",
            f"Commission cannot be negative ({commission})",
        ))

    slippage = bt.get("slippage", 0)
    if slippage < 0:
        vs.append(GuardViolation(
            "error", "backtest.slippage",
            f"Slippage cannot be negative ({slippage})",
        ))

    return vs


def _guard_model(config: dict[str, Any]) -> list[GuardViolation]:
    vs: list[GuardViolation] = []
    model = config.get("model", {})

    tp = model.get("meta_tp", 0)
    sl = model.get("meta_sl", 0)
    mb = model.get("meta_mb", 0)

    if tp <= 0:
        vs.append(GuardViolation("error", "model.meta_tp", f"TP must be > 0 (got {tp})"))
    if sl <= 0:
        vs.append(GuardViolation("error", "model.meta_sl", f"SL must be > 0 (got {sl})"))
    if mb < 1:
        vs.append(GuardViolation("error", "model.meta_mb", f"Max bars must be >= 1 (got {mb})"))

    if tp > 0 and sl > 0 and tp < sl:
        vs.append(GuardViolation(
            "warning", "model.meta_tp",
            f"TP ({tp:.4f}) < SL ({sl:.4f}) — risk/reward is inverted",
        ))

    return vs
