"""
Traffic-light verdict for a completed backtest.

Pure function of ``BacktestResult.metrics`` — no I/O, no randomness.

Verdict levels
--------------
GREEN  PRODUCTION_READY  — strong risk_adjusted returns, acceptable drawdown
YELLOW WARNING           — profitable but has structural concerns (bias, low WR)
RED    LOSING            — negative returns or catastrophic risk metrics

Usage::

    from analytics.verdict import compute_verdict
    v = compute_verdict(result.metrics)
    print(v.level, v.label, v.details)
"""

__all__ = ["Verdict", "VerdictLevel", "compute_verdict"]

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerdictLevel(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True)
class Verdict:
    """Immutable verdict result."""

    level: VerdictLevel
    label: str
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Thresholds (tuned to the project's acceptance criteria)
# ---------------------------------------------------------------------------

# RED if any of these fail
_RED_MAX_DD = -0.40           # MaxDD worse than -40%
_RED_MIN_RETURN = 0.0         # total_return <= 0
_RED_MIN_SHARPE = 0.0         # Sharpe <= 0
_RED_MIN_WIN_RATE = 0.30      # WR below 30%

# GREEN requires ALL of these
_GREEN_MIN_RETURN = 0.20      # total_return >= 20%
_GREEN_MIN_SHARPE = 1.50      # Sharpe >= 1.5
_GREEN_MAX_DD = -0.15         # MaxDD better than -15%
_GREEN_MIN_WIN_RATE = 0.45    # WR >= 45%
_GREEN_MIN_PROFIT_FACTOR = 1.5

# Direction bias: flag if >90% of trades are one direction
_BIAS_THRESHOLD = 0.90


def compute_verdict(metrics: dict[str, Any]) -> Verdict:
    """Compute a traffic-light verdict from backtest metrics.

    Args:
        metrics: The ``BacktestResult.metrics`` dict.  Expected keys:
            total_return, sharpe_ratio, max_drawdown, win_rate,
            total_trades, long_trades, short_trades, profit_factor.

    Returns:
        A :class:`Verdict` with level, label, and detail strings.
    """
    total_return = float(metrics.get("total_return", 0.0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    total_trades = int(metrics.get("total_trades", 0))
    long_trades = int(metrics.get("long_trades", 0))
    short_trades = int(metrics.get("short_trades", 0))
    profit_factor = float(metrics.get("profit_factor", 0.0))

    details: list[str] = []

    # --- Check direction bias ---
    bias_flag: str | None = None
    if total_trades > 0:
        long_ratio = long_trades / total_trades
        short_ratio = short_trades / total_trades
        if short_ratio >= _BIAS_THRESHOLD:
            bias_flag = "SHORT_BIAS"
            details.append(
                f"SHORT bias: {short_ratio:.1%} of trades are SHORT"
            )
        elif long_ratio >= _BIAS_THRESHOLD:
            bias_flag = "LONG_BIAS"
            details.append(
                f"LONG bias: {long_ratio:.1%} of trades are LONG"
            )

    # --- RED checks ---
    red_reasons: list[str] = []
    if total_return <= _RED_MIN_RETURN:
        red_reasons.append(f"total_return={total_return:.2%} <= 0")
    if sharpe <= _RED_MIN_SHARPE:
        red_reasons.append(f"sharpe={sharpe:.2f} <= 0")
    if max_dd < _RED_MAX_DD:
        red_reasons.append(f"max_drawdown={max_dd:.2%} < {_RED_MAX_DD:.0%}")
    if win_rate < _RED_MIN_WIN_RATE and total_trades > 0:
        red_reasons.append(f"win_rate={win_rate:.1%} < {_RED_MIN_WIN_RATE:.0%}")

    if red_reasons:
        details.extend(red_reasons)
        label = "LOSING"
        if bias_flag:
            label = f"LOSING"
        return Verdict(level=VerdictLevel.RED, label=label, details=details)

    # --- GREEN checks ---
    green_pass = (
        total_return >= _GREEN_MIN_RETURN
        and sharpe >= _GREEN_MIN_SHARPE
        and max_dd >= _GREEN_MAX_DD
        and win_rate >= _GREEN_MIN_WIN_RATE
        and profit_factor >= _GREEN_MIN_PROFIT_FACTOR
    )

    if green_pass and not bias_flag:
        details.append("All thresholds passed")
        return Verdict(
            level=VerdictLevel.GREEN,
            label="PRODUCTION_READY",
            details=details,
        )

    # --- YELLOW: profitable but with concerns ---
    if bias_flag:
        return Verdict(
            level=VerdictLevel.YELLOW,
            label=bias_flag,
            details=details,
        )

    # Profitable but didn't meet all GREEN thresholds
    yellow_reasons: list[str] = []
    if total_return < _GREEN_MIN_RETURN:
        yellow_reasons.append(f"total_return={total_return:.2%} < {_GREEN_MIN_RETURN:.0%}")
    if sharpe < _GREEN_MIN_SHARPE:
        yellow_reasons.append(f"sharpe={sharpe:.2f} < {_GREEN_MIN_SHARPE}")
    if max_dd < _GREEN_MAX_DD:
        yellow_reasons.append(f"max_drawdown={max_dd:.2%} < {_GREEN_MAX_DD:.0%}")
    if win_rate < _GREEN_MIN_WIN_RATE:
        yellow_reasons.append(f"win_rate={win_rate:.1%} < {_GREEN_MIN_WIN_RATE:.0%}")
    if profit_factor < _GREEN_MIN_PROFIT_FACTOR:
        yellow_reasons.append(f"profit_factor={profit_factor:.2f} < {_GREEN_MIN_PROFIT_FACTOR}")
    details.extend(yellow_reasons)

    return Verdict(
        level=VerdictLevel.YELLOW,
        label="MARGINAL",
        details=details,
    )
