"""
Equity curve health scan.

Analyses the equity curve (pd.Series with DatetimeIndex) for:
- prolonged underwater periods
- flatness (no meaningful change over long stretches)
- worst drawdown recovery time

All functions are pure — no I/O.

Usage::

    from diagnostics.equity_scan import scan_equity_curve
    report = scan_equity_curve(result.equity_curve)
"""

__all__ = ["EquityScanReport", "scan_equity_curve"]

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EquityScanReport:
    """Equity curve health diagnostics."""

    # Underwater analysis
    max_underwater_days: int = 0
    current_underwater_days: int = 0
    n_underwater_periods: int = 0
    pct_time_underwater: float = 0.0

    # Flatness detection
    flat_days: int = 0  # days where equity changed < 0.01%
    flat_pct: float = 0.0  # fraction of total days that are flat

    # Recovery
    worst_dd_recovery_days: int = 0  # days from trough back to HWM

    # Overall
    total_days: int = 0
    equity_points: int = 0

    @property
    def health(self) -> str:
        """'good', 'caution', or 'poor'."""
        if self.equity_points < 2:
            return "poor"
        if self.pct_time_underwater > 0.80 or self.flat_pct > 0.50:
            return "poor"
        if self.pct_time_underwater > 0.50 or self.flat_pct > 0.30:
            return "caution"
        return "good"


def scan_equity_curve(equity: pd.Series | None) -> EquityScanReport:
    """Analyse an equity curve for health issues.

    Args:
        equity: Portfolio value Series with DatetimeIndex, as stored
            in ``BacktestResult.equity_curve``.

    Returns:
        An :class:`EquityScanReport`.
    """
    if equity is None or len(equity) < 2:
        return EquityScanReport(equity_points=0 if equity is None else len(equity))

    if not isinstance(equity.index, pd.DatetimeIndex):
        return EquityScanReport(equity_points=len(equity))

    report = EquityScanReport(equity_points=len(equity))

    # Resample to daily for consistent analysis
    daily = equity.resample("1D").last().dropna()
    if len(daily) < 2:
        return report

    report.total_days = (daily.index[-1] - daily.index[0]).days

    # --- Underwater analysis ---
    cummax = daily.cummax()
    is_underwater = daily < cummax

    # Count underwater periods and their durations
    uw_durations: list[int] = []
    current_start = None
    for i in range(len(daily)):
        if is_underwater.iloc[i]:
            if current_start is None:
                current_start = daily.index[i]
        else:
            if current_start is not None:
                duration = (daily.index[i] - current_start).days
                uw_durations.append(duration)
                current_start = None

    # Handle ongoing underwater period
    if current_start is not None:
        duration = (daily.index[-1] - current_start).days
        uw_durations.append(duration)
        report.current_underwater_days = duration

    report.n_underwater_periods = len(uw_durations)
    report.max_underwater_days = max(uw_durations) if uw_durations else 0

    total_uw_days = int(is_underwater.sum())
    report.pct_time_underwater = total_uw_days / len(daily) if len(daily) > 0 else 0.0

    # --- Flatness detection ---
    daily_returns = daily.pct_change().dropna()
    flat_mask = daily_returns.abs() < 0.0001  # < 0.01% change
    report.flat_days = int(flat_mask.sum())
    report.flat_pct = report.flat_days / len(daily_returns) if len(daily_returns) > 0 else 0.0

    # --- Worst drawdown recovery ---
    drawdown = (daily - cummax) / cummax
    trough_idx = drawdown.idxmin()
    trough_pos = daily.index.get_loc(trough_idx)
    hwm_at_trough = cummax.iloc[trough_pos]

    # Find when equity recovered to the HWM after the trough
    post_trough = daily.iloc[trough_pos:]
    recovered = post_trough[post_trough >= hwm_at_trough]
    if len(recovered) > 0:
        recovery_date = recovered.index[0]
        report.worst_dd_recovery_days = (recovery_date - trough_idx).days
    else:
        # Still hasn't recovered
        report.worst_dd_recovery_days = (daily.index[-1] - trough_idx).days

    return report
