"""
Trade clustering scan — detect temporal bunching.

Flags periods where too many trades open in a short window, which may
indicate overtrading or signal noise.

Usage::

    from diagnostics.trade_clustering import scan_trade_clustering
    report = scan_trade_clustering(trades_df)
"""

__all__ = ["ClusteringScanReport", "scan_trade_clustering"]

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ClusteringScanReport:
    """Trade clustering diagnostics."""

    # Temporal bunching
    max_trades_per_day: int = 0
    avg_trades_per_day: float = 0.0
    busiest_day: str = ""
    days_with_trades: int = 0
    total_days: int = 0

    # Burst detection (>= 2x avg rate in a single day)
    burst_days: int = 0
    burst_pct: float = 0.0  # fraction of trading days that are bursts

    # Inter-trade gap
    median_gap_hours: float = 0.0
    min_gap_hours: float = 0.0

    total_trades: int = 0

    @property
    def has_clustering(self) -> bool:
        """True if burst_pct > 20% or max_trades_per_day > 5x avg."""
        if self.avg_trades_per_day < 0.01:
            return False
        if self.burst_pct > 0.20:
            return True
        if self.max_trades_per_day > 5 * self.avg_trades_per_day:
            return True
        return False


def scan_trade_clustering(trades: Any) -> ClusteringScanReport:
    """Detect temporal clustering in trade entries.

    Args:
        trades: ``BacktestResult.trades`` (list[Trade]) or a trades
            DataFrame with a ``timestamp`` column.

    Returns:
        A :class:`ClusteringScanReport`.
    """
    df = _to_df(trades)
    if df.empty or "timestamp" not in df.columns:
        return ClusteringScanReport()

    try:
        ts = pd.to_datetime(df["timestamp"])
    except Exception:
        return ClusteringScanReport(total_trades=len(df))

    ts = ts.sort_values().reset_index(drop=True)
    n = len(ts)
    report = ClusteringScanReport(total_trades=n)

    if n == 0:
        return report

    # Daily counts
    dates = ts.dt.date
    daily_counts = dates.value_counts()
    report.days_with_trades = len(daily_counts)
    report.max_trades_per_day = int(daily_counts.max())
    report.busiest_day = str(daily_counts.idxmax())

    total_calendar_days = (ts.iloc[-1] - ts.iloc[0]).days + 1
    report.total_days = total_calendar_days
    report.avg_trades_per_day = n / total_calendar_days if total_calendar_days > 0 else 0.0

    # Burst detection: days with >= 2x average
    threshold = max(2.0 * report.avg_trades_per_day, 2)
    report.burst_days = int((daily_counts >= threshold).sum())
    report.burst_pct = (
        report.burst_days / report.days_with_trades
        if report.days_with_trades > 0 else 0.0
    )

    # Inter-trade gaps
    if n > 1:
        gaps = ts.diff().dropna()
        gap_hours = gaps.dt.total_seconds() / 3600
        report.median_gap_hours = float(gap_hours.median())
        report.min_gap_hours = float(gap_hours.min())

    return report


def _to_df(trades: Any) -> pd.DataFrame:
    if isinstance(trades, pd.DataFrame):
        return trades
    rows = []
    for t in trades:
        rows.append({"timestamp": t.timestamp})
    return pd.DataFrame(rows)
