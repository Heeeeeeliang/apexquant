"""
Per-run trade quality metrics beyond standard backtest metrics.

Computes streak analysis, time-of-day edge, and drawdown decomposition
from a list of trades. All functions are pure — no I/O.

Usage::

    from diagnostics.trade_quality import compute_trade_quality
    report = compute_trade_quality(result.trades)
"""

__all__ = ["TradeQualityReport", "compute_trade_quality"]

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeQualityReport:
    """Trade quality diagnostics for a single backtest run."""

    # Streak analysis
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0

    # Time-of-day edge (hour → avg pnl)
    hourly_pnl: dict[int, float] = field(default_factory=dict)
    best_hour: int | None = None
    worst_hour: int | None = None

    # Ticker concentration
    ticker_trade_counts: dict[str, int] = field(default_factory=dict)
    ticker_pnl_sums: dict[str, float] = field(default_factory=dict)

    # Drawdown recovery
    avg_bars_to_recover: float = 0.0

    # Exit reason win rates
    exit_reason_win_rates: dict[str, float] = field(default_factory=dict)

    # Summary
    total_trades: int = 0


def compute_trade_quality(trades: Any) -> TradeQualityReport:
    """Compute trade quality diagnostics.

    Args:
        trades: ``BacktestResult.trades`` (list[Trade]) or a trades
            DataFrame with pnl, timestamp, ticker, exit_reason columns.

    Returns:
        A :class:`TradeQualityReport`.
    """
    df = _trades_to_df(trades)
    if df.empty or "pnl" not in df.columns:
        return TradeQualityReport()

    df = df.dropna(subset=["pnl"])
    if df.empty:
        return TradeQualityReport()

    report = TradeQualityReport(total_trades=len(df))

    # --- Streak analysis ---
    wins = (df["pnl"] > 0).values
    report.max_win_streak, report.avg_win_streak = _streak_stats(wins, True)
    report.max_loss_streak, report.avg_loss_streak = _streak_stats(wins, False)

    # --- Hourly PnL ---
    if "hour" in df.columns:
        hourly = df.groupby("hour")["pnl"].mean()
        report.hourly_pnl = hourly.to_dict()
        if len(hourly) > 0:
            report.best_hour = int(hourly.idxmax())
            report.worst_hour = int(hourly.idxmin())

    # --- Ticker concentration ---
    if "ticker" in df.columns:
        report.ticker_trade_counts = df["ticker"].value_counts().to_dict()
        report.ticker_pnl_sums = df.groupby("ticker")["pnl"].sum().to_dict()

    # --- Exit reason win rates ---
    if "exit_reason" in df.columns:
        for reason, group in df.groupby("exit_reason"):
            total = len(group)
            wins_count = (group["pnl"] > 0).sum()
            report.exit_reason_win_rates[reason] = wins_count / total if total > 0 else 0.0

    # --- Avg bars in drawdown (consecutive losing trades) ---
    if "bars_held" in df.columns:
        loss_bars = df.loc[df["pnl"] <= 0, "bars_held"]
        report.avg_bars_to_recover = float(loss_bars.mean()) if len(loss_bars) > 0 else 0.0

    return report


def _trades_to_df(trades: Any) -> pd.DataFrame:
    """Normalise input to DataFrame."""
    if isinstance(trades, pd.DataFrame):
        df = trades.copy()
        # Try to extract hour from timestamp
        if "hour" not in df.columns and "timestamp" in df.columns:
            try:
                ts = pd.to_datetime(df["timestamp"])
                df["hour"] = ts.dt.hour
            except Exception:
                pass
        return df

    rows: list[dict[str, Any]] = []
    for t in trades:
        ts = t.timestamp
        hour = ts.hour if hasattr(ts, "hour") else None
        rows.append({
            "pnl": t.pnl,
            "exit_reason": t.exit_reason or "unknown",
            "ticker": t.ticker,
            "bars_held": t.bars_held,
            "hour": hour,
            "signal": t.signal.value if hasattr(t.signal, "value") else str(t.signal),
        })
    return pd.DataFrame(rows)


def _streak_stats(is_win: np.ndarray, target: bool) -> tuple[int, float]:
    """Compute max and avg streak length for target (True=wins, False=losses)."""
    if len(is_win) == 0:
        return 0, 0.0

    streaks: list[int] = []
    current = 0
    for w in is_win:
        if w == target:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)

    if not streaks:
        return 0, 0.0
    return max(streaks), float(np.mean(streaks))
