"""
Performance metrics for ApexQuant backtests.

Computes a comprehensive set of risk_adjusted and trade-level
metrics from a :class:`BacktestResult`.

Usage::

    from backtest.metrics import compute_metrics
    from backtest.engine import BacktestResult

    result = engine.run(bars_by_ticker)
    metrics = compute_metrics(result)
    print(metrics["sharpe_ratio"])
"""

__all__ = ["compute_metrics"]

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backtest.engine import BacktestResult


def _default_metrics() -> dict[str, Any]:
    """Return a zeroed-out metrics dict for empty / failed backtests."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0,
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_trade_pnl": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "win_loss_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_bars_held": 0.0,
        "max_bars_held": 0,
        "exit_reasons": {},
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
        "calmar_ratio": 0.0,
    }


def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, Any]:
    """Compute full performance metrics from a backtest result.

    Populates ``result.metrics`` in-place and also returns the dict.

    Args:
        result: A completed :class:`BacktestResult`.
        risk_free_rate: Annual riskless rate for Sharpe / Sortino.
        trading_days: Trading days per year (default 252).

    Returns:
        Dict with all computed metrics.
    """
    equity = result.equity_curve
    trades = result.trades

    # --- Guard: empty or non-resamplable equity curve ---
    if len(equity) < 2 or not isinstance(equity.index, pd.DatetimeIndex):
        logger.warning(
            "Equity curve is empty or lacks DatetimeIndex "
            "(len={}, index_type={}) — returning zero metrics",
            len(equity),
            type(equity.index).__name__,
        )
        metrics = _default_metrics()
        metrics["total_trades"] = len(trades)
        result.metrics = metrics
        return metrics

    metrics: dict[str, Any] = {}

    # --- Return metrics ---
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days > 0:
        annualized_return = (1.0 + total_return) ** (365.0 / n_days) - 1.0
    else:
        annualized_return = 0.0

    metrics["total_return"] = float(total_return)
    metrics["annualized_return"] = float(annualized_return)

    # --- Daily returns (resample to end-of-day to avoid intra-bar noise) ---
    daily_equity = equity.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    # --- Sharpe ratio ---
    if len(daily_returns) > 1 and daily_returns.std() > 1e-10:
        daily_rf = risk_free_rate / trading_days
        excess = daily_returns - daily_rf
        sharpe = float(excess.mean() / excess.std() * np.sqrt(trading_days))
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = sharpe

    # --- Sortino ratio ---
    if len(daily_returns) > 1:
        daily_rf = risk_free_rate / trading_days
        excess = daily_returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) > 0 and downside.std() > 1e-10:
            sortino = float(excess.mean() / downside.std() * np.sqrt(trading_days))
        else:
            sortino = 0.0
    else:
        sortino = 0.0
    metrics["sortino_ratio"] = sortino

    # --- Max drawdown ---
    dd, dd_duration = _max_drawdown(equity)
    metrics["max_drawdown"] = float(dd)
    metrics["max_drawdown_duration_days"] = int(dd_duration)

    # --- Trade metrics ---
    pnls = [t.pnl for t in trades if t.pnl is not None]

    metrics["total_trades"] = len(trades)

    if pnls:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics["win_rate"] = float(len(wins) / len(pnls))
        metrics["avg_trade_pnl"] = float(np.mean(pnls))
        metrics["avg_win"] = float(np.mean(wins)) if wins else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if losses else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        metrics["profit_factor"] = (
            float(gross_profit / gross_loss) if gross_loss > 1e-10 else float("inf")
        )

        bars_held = [t.bars_held for t in trades]
        metrics["avg_bars_held"] = float(np.mean(bars_held)) if bars_held else 0.0
        metrics["max_bars_held"] = int(max(bars_held)) if bars_held else 0

        # Exit reason breakdown
        reasons: dict[str, int] = {}
        for t in trades:
            r = t.exit_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        metrics["exit_reasons"] = reasons

        # Long vs short breakdown
        long_pnls = [t.pnl for t in trades if t.is_long and t.pnl is not None]
        short_pnls = [t.pnl for t in trades if t.is_short and t.pnl is not None]
        metrics["long_trades"] = len(long_pnls)
        metrics["short_trades"] = len(short_pnls)
        metrics["long_win_rate"] = (
            float(sum(1 for p in long_pnls if p > 0) / len(long_pnls))
            if long_pnls
            else 0.0
        )
        metrics["short_win_rate"] = (
            float(sum(1 for p in short_pnls if p > 0) / len(short_pnls))
            if short_pnls
            else 0.0
        )

        # PnL-weighted win rate: sum(|pnl| for wins) / sum(|pnl| for all)
        abs_pnls = [abs(p) for p in pnls]
        abs_wins = [abs(p) for p in pnls if p > 0]
        total_abs = sum(abs_pnls)
        metrics["weighted_win_rate"] = float(sum(abs_wins) / total_abs) if total_abs > 1e-15 else 0.0

        # Win/loss size ratio
        metrics["win_loss_ratio"] = (
            float(metrics["avg_win"] / abs(metrics["avg_loss"]))
            if metrics["avg_loss"] != 0.0 else float("inf")
        )
    else:
        metrics["win_rate"] = 0.0
        metrics["weighted_win_rate"] = 0.0
        metrics["win_loss_ratio"] = 0.0
        metrics["avg_trade_pnl"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_bars_held"] = 0.0
        metrics["max_bars_held"] = 0
        metrics["exit_reasons"] = {}
        metrics["long_trades"] = 0
        metrics["short_trades"] = 0
        metrics["long_win_rate"] = 0.0
        metrics["short_win_rate"] = 0.0

    # --- Calmar ratio ---
    if abs(dd) > 1e-10:
        metrics["calmar_ratio"] = float(annualized_return / abs(dd))
    else:
        metrics["calmar_ratio"] = 0.0

    # --- Populate result in-place ---
    result.metrics = metrics

    logger.info(
        "Metrics: return={:.2%}, sharpe={:.2f}, sortino={:.2f}, "
        "max_dd={:.2%}, win_rate={:.1%}, trades={}",
        metrics["total_return"],
        metrics["sharpe_ratio"],
        metrics["sortino_ratio"],
        metrics["max_drawdown"],
        metrics["win_rate"],
        metrics["total_trades"],
    )

    return metrics


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Compute maximum drawdown and its duration in days.

    Args:
        equity: Portfolio equity time series with DatetimeIndex.

    Returns:
        Tuple of ``(max_drawdown, duration_days)``.
        Drawdown is negative (e.g. -0.15 for 15% drawdown).
    """
    if len(equity) < 2:
        return 0.0, 0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax

    max_dd = float(drawdown.min())

    # Duration: longest streak below the high-water mark
    is_in_dd = equity < cummax
    if not is_in_dd.any():
        return max_dd, 0

    # Find contiguous drawdown periods
    max_duration_days = 0
    current_start = None

    for i in range(len(equity)):
        if is_in_dd.iloc[i]:
            if current_start is None:
                current_start = equity.index[i]
        else:
            if current_start is not None:
                duration = (equity.index[i] - current_start).days
                max_duration_days = max(max_duration_days, duration)
                current_start = None

    # Handle drawdown that extends to the end
    if current_start is not None:
        duration = (equity.index[-1] - current_start).days
        max_duration_days = max(max_duration_days, duration)

    return max_dd, max_duration_days
