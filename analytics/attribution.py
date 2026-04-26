"""
PnL attribution for a completed backtest.

Groups trades by exit reason or conviction tier and computes
percentage-based metrics.  All functions are pure — no I/O, no randomness.

Usage::

    from analytics.attribution import by_exit_reason, by_conviction_tier

    # From BacktestResult.trades (list[Trade])
    table = by_exit_reason(result.trades)

    # From a trades DataFrame (loaded from CSV)
    table = by_exit_reason(trades_df)
"""

__all__ = ["by_exit_reason", "by_conviction_tier"]

from typing import Any

import pandas as pd


def _trades_to_df(trades: Any) -> pd.DataFrame:
    """Normalise input to a DataFrame with 'pnl' and grouping columns.

    Accepts either a list of Trade objects or a pandas DataFrame.
    """
    if isinstance(trades, pd.DataFrame):
        return trades.copy()

    rows: list[dict[str, Any]] = []
    for t in trades:
        rows.append({
            "pnl": t.pnl,
            "exit_reason": t.exit_reason or "unknown",
            "conviction_tier": getattr(t, "conviction_tier", "") or "",
            "signal": t.signal.value if hasattr(t.signal, "value") else str(t.signal),
        })
    return pd.DataFrame(rows)


def by_exit_reason(trades: Any) -> pd.DataFrame:
    """Attribution table grouped by exit reason.

    Args:
        trades: ``BacktestResult.trades`` (list[Trade]) or a trades DataFrame
            with at least ``pnl`` and ``exit_reason`` columns.

    Returns:
        DataFrame with columns:
            exit_reason, trade_count, win_count, win_rate,
            avg_pnl_pct, sum_pnl_pct, contribution_pct
        Sorted by contribution_pct descending.
    """
    df = _trades_to_df(trades)
    if df.empty or "pnl" not in df.columns or "exit_reason" not in df.columns:
        return _empty_exit_reason_df()

    df = df.dropna(subset=["pnl"])
    if df.empty:
        return _empty_exit_reason_df()

    total_sum_pnl = df["pnl"].sum()

    groups = df.groupby("exit_reason", sort=False)

    rows: list[dict[str, Any]] = []
    for reason, group in groups:
        pnls = group["pnl"]
        count = len(pnls)
        wins = int((pnls > 0).sum())
        sum_pnl = float(pnls.sum())

        rows.append({
            "exit_reason": reason,
            "trade_count": count,
            "win_count": wins,
            "win_rate": wins / count if count > 0 else 0.0,
            "avg_pnl_pct": float(pnls.mean()),
            "sum_pnl_pct": sum_pnl,
            "contribution_pct": (
                sum_pnl / total_sum_pnl if abs(total_sum_pnl) > 1e-15 else 0.0
            ),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("contribution_pct", ascending=False).reset_index(drop=True)
    return result


def by_conviction_tier(trades: Any) -> pd.DataFrame:
    """Attribution table grouped by conviction tier.

    Only meaningful when ``conviction_tier`` is populated on trades.
    Values are ``"high"`` (vol_prob > 0.7) and ``"mid"`` (vol_prob 0.5-0.7).
    Trades with vol_prob <= 0.5 are filtered before entry and never appear.

    Args:
        trades: ``BacktestResult.trades`` (list[Trade]) or a trades DataFrame
            with at least ``pnl`` and ``conviction_tier`` columns.

    Returns:
        DataFrame with columns:
            conviction_tier, trade_count, win_count, win_rate,
            avg_pnl_pct, sum_pnl_pct, contribution_pct
        Sorted by conviction_tier (high first).
        Returns empty DataFrame if conviction_tier data is unavailable.
    """
    df = _trades_to_df(trades)
    if df.empty or "pnl" not in df.columns:
        return _empty_conviction_df()

    if "conviction_tier" not in df.columns:
        return _empty_conviction_df()

    df = df.dropna(subset=["pnl"])

    # Filter to rows that actually have a tier value
    df = df[df["conviction_tier"].isin(("high", "mid"))]
    if df.empty:
        return _empty_conviction_df()

    total_sum_pnl = df["pnl"].sum()

    groups = df.groupby("conviction_tier", sort=False)

    rows: list[dict[str, Any]] = []
    for tier, group in groups:
        pnls = group["pnl"]
        count = len(pnls)
        wins = int((pnls > 0).sum())
        sum_pnl = float(pnls.sum())

        rows.append({
            "conviction_tier": tier,
            "trade_count": count,
            "win_count": wins,
            "win_rate": wins / count if count > 0 else 0.0,
            "avg_pnl_pct": float(pnls.mean()),
            "sum_pnl_pct": sum_pnl,
            "contribution_pct": (
                sum_pnl / total_sum_pnl if abs(total_sum_pnl) > 1e-15 else 0.0
            ),
        })

    result = pd.DataFrame(rows)
    # Sort: high first
    tier_order = {"high": 0, "mid": 1}
    result["_sort"] = result["conviction_tier"].map(tier_order).fillna(2)
    result = result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return result


def _empty_exit_reason_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "exit_reason", "trade_count", "win_count", "win_rate",
        "avg_pnl_pct", "sum_pnl_pct", "contribution_pct",
    ])


def _empty_conviction_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "conviction_tier", "trade_count", "win_count", "win_rate",
        "avg_pnl_pct", "sum_pnl_pct", "contribution_pct",
    ])
