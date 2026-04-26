"""
PnL autocorrelation scan — detect serial dependence in trade outcomes.

Significant positive autocorrelation at lag-1 suggests regime-dependent
performance (winning streaks beget wins). Negative autocorrelation
suggests mean-reversion in outcomes. Neither is inherently bad, but
both are worth knowing about.

Usage::

    from diagnostics.pnl_autocorrelation import scan_pnl_autocorrelation
    report = scan_pnl_autocorrelation(trades_df)
"""

__all__ = ["PnlAutocorrReport", "scan_pnl_autocorrelation"]

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PnlAutocorrReport:
    """PnL autocorrelation diagnostics."""

    # Lag-1 autocorrelation of PnL series
    lag1_autocorr: float = 0.0

    # Win/loss transition probabilities
    p_win_after_win: float = 0.0
    p_win_after_loss: float = 0.0
    p_loss_after_win: float = 0.0
    p_loss_after_loss: float = 0.0

    # Significance
    n_trades: int = 0
    is_significant: bool = False  # |autocorr| > 2/sqrt(n)

    @property
    def regime_signal(self) -> str:
        """'momentum' (positive autocorr), 'mean_revert' (negative), or 'none'."""
        if not self.is_significant:
            return "none"
        if self.lag1_autocorr > 0:
            return "momentum"
        return "mean_revert"


def scan_pnl_autocorrelation(trades: Any) -> PnlAutocorrReport:
    """Compute PnL autocorrelation and transition probabilities.

    Args:
        trades: ``BacktestResult.trades`` (list[Trade]) or a trades
            DataFrame with a ``pnl`` column.

    Returns:
        A :class:`PnlAutocorrReport`.
    """
    df = _to_df(trades)
    if df.empty or "pnl" not in df.columns:
        return PnlAutocorrReport()

    pnls = df["pnl"].dropna().values
    n = len(pnls)
    if n < 10:
        return PnlAutocorrReport(n_trades=n)

    report = PnlAutocorrReport(n_trades=n)

    # Lag-1 autocorrelation
    pnl_series = pd.Series(pnls)
    autocorr = float(pnl_series.autocorr(lag=1))
    if np.isnan(autocorr):
        autocorr = 0.0
    report.lag1_autocorr = autocorr

    # Significance: |autocorr| > 2/sqrt(n) (approximate 95% CI)
    threshold = 2.0 / np.sqrt(n)
    report.is_significant = abs(autocorr) > threshold

    # Win/loss transition matrix
    wins = pnls > 0
    if n > 1:
        # Transitions
        ww = wl = lw = ll = 0
        for i in range(1, n):
            if wins[i - 1]:  # previous was win
                if wins[i]:
                    ww += 1
                else:
                    wl += 1
            else:  # previous was loss
                if wins[i]:
                    lw += 1
                else:
                    ll += 1

        total_after_win = ww + wl
        total_after_loss = lw + ll

        if total_after_win > 0:
            report.p_win_after_win = ww / total_after_win
            report.p_loss_after_win = wl / total_after_win
        if total_after_loss > 0:
            report.p_win_after_loss = lw / total_after_loss
            report.p_loss_after_loss = ll / total_after_loss

    return report


def _to_df(trades: Any) -> pd.DataFrame:
    if isinstance(trades, pd.DataFrame):
        return trades
    rows = []
    for t in trades:
        rows.append({"pnl": t.pnl})
    return pd.DataFrame(rows)
