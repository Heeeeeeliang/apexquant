"""
Diagnostics module for ApexQuant backtest post-run analysis.

Provides:

- :mod:`diagnostics.feature_drift` — detect feature mismatches between
  training-time and runtime feature sets
- :mod:`diagnostics.trade_quality` — per-run trade quality metrics
  (streak analysis, time-of-day patterns, drawdown decomposition)
- :mod:`diagnostics.engine_hooks` — lightweight collector that attaches
  to a BacktestResult after a run completes
"""

from diagnostics.engine_hooks import DiagnosticsReport, collect_diagnostics
from diagnostics.equity_scan import scan_equity_curve
from diagnostics.feature_drift import check_feature_drift
from diagnostics.pnl_autocorrelation import scan_pnl_autocorrelation
from diagnostics.trade_clustering import scan_trade_clustering
from diagnostics.trade_quality import compute_trade_quality

__all__ = [
    "DiagnosticsReport",
    "collect_diagnostics",
    "check_feature_drift",
    "compute_trade_quality",
    "scan_equity_curve",
    "scan_trade_clustering",
    "scan_pnl_autocorrelation",
]
