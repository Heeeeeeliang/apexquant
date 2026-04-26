"""
Analytics module for ApexQuant backtest evaluation.

Provides:

- :mod:`analytics.verdict` — traffic-light verdict from BacktestResult metrics
- :mod:`analytics.attribution` — PnL attribution by exit reason and conviction tier
- :mod:`analytics.health` — preflight health checks for backtest readiness
"""

from analytics.attribution import by_conviction_tier, by_exit_reason
from analytics.health import (
    HealthReport,
    Segment,
    check_feature_alignment,
    check_vol_prob_distribution,
    run_preflight,
)
from analytics.verdict import Verdict, VerdictLevel, compute_verdict

__all__ = [
    "Verdict",
    "VerdictLevel",
    "compute_verdict",
    "by_exit_reason",
    "by_conviction_tier",
    "HealthReport",
    "Segment",
    "run_preflight",
]
