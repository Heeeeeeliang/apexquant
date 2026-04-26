"""
ApexQuant Backtest Module
==========================

Event-driven backtester that iterates bars chronologically, attaches
AI predictions, calls strategy hooks, manages positions with TP/SL/max_bars,
and produces comprehensive performance metrics.

Usage::

    from backtest import BacktestEngine, BacktestResult
    from backtest.metrics import compute_metrics
    from backtest.reporter import BacktestReporter
    from backtest.runner import run_backtest, run_comparison
    from config.default import CONFIG

    # Direct engine usage
    engine = BacktestEngine(strategy, CONFIG)
    result = engine.run(bars_by_ticker, predictions_by_ticker)
    metrics = compute_metrics(result)

    # High-level runner
    result = run_backtest(CONFIG, strategy_name="ai")
    ai_result, tech_result = run_comparison(CONFIG)
"""

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestReporter",
    "run_backtest",
    "run_comparison",
]

from backtest.engine import BacktestEngine, BacktestResult
from backtest.reporter import BacktestReporter
from backtest.runner import run_backtest, run_comparison
