"""
Backtrader-based backtest runner for ApexQuant.

Drop-in replacement for the legacy engine path in ``backtest/runner.py``.
Sets up ``bt.Cerebro`` with multi-ticker feeds, runs the
:class:`BtStrategyWrapper`, and extracts results into the existing
:class:`BacktestResult` contract.

This module is NOT called directly by the frontend — it is used by
``backtest/runner.py`` which handles strategy instantiation, data
loading, metrics computation, and report generation.

Usage::

    from backtest.bt_runner import run_engine
    result = run_engine(strategy, config, bars_by_ticker, predictions_by_ticker)
"""

__all__ = ["run_engine"]

import time
import traceback
from copy import deepcopy
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd
from loguru import logger

from backtest.bt_feeds import make_feed
from backtest.bt_strategy import BtStrategyWrapper
from backtest.engine import BacktestResult
from strategies.base import BaseStrategy


def _compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vectorized ATR as fraction of close price. No look-ahead."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return (tr / close).rolling(window=period, min_periods=period).mean()


def run_engine(
    strategy: BaseStrategy,
    config: dict[str, Any],
    bars_by_ticker: dict[str, pd.DataFrame],
    predictions_by_ticker: dict[str, pd.DataFrame] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    progress_callback=None,
) -> BacktestResult:
    """Run a backtest using Backtrader.

    This replaces ``BacktestEngine.run()`` with identical inputs/outputs.

    Args:
        strategy: A BaseStrategy subclass instance.
        config: Full CONFIG dict.
        bars_by_ticker: Dict mapping ticker to DataFrame with
            OHLCV + indicator columns and a DatetimeIndex.
        predictions_by_ticker: Optional dict mapping ticker to
            predictions DataFrame.
        start_date: Optional start date filter (inclusive).
        end_date: Optional end date filter (inclusive).

    Returns:
        BacktestResult with trades, equity curve, and metadata.
    """
    if predictions_by_ticker is None:
        predictions_by_ticker = {}

    bt_cfg = config.get("backtest", {})
    capital = bt_cfg.get("initial_capital", 100_000)

    # --- Build Cerebro ---
    cerebro = bt.Cerebro(stdstats=False)

    # We do NOT use Backtrader's broker for execution — it's all manual
    # in BtStrategyWrapper. Set cash high to prevent Backtrader complaints.
    cerebro.broker.setcash(capital * 100)

    # --- Add data feeds ---
    # features_by_ticker: ticker → {ts_ns: {col: val}} — side-channel
    features_by_ticker: dict[str, dict[int, dict[str, float]]] = {}
    extra_cols_by_ticker: dict[str, list[str]] = {}
    has_data = False
    total_bars = 0

    for ticker, df in bars_by_ticker.items():
        # Apply date filters
        filtered = df.copy()
        if start_date is not None:
            tz = filtered.index.tz
            filtered = filtered[
                filtered.index >= pd.Timestamp(start_date, tz=tz)
            ]
        if end_date is not None:
            tz = filtered.index.tz
            filtered = filtered[
                filtered.index <= pd.Timestamp(end_date, tz=tz)
            ]

        if filtered.empty:
            logger.warning("No data for {} in date range", ticker)
            continue

        t_feed_start = time.perf_counter()
        feed, features_ns, extra_cols = make_feed(filtered, name=ticker)
        t_feed_elapsed = time.perf_counter() - t_feed_start
        logger.info(
            "[perf] make_feed({}): {:.3f}s, {} bars, {} features",
            ticker, t_feed_elapsed, len(filtered), len(extra_cols),
        )
        cerebro.adddata(feed, name=ticker)
        features_by_ticker[ticker] = features_ns
        extra_cols_by_ticker[ticker] = extra_cols
        total_bars += len(filtered)
        has_data = True

    if not has_data:
        logger.warning("No data feeds loaded — returning empty result")
        empty = BacktestResult(
            strategy_name=strategy.name,
            config_snapshot=deepcopy(config),
        )
        return empty

    # --- Pre-compute ATR for all tickers (vectorized, O(1) per-bar lookup) ---
    atr_lookup: dict[str, pd.Series] = {}
    for ticker, df in bars_by_ticker.items():
        cols_lower = {c.lower() for c in df.columns}
        if "high" in cols_lower and "low" in cols_lower and "close" in cols_lower:
            # Normalise to lowercase for ATR computation
            df_lc = df.rename(columns={c: c.lower() for c in df.columns})
            atr_s = _compute_atr_series(df_lc, period=14)
            # Strip tz so lookup matches tz-naive bar timestamps
            if atr_s.index.tz is not None:
                atr_s.index = atr_s.index.tz_localize(None)
            atr_lookup[ticker] = atr_s
            logger.debug("Pre-computed ATR series for {} ({} values)", ticker, atr_lookup[ticker].notna().sum())
    # Inject into strategy instance so get_tp_sl uses O(1) lookups
    if hasattr(strategy, "_atr_lookup"):
        strategy._atr_lookup = atr_lookup

    # --- Build label-to-path map ---
    label_to_path = _build_label_to_path()

    # --- Add strategy ---
    cerebro.addstrategy(
        BtStrategyWrapper,
        base_strategy=strategy,
        config=config,
        predictions_by_ticker=predictions_by_ticker,
        features_by_ticker=features_by_ticker,
        extra_cols_by_ticker=extra_cols_by_ticker,
        label_to_path=label_to_path,
        progress_callback=progress_callback,
        total_bars=total_bars,
    )

    # --- Run ---
    logger.info(
        "Running Backtrader: {} tickers, capital={:.0f}, strategy={}",
        len(extra_cols_by_ticker),
        capital,
        strategy.name,
    )
    t_run_start = time.perf_counter()
    try:
        results = cerebro.run()
    except Exception:
        logger.error(
            "cerebro.run() FAILED:\n{}", traceback.format_exc(),
        )
        raise
    t_run_elapsed = time.perf_counter() - t_run_start
    logger.info("[perf] cerebro.run(): {:.3f}s", t_run_elapsed)
    bt_strat = results[0]

    # --- Extract results ---
    closed_trades = bt_strat._closed_trades
    equity_events = bt_strat._equity_events

    # Build equity curve as pd.Series with DatetimeIndex
    try:
        if equity_events:
            sorted_eq = sorted(equity_events.items())
            timestamps = [pd.Timestamp(t) for t, _ in sorted_eq]
            values = [float(v) for _, v in sorted_eq]
            equity_series = pd.Series(
                values,
                index=pd.DatetimeIndex(timestamps),
                name="equity",
            )
        else:
            equity_series = pd.Series(
                [float(capital)],
                index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")]),
                name="equity",
            )
    except Exception as exc:
        logger.warning(
            "Failed to build equity Series from {} events ({}): {} — "
            "using flat equity",
            len(equity_events), type(exc).__name__, exc,
        )
        equity_series = pd.Series(
            [float(capital)],
            index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")]),
            name="equity",
        )

    # Determine date range
    first_ts = equity_series.index[0] if len(equity_series) > 0 else None
    last_ts = equity_series.index[-1] if len(equity_series) > 0 else None

    result = BacktestResult(
        trades=list(closed_trades),
        equity_curve=equity_series,
        strategy_name=strategy.name,
        start_date=first_ts,
        end_date=last_ts,
        config_snapshot=deepcopy(config),
    )

    logger.info(
        "Backtrader complete: {} trades, final equity={:.2f}",
        len(closed_trades),
        bt_strat._portfolio_value,
    )

    return result


def _build_label_to_path() -> dict[str, str]:
    """Build the label-to-path map from the predictor registry."""
    try:
        from predictors.registry import REGISTRY
        return REGISTRY.get_label_to_path_map()
    except Exception:
        return {}
