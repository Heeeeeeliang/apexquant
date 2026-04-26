"""
Event-driven backtest engine for ApexQuant.

Iterates bars chronologically across all tickers.  On each bar:
attach AI signals, call ``strategy.on_bar()``, manage positions
(TP / SL / max-bars exits), and record trades.  Supports any
:class:`BaseStrategy` subclass including user-defined ones.

Usage::

    from backtest.engine import BacktestEngine, BacktestResult
    from strategies.builtin import AIStrategy
    from config.default import CONFIG

    strat = AIStrategy(CONFIG)
    engine = BacktestEngine(strat, CONFIG)
    result = engine.run(bars_by_ticker, predictions_by_ticker)
    print(result.metrics)
"""

__all__ = ["BacktestEngine", "BacktestResult"]

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from data.bar import Bar
from predictors.signals import SignalsProxy
from strategies.base import BaseStrategy, Signal, Trade

# Pipeline executor is opt-in — only used when BacktestEngine receives
# use_pipeline=True.  This prevents the pipeline from silently overriding
# all strategy logic.


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for backtest output.

    Attributes:
        trades: All closed trades.
        equity_curve: Portfolio value over time.
        metrics: Performance metrics (populated by :func:`backtest.metrics.compute_metrics`).
        strategy_name: Name of the strategy that was tested.
        start_date: First bar timestamp.
        end_date: Last bar timestamp.
        config_snapshot: Frozen copy of the config used.
    """

    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    start_date: datetime | None = None
    end_date: datetime | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of key fields.

        Returns:
            Dict with strategy name, trade count, date range, and metrics.
        """
        return {
            "strategy": self.strategy_name,
            "trades": len(self.trades),
            "start": self.start_date,
            "end": self.end_date,
            **self.metrics,
        }


# ---------------------------------------------------------------------------
# Indicator columns on Bar
# ---------------------------------------------------------------------------

_INDICATOR_MAP: dict[str, str] = {
    "ema_8": "ema_8",
    "ema_21": "ema_21",
    "ema_50": "ema_50",
    "rsi_14": "rsi_14",
    "macd": "macd",
    "macd_signal": "macd_signal",
    "macd_hist": "macd_hist",
    "atr_14": "atr_14",
    "bb_upper": "bb_upper",
    "bb_lower": "bb_lower",
    "bb_mid": "bb_mid",
    "volume_ratio": "volume_ratio",
    "vwap": "vwap",
    "obv": "obv",
    "adx_14": "adx_14",
    "stoch_k": "stoch_k",
    "stoch_d": "stoch_d",
    "willr_14": "willr_14",
    "cci_20": "cci_20",
    "mfi_14": "mfi_14",
}


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtester.

    Iterates bars chronologically.  On each bar: attach AI signals,
    call ``strategy.on_bar()``, manage positions, record trades.

    Attributes:
        strategy: The strategy being tested.
        config: Full CONFIG dict.
        capital: Initial capital.
        commission: Commission rate (decimal, e.g. 0.001).
        slippage: Slippage rate (decimal, e.g. 0.0005).
        open_positions: Currently open positions by ticker.
        closed_trades: All completed trades.
        equity_curve: Timestamped portfolio values.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: dict[str, Any],
        use_pipeline: bool = False,
    ) -> None:
        """Initialise the backtest engine.

        Args:
            strategy: A :class:`BaseStrategy` subclass instance.
            config: Full CONFIG dict.
            use_pipeline: If True, load ``config/pipeline.json`` and use
                the pipeline executor instead of the strategy's ``on_bar``.
        """
        self.strategy = strategy
        self.config = config

        bt_cfg = config.get("backtest", {})
        self.capital: float = bt_cfg.get("initial_capital", 100_000)
        self.commission: float = bt_cfg.get("commission", 0.001)
        self.slippage: float = bt_cfg.get("slippage", 0.0005)
        self.max_positions: int = bt_cfg.get("max_positions", 8)

        self.open_positions: dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []

        self._cash: float = self.capital
        self._portfolio_value: float = self.capital
        self._realised_pnl: float = 0.0  # running sum of closed-trade PnL
        self._equity_events: dict[datetime, float] = {}  # ts → equity at trade close

        # Store next-bar open prices for execution simulation
        self._next_opens: dict[str, float] = {}

        # Cache label-to-path mapping for SignalsProxy
        self._label_to_path: dict[str, str] = self._build_label_to_path()

        # Pipeline executor — opt-in only
        self._pipeline = None
        if use_pipeline:
            self._pipeline = self._load_pipeline()

        logger.info(
            "BacktestEngine: capital={:.0f}, commission={}, slippage={}, "
            "strategy={}, pipeline={}",
            self.capital,
            self.commission,
            self.slippage,
            strategy.name,
            self._pipeline is not None,
        )

    @staticmethod
    def _build_label_to_path() -> dict[str, str]:
        """Build the label-to-path map from the predictor registry."""
        try:
            from predictors.registry import REGISTRY
            return REGISTRY.get_label_to_path_map()
        except Exception:
            return {}

    @staticmethod
    def _load_pipeline():
        """Load pipeline executor from config/pipeline.json if it exists."""
        try:
            from pathlib import Path
            pipeline_path = Path("config/pipeline.json")
            if pipeline_path.exists():
                from pipeline.schema import Pipeline as PipelineModel
                from pipeline.executor import PipelineExecutor
                pipe = PipelineModel.model_validate_json(
                    pipeline_path.read_text(encoding="utf-8")
                )
                logger.info("Loaded pipeline from {}", pipeline_path)
                return PipelineExecutor(pipe)
        except Exception as exc:
            logger.debug("Failed to load pipeline: {}", exc)
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        bars_by_ticker: dict[str, pd.DataFrame],
        predictions_by_ticker: dict[str, pd.DataFrame] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            bars_by_ticker: Dict mapping ticker to DataFrame with
                OHLCV + indicator columns and a DatetimeIndex.
            predictions_by_ticker: Optional dict mapping ticker to
                DataFrame with prediction columns (e.g. ``vol_prob``,
                ``tp_score``, ``meta_prob``).  Columns are attached
                to ``bar.predictions``.
            start_date: Optional start date filter (inclusive).
            end_date: Optional end date filter (inclusive).

        Returns:
            :class:`BacktestResult` with trades, equity curve, and
            metadata.  Call :func:`backtest.metrics.compute_metrics`
            to populate ``result.metrics``.
        """
        if predictions_by_ticker is None:
            predictions_by_ticker = {}

        # --- Build unified chronological event stream ---
        events = self._build_event_stream(
            bars_by_ticker, predictions_by_ticker, start_date, end_date
        )

        if len(events) == 0:
            logger.warning("No events to process")
            return BacktestResult(
                strategy_name=self.strategy.name,
                config_snapshot=deepcopy(self.config),
            )

        logger.info(
            "Running backtest: {} events, {} tickers",
            len(events),
            len(bars_by_ticker),
        )

        # --- Lifecycle ---
        self.strategy.on_start()

        # --- Pending signals: execute at next bar's open ---
        pending_signals: dict[str, Signal] = {}

        for i, (timestamp, ticker, row, preds) in enumerate(events):
            bar = self._build_bar(ticker, timestamp, row, preds)

            # Check exits FIRST — before executing new entries so that
            # a trade opened this bar is never exit-checked until next bar
            self._check_exits(bar)

            # Execute pending signal from previous bar
            if ticker in pending_signals:
                signal = pending_signals.pop(ticker)
                next_open = bar.open
                self._execute_signal(signal, bar, next_open)

            # Get new signal from strategy (or pipeline)
            if self._pipeline is not None:
                action = self._pipeline.run(bar, bar.signals)
                if action == "BUY":
                    new_signal = Signal.BUY
                elif action == "SELL":
                    new_signal = Signal.SHORT
                else:
                    new_signal = Signal.HOLD
            else:
                new_signal = self.strategy.on_bar(bar)

            # Queue signal for next-bar execution (realistic simulation)
            if new_signal in (Signal.BUY, Signal.SHORT):
                if ticker in self.open_positions:
                    existing = self.open_positions[ticker]
                    # Auto-close opposing direction, then queue new signal
                    # Require minimum 3 bars held to avoid whipsaw reversals
                    if (new_signal == Signal.BUY and existing.is_short) or \
                       (new_signal == Signal.SHORT and existing.is_long):
                        if existing.bars_held >= 3:
                            self._close_position(ticker, bar.close, "opposing_signal", exit_ts=bar.timestamp)
                            if len(self.open_positions) < self.max_positions:
                                pending_signals[ticker] = new_signal
                    # Same direction already open — ignore duplicate
                else:
                    if len(self.open_positions) < self.max_positions:
                        pending_signals[ticker] = new_signal
                    else:
                        logger.debug(
                            "Max positions ({}) reached, skipping {} for {}",
                            self.max_positions,
                            new_signal.value,
                            ticker,
                        )
            elif new_signal == Signal.SELL:
                if ticker in self.open_positions:
                    self._close_position(ticker, bar.close, "signal", exit_ts=bar.timestamp)
            elif new_signal == Signal.COVER:
                if ticker in self.open_positions:
                    trade = self.open_positions[ticker]
                    if trade.is_short:
                        self._close_position(ticker, bar.close, "signal", exit_ts=bar.timestamp)

            # Update portfolio value
            self._update_portfolio_value(bar)

        # --- Close all remaining positions ---
        self._close_all_positions(events)

        # Record final equity after end-of-backtest closures
        last_ts = events[-1][0]
        if hasattr(last_ts, "to_pydatetime"):
            last_ts_dt = last_ts.to_pydatetime()
        else:
            last_ts_dt = last_ts
        self._equity_events[last_ts_dt] = self._portfolio_value

        # --- Lifecycle ---
        self.strategy.on_end()

        # --- Build equity curve: sorted dict → Series, forward-filled ---
        sorted_eq = sorted(self._equity_events.items())
        equity_series = pd.Series(
            [v for _, v in sorted_eq],
            index=pd.DatetimeIndex([t for t, _ in sorted_eq]),
            name="equity",
        )

        first_ts = events[0][0]
        last_ts = events[-1][0]

        result = BacktestResult(
            trades=list(self.closed_trades),
            equity_curve=equity_series,
            strategy_name=self.strategy.name,
            start_date=first_ts.to_pydatetime()
            if hasattr(first_ts, "to_pydatetime")
            else first_ts,
            end_date=last_ts.to_pydatetime()
            if hasattr(last_ts, "to_pydatetime")
            else last_ts,
            config_snapshot=deepcopy(self.config),
        )

        logger.info(
            "Backtest complete: {} trades, final equity={:.2f}",
            len(self.closed_trades),
            self._portfolio_value,
        )
        return result

    # ------------------------------------------------------------------
    # Event stream
    # ------------------------------------------------------------------

    def _build_event_stream(
        self,
        bars_by_ticker: dict[str, pd.DataFrame],
        predictions_by_ticker: dict[str, pd.DataFrame],
        start_date: str | None,
        end_date: str | None,
    ) -> list[tuple[pd.Timestamp, str, pd.Series, dict[str, float]]]:
        """Merge all tickers into a single chronological event stream.

        Args:
            bars_by_ticker: OHLCV DataFrames by ticker.
            predictions_by_ticker: Prediction DataFrames by ticker.
            start_date: Optional start filter.
            end_date: Optional end filter.

        Returns:
            Sorted list of ``(timestamp, ticker, row, predictions_dict)``
            tuples.
        """
        events: list[tuple[pd.Timestamp, str, pd.Series, dict[str, float]]] = []

        for ticker, df in bars_by_ticker.items():
            # Apply date filters
            filtered = df
            if start_date is not None:
                filtered = filtered[filtered.index >= pd.Timestamp(start_date, tz=filtered.index.tz)]
            if end_date is not None:
                filtered = filtered[filtered.index <= pd.Timestamp(end_date, tz=filtered.index.tz)]

            # Get predictions for this ticker
            pred_df = predictions_by_ticker.get(ticker)

            for ts, row in filtered.iterrows():
                preds: dict[str, float] = {}
                if pred_df is not None and ts in pred_df.index:
                    pred_row = pred_df.loc[ts]
                    for col in pred_df.columns:
                        val = pred_row[col] if not isinstance(pred_row, pd.DataFrame) else pred_row[col].iloc[0]
                        if pd.notna(val):
                            preds[col] = float(val)

                events.append((ts, ticker, row, preds))

        # Sort chronologically, then by ticker for determinism
        events.sort(key=lambda e: (e[0], e[1]))

        return events

    # ------------------------------------------------------------------
    # Bar construction
    # ------------------------------------------------------------------

    def _build_bar(
        self,
        ticker: str,
        timestamp: pd.Timestamp,
        row: pd.Series,
        predictions: dict[str, float],
    ) -> Bar:
        """Construct a Bar from a DataFrame row and predictions.

        Args:
            ticker: Stock symbol.
            timestamp: Bar timestamp.
            row: Series with OHLCV and optional indicator columns.
            predictions: Dict of prediction label to value.

        Returns:
            Fully populated :class:`Bar`.
        """
        bar = Bar(
            ticker=ticker,
            timestamp=timestamp,
            open=float(row.get("Open", 0.0)),
            high=float(row.get("High", 0.0)),
            low=float(row.get("Low", 0.0)),
            close=float(row.get("Close", 0.0)),
            volume=float(row.get("Volume", 0.0)),
        )

        # Attach technical indicators
        for col_name, bar_attr in _INDICATOR_MAP.items():
            if col_name in row.index:
                val = row[col_name]
                if pd.notna(val):
                    setattr(bar, bar_attr, float(val))

        # Attach predictions
        bar.predictions = dict(predictions)

        # Build SignalsProxy for path-based access
        bar.signals = SignalsProxy(bar.predictions, self._label_to_path)

        # Store extra columns in features
        ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
        indicator_cols = set(_INDICATOR_MAP.keys())
        for col in row.index:
            if col not in ohlcv_cols and col not in indicator_cols:
                val = row[col]
                if pd.notna(val):
                    try:
                        bar.features[col] = float(val)
                    except (TypeError, ValueError):
                        pass

        return bar

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    def _execute_signal(
        self, signal: Signal, bar: Bar, next_open: float
    ) -> Trade | None:
        """Open a new position at next bar's open with slippage.

        Args:
            signal: BUY or SHORT.
            bar: The current bar (used for metadata).
            next_open: Next bar's open price.

        Returns:
            The opened :class:`Trade`, or ``None`` if execution failed.
        """
        if bar.ticker in self.open_positions:
            return None

        # Apply slippage
        if signal == Signal.BUY:
            fill_price = next_open * (1.0 + self.slippage)
        else:
            fill_price = next_open * (1.0 - self.slippage)

        # Fixed position size from initial capital — no compounding
        size = self.strategy.get_position_size(bar)
        notional = self.capital * size
        entry_commission = notional * self.commission

        if notional <= 0 or self._cash < notional + entry_commission:
            logger.debug("Insufficient cash for {} {}", signal.value, bar.ticker)
            return None

        trade = Trade(
            ticker=bar.ticker,
            timestamp=bar.timestamp.to_pydatetime()
            if hasattr(bar.timestamp, "to_pydatetime")
            else bar.timestamp,
            signal=signal,
            entry_price=fill_price,
            size=size,
            commission=entry_commission,
        )

        self.open_positions[bar.ticker] = trade
        self._cash -= notional + entry_commission

        self.strategy.on_fill(trade)

        logger.debug(
            "Opened {} {} @ {:.2f} (size={:.4f}, notional={:.0f})",
            signal.value,
            bar.ticker,
            fill_price,
            size,
            notional,
        )
        return trade

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exits(self, bar: Bar) -> list[Trade]:
        """Check TP, SL, max_bars for the open position in this ticker.

        Args:
            bar: The current bar.

        Returns:
            List of trades closed on this bar.
        """
        closed: list[Trade] = []
        ticker = bar.ticker

        if ticker not in self.open_positions:
            return closed

        trade = self.open_positions[ticker]
        result = self.strategy.check_exit_conditions(bar, trade)

        if result is not None:
            reason, fill_price = result
            self._close_position(ticker, fill_price, reason, exit_ts=bar.timestamp)
            closed.append(trade)

        return closed

    def _close_position(
        self, ticker: str, exit_price: float, reason: str,
        exit_ts: datetime | None = None,
    ) -> Trade | None:
        """Close an open position and record it.

        Args:
            ticker: Stock symbol.
            exit_price: Price at exit.
            reason: Exit reason string.
            exit_ts: Exit timestamp (bar timestamp).

        Returns:
            The closed trade, or ``None``.
        """
        trade = self.open_positions.pop(ticker, None)
        if trade is None:
            return None

        # Apply slippage to exit
        if trade.is_long:
            fill_price = exit_price * (1.0 - self.slippage)
        else:
            fill_price = exit_price * (1.0 + self.slippage)

        trade.exit_price = fill_price
        trade.exit_reason = reason
        if exit_ts is not None and hasattr(exit_ts, "to_pydatetime"):
            exit_ts = exit_ts.to_pydatetime()
        trade.exit_timestamp = exit_ts

        # Compute PnL (percentage return on the trade)
        notional = self._cash_at_entry(trade)
        if trade.is_long:
            trade.pnl = (fill_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl = (trade.entry_price - fill_price) / trade.entry_price

        # Exit commission
        exit_commission = abs(notional) * self.commission
        trade.commission += exit_commission
        trade.slippage = self.slippage * 2

        # Return notional + profit/loss - exit commission to cash
        pnl_amount = trade.pnl * notional
        self._cash += notional + pnl_amount - exit_commission

        # Track running realised PnL (dollar PnL minus all commissions)
        net_trade_pnl = pnl_amount - trade.commission
        self._realised_pnl += net_trade_pnl
        self._portfolio_value = self.capital + self._realised_pnl

        self.closed_trades.append(trade)

        logger.debug(
            "Closed {} {} @ {:.2f} (pnl={:+.4f}, reason={}, bars={})",
            trade.signal.value,
            ticker,
            fill_price,
            trade.pnl,
            reason,
            trade.bars_held,
        )
        return trade

    def _close_all_positions(
        self, events: list[tuple[pd.Timestamp, str, pd.Series, dict[str, float]]]
    ) -> None:
        """Close all open positions at the last available price.

        Args:
            events: The full event stream (used to find last prices).
        """
        if not self.open_positions:
            return

        # Find last close price and timestamp per ticker
        last_prices: dict[str, tuple[float, pd.Timestamp]] = {}
        for ts, ticker, row, _ in reversed(events):
            if ticker not in last_prices:
                last_prices[ticker] = (float(row.get("Close", 0.0)), ts)
            if len(last_prices) >= len(self.open_positions):
                break

        tickers_to_close = list(self.open_positions.keys())
        for ticker in tickers_to_close:
            price, ts = last_prices.get(ticker, (0.0, events[-1][0]))
            if price > 0:
                self._close_position(ticker, price, "end_of_backtest", exit_ts=ts)
                logger.info(
                    "Force-closed {} at end of backtest @ {:.2f}",
                    ticker,
                    price,
                )

    # ------------------------------------------------------------------
    # Portfolio tracking
    # ------------------------------------------------------------------

    def _update_portfolio_value(self, bar: Bar) -> None:
        """Record equity snapshot for the current bar.

        Equity = initial_capital + realised_pnl (updated incrementally
        in _close_position).  Only changes when trades close, producing
        a stepped equity curve.

        Args:
            bar: The current bar (provides timestamp).
        """
        ts = bar.timestamp
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        # Overwrite same timestamp (multiple tickers per bar)
        self._equity_events[ts] = self._portfolio_value

    def _cash_at_entry(self, trade: Trade) -> float:
        """Estimate the notional cash allocated to a trade.

        Args:
            trade: An open or closed trade.

        Returns:
            Notional amount based on initial capital and position size.
        """
        return self.capital * trade.size

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BacktestEngine(strategy={self.strategy.name!r}, "
            f"capital={self.capital:.0f}, "
            f"positions={len(self.open_positions)})"
        )
