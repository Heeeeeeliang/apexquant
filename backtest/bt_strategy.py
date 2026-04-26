"""
Backtrader strategy wrapper for ApexQuant.

Bridges Backtrader's ``bt.Strategy.next()`` loop with ApexQuant's
``BaseStrategy.on_bar(bar) → Signal`` interface.  Manages positions,
TP/SL/max-bars exits, opposing-signal logic, and builds Trade objects
compatible with the existing BacktestResult contract.

This wrapper does NOT use Backtrader's built-in broker/order system.
Instead it manages positions manually to maintain exact behavioural
parity with the legacy engine (closed-trade-only equity, specific
slippage model, fixed-notional position sizing from initial capital).

Features and indicators are passed as plain Python dicts (the
``features_by_ticker`` side-channel) — they are **not** registered
as Backtrader Line objects.  This eliminates the per-bar
synchronisation cost of 160+ extra lines.

Usage::

    cerebro.addstrategy(
        BtStrategyWrapper,
        base_strategy=my_strategy,
        config=CONFIG,
        predictions_by_ticker=predictions,
        features_by_ticker=features_dicts,
        extra_cols_by_ticker=extra_col_names,
    )
"""

__all__ = ["BtStrategyWrapper"]

import time
from datetime import datetime
from typing import Any

import backtrader as bt
import pandas as pd
from loguru import logger

from data.bar import Bar
from predictors.signals import SignalsProxy
from strategies.base import BaseStrategy, Signal, Trade


# Indicator columns that map directly to Bar dataclass fields
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

# Pre-build set for O(1) membership test
_INDICATOR_NAMES = frozenset(_INDICATOR_MAP)


class BtStrategyWrapper(bt.Strategy):
    """Backtrader strategy that wraps an ApexQuant BaseStrategy.

    All position management is done manually (not via Backtrader's broker)
    to maintain exact parity with the legacy engine.

    Params:
        base_strategy: A BaseStrategy subclass instance.
        config: Full CONFIG dict.
        predictions_by_ticker: Dict of ticker → predictions DataFrame.
        features_by_ticker: Dict of ticker → {ts_ns: {col: val}}.
        extra_cols_by_ticker: Dict of ticker → list of extra column names.
        label_to_path: Dict mapping prediction labels to folder paths.
    """

    params = (
        ("base_strategy", None),
        ("config", None),
        ("predictions_by_ticker", None),
        ("features_by_ticker", None),
        ("extra_cols_by_ticker", None),
        ("label_to_path", None),
        ("use_pipeline", False),
        ("progress_callback", None),
        ("total_bars", 0),
    )

    def __init__(self):
        self._base: BaseStrategy = self.p.base_strategy
        self._config: dict = self.p.config
        self._predictions: dict = self.p.predictions_by_ticker or {}
        self._features: dict[str, dict[int, dict[str, float]]] = (
            self.p.features_by_ticker or {}
        )
        self._extra_cols: dict = self.p.extra_cols_by_ticker or {}
        self._label_to_path: dict = self.p.label_to_path or {}

        # Backtest config
        bt_cfg = self._config.get("backtest", {})
        self._initial_capital: float = bt_cfg.get("initial_capital", 100_000)
        self._commission_rate: float = bt_cfg.get("commission", 0.001)
        self._slippage_rate: float = bt_cfg.get("slippage", 0.0005)
        self._min_commission: float = bt_cfg.get("min_commission", 0.0)
        self._max_positions: int = bt_cfg.get("max_positions", 8)
        self._min_hold_bars: int = 3

        # Position tracking (manual — not using Backtrader's broker)
        self._open_positions: dict[str, Trade] = {}
        self._closed_trades: list[Trade] = []
        self._cash: float = self._initial_capital
        self._realised_pnl: float = 0.0
        self._portfolio_value: float = self._initial_capital

        # Conviction-tier position limits
        self._max_high_positions: int = 3
        self._max_mid_positions: int = 5
        self._max_exposure_pct: float = 0.85

        # Equity curve: timestamp → mark-to-market portfolio value
        self._equity_events: dict[datetime, float] = {}
        self._last_prices: dict[str, float] = {}  # ticker → last close for MTM

        # Pending signals: execute at next bar's open
        self._pending_signals: dict[str, Signal] = {}

        # Track entry bar index per ticker for bars_held computation
        self._entry_bar_idx: dict[str, int] = {}

        # Map data feeds by name for multi-ticker iteration
        self._feeds: dict[str, bt.AbstractDataBase] = {}
        for data in self.datas:
            name = data._name
            if name:
                self._feeds[name] = data

        # Progress tracking
        self._progress_cb = self.p.progress_callback
        self._total_bars: int = self.p.total_bars or 0
        self._bar_count: int = 0

        # Pipeline executor (opt-in, same as legacy engine)
        self._pipeline = None
        if self.p.use_pipeline:
            self._pipeline = self._load_pipeline()

        # ----------------------------------------------------------
        # Precompute prediction lookup: ticker → {ts_ns: {col: val}}
        # This replaces per-bar DataFrame.loc lookups with O(1) dict.
        # ----------------------------------------------------------
        self._pred_lookup: dict[str, dict[int, dict[str, float]]] = {}
        for ticker, pred_df in self._predictions.items():
            if pred_df is None or pred_df.empty:
                continue
            idx = pred_df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            lookup: dict[int, dict[str, float]] = {}
            cols = list(pred_df.columns)
            for i in range(len(pred_df)):
                ts_ns = int(idx[i].value)
                row = pred_df.iloc[i]
                preds: dict[str, float] = {}
                for col in cols:
                    val = row[col]
                    if pd.notna(val):
                        preds[col] = float(val)
                if preds:
                    lookup[ts_ns] = preds
            self._pred_lookup[ticker] = lookup
            logger.info(
                "Precomputed {} prediction rows for {}",
                len(lookup), ticker,
            )

        # Ablation feature flags (read from config["_ablation"])
        _abl = self._config.get("_ablation", {})
        self._enable_tranches: bool = _abl.get("enable_tranches", True)
        self._enable_signal_reversal: bool = _abl.get("enable_signal_reversal", True)
        self._enable_vol_collapse: bool = _abl.get("enable_vol_collapse", True)
        self._enable_preemption: bool = _abl.get("enable_preemption", True)
        self._enable_trail_stop: bool = _abl.get("enable_trail_stop", True)

        # Per-ticker: which feature columns are also indicator attrs
        self._indicator_cols_per_ticker: dict[str, list[tuple[str, str]]] = {}
        for ticker, cols in self._extra_cols.items():
            self._indicator_cols_per_ticker[ticker] = [
                (col, _INDICATOR_MAP[col])
                for col in cols if col in _INDICATOR_NAMES
            ]

        # ----------------------------------------------------------
        # Timing accumulators
        # ----------------------------------------------------------
        self._t_build_bar: float = 0.0
        self._t_on_bar: float = 0.0
        self._t_exits: float = 0.0
        self._t_execute: float = 0.0
        self._t_process: float = 0.0

        logger.info(
            "BtStrategyWrapper: {} tickers, capital={:.0f}, strategy={}, "
            "features_side_channel={} tickers",
            len(self._feeds),
            self._initial_capital,
            self._base.name,
            len(self._features),
        )

    def _load_pipeline(self):
        """Load pipeline executor if config/pipeline.json exists."""
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
                strat_cfg = self._base.config.get("strategy", {})
                tb_config = {
                    "trend_bypass_period": strat_cfg.get("trend_bypass_period", 20),
                    "trend_bypass_pct": strat_cfg.get("trend_bypass_pct", 0.03),
                    "trend_bypass_min_vol": strat_cfg.get("trend_bypass_min_vol", 0.35),
                }
                return PipelineExecutor(pipe, trend_bypass_config=tb_config)
        except Exception as exc:
            logger.debug("Failed to load pipeline: {}", exc)
        return None

    # ------------------------------------------------------------------
    # Backtrader lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Called by Backtrader before the first bar."""
        self._base.on_start()

    def stop(self):
        """Called by Backtrader after the last bar — force-close open positions."""
        self._close_all_positions()

        # Record final equity
        if self.datas and len(self.datas[0]) > 0:
            last_dt = self.datas[0].datetime.datetime(0)
            self._equity_events[last_dt] = self._portfolio_value

        self._base.on_end()

        # Log final timing summary
        total = (
            self._t_build_bar + self._t_on_bar
            + self._t_exits + self._t_execute + self._t_process
        )
        logger.info(
            "[perf] FINAL after {} bars: build_bar={:.3f}s, "
            "on_bar={:.3f}s, exits={:.3f}s, "
            "execute={:.3f}s, process={:.3f}s, TOTAL={:.3f}s",
            self._bar_count,
            self._t_build_bar,
            self._t_on_bar,
            self._t_exits,
            self._t_execute,
            self._t_process,
            total,
        )
        if self._bar_count > 0:
            logger.info(
                "[perf] Per-bar avg: {:.1f}us total, "
                "build_bar={:.1f}us, on_bar={:.1f}us, exits={:.1f}us",
                total / self._bar_count * 1e6,
                self._t_build_bar / self._bar_count * 1e6,
                self._t_on_bar / self._bar_count * 1e6,
                self._t_exits / self._bar_count * 1e6,
            )

    def next(self):
        """Process one bar across all tickers."""
        for ticker, data in self._feeds.items():
            if len(data) == 0:
                continue

            # Report progress
            self._bar_count += 1
            if self._progress_cb is not None and self._total_bars > 0:
                self._progress_cb(
                    self._bar_count, self._total_bars, ticker
                )

            ts = data.datetime.datetime(0)

            # 0. Build bar (features from side-channel dict, not BT lines)
            t0 = time.perf_counter()
            bar = self._build_bar(ticker, data, ts)
            t1 = time.perf_counter()
            self._t_build_bar += t1 - t0

            # 1. Check exits BEFORE new entries
            if ticker in self._open_positions:
                self._check_exits(bar)
            t2 = time.perf_counter()
            self._t_exits += t2 - t1

            # 2. Execute pending signal from previous bar
            if ticker in self._pending_signals:
                signal = self._pending_signals.pop(ticker)
                self._execute_signal(signal, bar, bar.open)
            t3 = time.perf_counter()
            self._t_execute += t3 - t2

            # 3. Get new signal from strategy (or pipeline)
            if self._pipeline is not None:
                action = self._pipeline.run(bar, bar.signals)
                if action == "BUY":
                    new_signal = Signal.BUY
                elif action == "SHORT":
                    new_signal = Signal.SHORT
                elif action in ("CLOSE", "SELL"):
                    new_signal = Signal.SELL
                else:
                    new_signal = Signal.HOLD
            else:
                new_signal = self._base.on_bar(bar)
            t4 = time.perf_counter()
            self._t_on_bar += t4 - t3

            # 4. Queue signal for next-bar execution
            self._process_signal(ticker, new_signal, bar)
            t5 = time.perf_counter()
            self._t_process += t5 - t4

            # 5. Update equity snapshot (mark-to-market)
            self._last_prices[ticker] = bar.close
            unrealised = 0.0
            for pos_ticker, trade in self._open_positions.items():
                notional = trade.notional if trade.notional > 0 else self._initial_capital * trade.size
                price = self._last_prices.get(pos_ticker, trade.entry_price)
                if trade.is_long:
                    unrealised += notional * (price - trade.entry_price) / trade.entry_price
                else:
                    unrealised += notional * (trade.entry_price - price) / trade.entry_price
            self._equity_events[ts] = self._initial_capital + self._realised_pnl + unrealised

            # Periodic timing report
            if self._bar_count % 1000 == 0:
                logger.info(
                    "[perf] Bar {}: build_bar={:.3f}s, "
                    "exits={:.3f}s, execute={:.3f}s, "
                    "on_bar={:.3f}s, process={:.3f}s",
                    self._bar_count,
                    self._t_build_bar,
                    self._t_exits,
                    self._t_execute,
                    self._t_on_bar,
                    self._t_process,
                )

    # ------------------------------------------------------------------
    # Bar construction
    # ------------------------------------------------------------------

    def _build_bar(
        self, ticker: str, data: bt.AbstractDataBase, ts: datetime
    ) -> Bar:
        """Construct a Bar from OHLCV lines + side-channel features dict."""
        ts_pd = pd.Timestamp(ts)
        bar = Bar(
            ticker=ticker,
            timestamp=ts_pd,
            open=float(data.open[0]),
            high=float(data.high[0]),
            low=float(data.low[0]),
            close=float(data.close[0]),
            volume=float(data.volume[0]),
        )

        # Look up precomputed features by naive nanosecond key
        ts_ns = ts_pd.value
        feat = self._features.get(ticker, {}).get(ts_ns)
        if feat is not None:
            bar.features = feat
            # Set indicator attrs on Bar from the same dict
            for col, attr in self._indicator_cols_per_ticker.get(ticker, []):
                val = feat.get(col)
                if val is not None:
                    setattr(bar, attr, val)

        # Look up precomputed predictions by naive nanosecond key
        preds = self._pred_lookup.get(ticker, {}).get(ts_ns, {})
        bar.predictions = preds
        bar.signals = SignalsProxy(preds, self._label_to_path)

        return bar

    _pred_diag_count: int = 0
    _pred_miss_count: int = 0

    # ------------------------------------------------------------------
    # Conviction-tier helpers
    # ------------------------------------------------------------------

    def _tier_counts(self) -> dict[str, int]:
        """Count open positions by conviction tier."""
        counts: dict[str, int] = {"high": 0, "mid": 0}
        for trade in self._open_positions.values():
            tier = getattr(trade, "conviction_tier", "mid") or "mid"
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def _total_exposure(self) -> float:
        """Sum of notional exposure across all open positions."""
        return sum(
            getattr(t, "notional", self._initial_capital * t.size)
            for t in self._open_positions.values()
        )

    def _can_open_tier(self, tier: str) -> bool:
        """Check total + per-tier position limits."""
        if len(self._open_positions) >= self._max_positions:
            return False
        counts = self._tier_counts()
        if tier == "high" and counts.get("high", 0) >= self._max_high_positions:
            return False
        if tier == "mid" and counts.get("mid", 0) >= self._max_mid_positions:
            return False
        return True

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    @staticmethod
    def _find_pred(bar: Bar, *keywords: str) -> float:
        """Search bar.predictions for a value matching any keyword."""
        preds = getattr(bar, "predictions", None)
        if not isinstance(preds, dict) or not preds:
            return 0.0
        for key, val in preds.items():
            key_lower = key.lower()
            for kw in keywords:
                if kw in key_lower:
                    return float(val.prob) if hasattr(val, "prob") else float(val)
        return 0.0

    def _process_signal(self, ticker: str, signal: Signal, bar: Bar):
        """Process a new signal: queue entry, handle opposing, or ignore."""
        if signal in (Signal.BUY, Signal.SHORT):
            # Determine conviction tier from strategy
            tier = "mid"
            if hasattr(self._base, "get_conviction_tier"):
                tier = self._base.get_conviction_tier()

            if ticker in self._open_positions:
                existing = self._open_positions[ticker]

                # --- Comparison-based opposing exit ---
                top_prob = self._find_pred(bar, "top")
                bot_prob = self._find_pred(bar, "bottom", "bot")
                min_gap = 0.03

                if existing.is_long and (top_prob - bot_prob) > min_gap and existing.bars_held >= 2:
                    self._close_position(
                        ticker, bar.close, "opposing_signal", bar.timestamp
                    )
                    if self._can_open_tier(tier):
                        self._pending_signals[ticker] = signal
                elif existing.is_short and (bot_prob - top_prob) > min_gap and existing.bars_held >= 2:
                    self._close_position(
                        ticker, bar.close, "opposing_signal", bar.timestamp
                    )
                    if self._can_open_tier(tier):
                        self._pending_signals[ticker] = signal
                # Same direction already open — ignore
            else:
                if self._can_open_tier(tier):
                    self._pending_signals[ticker] = signal
                elif self._enable_preemption and tier == "high" and not self._can_open_tier(tier):
                    # Preemption: high-conviction signal but max positions reached
                    # Find worst open position and preempt if it's not worth holding
                    worst_ticker = None
                    worst_pnl = float("inf")
                    for t_ticker, t_trade in self._open_positions.items():
                        if t_trade.current_pnl_pct < worst_pnl:
                            worst_pnl = t_trade.current_pnl_pct
                            worst_ticker = t_ticker
                    if worst_ticker is not None and worst_pnl < 0.003:
                        worst_trade = self._open_positions[worst_ticker]
                        last_price = self._last_prices.get(worst_ticker, worst_trade.entry_price)
                        self._close_position(
                            worst_ticker, last_price, "preempted", bar.timestamp
                        )
                        self._pending_signals[ticker] = signal
                        logger.debug(
                            "Preempted {} (pnl={:+.4f}) for high-conviction {} {}",
                            worst_ticker, worst_pnl, signal.value, ticker,
                        )
                    else:
                        logger.debug(
                            "Position limit reached, no preemptable position for {}",
                            ticker,
                        )
                else:
                    logger.debug(
                        "Position limit reached (tier={}), skipping {} for {}",
                        tier,
                        signal.value,
                        ticker,
                    )

        elif signal == Signal.SELL:
            if ticker in self._open_positions:
                self._close_position(
                    ticker, bar.close, "signal", bar.timestamp
                )

        elif signal == Signal.COVER:
            if ticker in self._open_positions:
                trade = self._open_positions[ticker]
                if trade.is_short:
                    self._close_position(
                        ticker, bar.close, "signal", bar.timestamp
                    )

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    def _execute_signal(
        self, signal: Signal, bar: Bar, next_open: float
    ) -> Trade | None:
        """Open a new position at next bar's open with slippage."""
        if bar.ticker in self._open_positions:
            return None

        # Apply slippage
        if signal == Signal.BUY:
            fill_price = next_open * (1.0 + self._slippage_rate)
        else:
            fill_price = next_open * (1.0 - self._slippage_rate)

        # Conviction tier
        tier = "mid"
        if hasattr(self._base, "get_conviction_tier"):
            tier = self._base.get_conviction_tier()

        # Per-tier limit check (belt-and-suspenders with _process_signal)
        if not self._can_open_tier(tier):
            return None

        # Position size from *current equity* (enables compounding)
        # Full position entry
        size = self._base.get_position_size(bar)
        notional = self._portfolio_value * size
        entry_commission = max(notional * self._commission_rate, self._min_commission)

        if notional <= 0 or self._cash < notional + entry_commission:
            logger.debug(
                "Insufficient cash for {} {}", signal.value, bar.ticker
            )
            return None

        # Exposure guard: total notional must not exceed 85% of equity
        if self._total_exposure() + notional > self._max_exposure_pct * self._portfolio_value:
            logger.debug(
                "Exposure guard: {:.0f} + {:.0f} > {:.0f} (85% of {:.0f}), skipping {}",
                self._total_exposure(), notional,
                self._max_exposure_pct * self._portfolio_value,
                self._portfolio_value, bar.ticker,
            )
            return None

        ts = bar.timestamp
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        # Lock TP/SL at entry time so exit checks use the entry-bar's
        # vol regime, not whatever the current bar happens to show.
        tp_pct, sl_pct = self._base.get_tp_sl(bar)

        trade = Trade(
            ticker=bar.ticker,
            timestamp=ts,
            signal=signal,
            entry_price=fill_price,
            size=size,
            commission=entry_commission,
            conviction_tier=tier,
            notional=notional,
            entry_tp_pct=tp_pct,
            entry_sl_pct=sl_pct,
            target_size=size,
        )

        self._open_positions[bar.ticker] = trade
        self._entry_bar_idx[bar.ticker] = len(self._feeds[bar.ticker])
        self._cash -= notional + entry_commission

        self._base.on_fill(trade)

        logger.debug(
            "Opened {} {} @ {:.2f} (size={:.4f}, notional={:.0f}, tier={})",
            signal.value,
            bar.ticker,
            fill_price,
            size,
            notional,
            tier,
        )
        return trade

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exits(self, bar: Bar):
        """Exit logic with independent ablation flags.

        Each flag is checked independently:
        - enable_tranches: hard SL (-0.6%) + tranche-style max_bars
        - enable_signal_reversal: 3-bar probability reversal exit
        - enable_vol_collapse: vol_prob < 0.45 exit
        - enable_trail_stop: ATR-based trailing stop

        When none of the above fire, falls back to base strategy TP/SL/max_bars.
        """
        ticker = bar.ticker
        if ticker not in self._open_positions:
            return

        trade = self._open_positions[ticker]
        entry = trade.entry_price

        # Update peak / trough every bar
        if trade.is_long:
            if trade.peak_price is None:
                trade.peak_price = entry
            trade.peak_price = max(trade.peak_price, bar.close)
        elif trade.is_short:
            if trade.trough_price is None:
                trade.trough_price = entry
            trade.trough_price = min(trade.trough_price, bar.close)

        # Update current_pnl_pct for preemption logic
        if trade.is_long:
            trade.current_pnl_pct = (bar.close - entry) / entry
        else:
            trade.current_pnl_pct = (entry - bar.close) / entry

        # Increment bars_held once up front.  The non-tranche fallback
        # undoes this before calling base.check_exit_conditions (which
        # does its own increment).
        trade.bars_held += 1

        # --- Hard SL: -0.6% on full position (tranche mode only) ---
        if self._enable_tranches:
            if trade.is_long and bar.low <= entry * 0.994:
                self._close_position(ticker, entry * 0.994, "sl", bar.timestamp)
                return
            if trade.is_short and bar.high >= entry * 1.006:
                self._close_position(ticker, entry * 1.006, "sl", bar.timestamp)
                return

        # --- Signal reversal (independent flag) ---
        if self._enable_signal_reversal:
            top_prob = self._find_pred(bar, "top")
            bot_prob = self._find_pred(bar, "bottom", "bot")
            trade.tp_top_hist.append(top_prob)
            trade.tp_bottom_hist.append(bot_prob)

            if len(trade.tp_top_hist) == 3 and trade.bars_held >= 3:
                t0, t1, t2 = trade.tp_top_hist
                b0, b1, b2 = trade.tp_bottom_hist

                if trade.is_long:
                    if t2 > t1 > t0 and b2 < b1 < b0:
                        self._close_position(ticker, bar.close, "signal_reversal", bar.timestamp)
                        return
                elif trade.is_short:
                    if b2 > b1 > b0 and t2 < t1 < t0:
                        self._close_position(ticker, bar.close, "signal_reversal", bar.timestamp)
                        return

        # --- Vol collapse exit (independent flag) ---
        if self._enable_vol_collapse:
            vol_prob = self._find_pred(bar, "vol")
            if vol_prob > 0 and vol_prob < 0.45 and trade.bars_held >= 3:
                self._close_position(ticker, bar.close, "vol_collapse", bar.timestamp)
                return

        # --- Trail stop (independent flag) ---
        if self._enable_trail_stop:
            trail_pct = 0.008  # default fallback
            min_lock = 0.004
            if hasattr(self._base, "get_trail_params"):
                trail_pct, min_lock = self._base.get_trail_params(bar)

            pnl_pct = trade.current_pnl_pct

            if pnl_pct >= min_lock:
                if trade.is_long:
                    trail_price = trade.peak_price * (1.0 - trail_pct)
                    if bar.close < trail_price:
                        self._close_position(ticker, bar.close, "trail_stop", bar.timestamp)
                        return
                elif trade.is_short:
                    trail_price = trade.trough_price * (1.0 + trail_pct)
                    if bar.close > trail_price:
                        self._close_position(ticker, bar.close, "trail_stop", bar.timestamp)
                        return

        # --- Max bars / TP / SL fallback ---
        if self._enable_tranches:
            # Tranche max_bars: cut losers, trail aged winners
            max_bars = self._base.get_max_bars(bar)
            if trade.bars_held >= max_bars:
                if trade.current_pnl_pct <= 0:
                    self._close_position(ticker, bar.close, "max_bars", bar.timestamp)
                    return

                # Profitable and aged: activate ATR trailing stop
                trail_pct = 0.008  # default fallback
                if hasattr(self._base, "get_trail_atr_pct"):
                    trail_pct = self._base.get_trail_atr_pct(bar)

                if trade.is_long:
                    trail_price = trade.peak_price * (1.0 - trail_pct)
                    if bar.close < trail_price:
                        self._close_position(ticker, bar.close, "trail_aged", bar.timestamp)
                        return
                elif trade.is_short:
                    trail_price = trade.trough_price * (1.0 + trail_pct)
                    if bar.close > trail_price:
                        self._close_position(ticker, bar.close, "trail_aged", bar.timestamp)
                        return
        else:
            # Base strategy TP/SL/max_bars
            # Undo our bars_held increment — base will re-increment
            trade.bars_held -= 1
            result = self._base.check_exit_conditions(bar, trade)
            if result:
                reason, fill_price = result
                self._close_position(ticker, fill_price, reason, bar.timestamp)

    def _close_position(
        self,
        ticker: str,
        exit_price: float,
        reason: str,
        exit_ts: Any = None,
    ) -> Trade | None:
        """Close an open position and record the trade."""
        trade = self._open_positions.pop(ticker, None)
        if trade is None:
            return None

        # Compute bars_held from bar indices
        entry_idx = self._entry_bar_idx.pop(ticker, None)
        data = self._feeds.get(ticker)
        if entry_idx is not None and data is not None:
            trade.bars_held = len(data) - entry_idx

        # Apply slippage to exit
        if trade.is_long:
            fill_price = exit_price * (1.0 - self._slippage_rate)
        else:
            fill_price = exit_price * (1.0 + self._slippage_rate)

        trade.exit_price = fill_price
        trade.exit_reason = reason

        # Convert exit timestamp
        if exit_ts is not None:
            if hasattr(exit_ts, "to_pydatetime"):
                exit_ts = exit_ts.to_pydatetime()
        trade.exit_timestamp = exit_ts

        # Compute PnL (percentage return on the trade)
        notional = trade.notional if trade.notional > 0 else self._initial_capital * trade.size
        if trade.is_long:
            trade.pnl = (fill_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl = (trade.entry_price - fill_price) / trade.entry_price

        # Exit commission
        exit_commission = max(abs(notional) * self._commission_rate, self._min_commission)
        trade.commission += exit_commission
        trade.slippage = self._slippage_rate * 2

        # Return notional + profit/loss - exit commission to cash
        pnl_amount = trade.pnl * notional
        self._cash += notional + pnl_amount - exit_commission

        # Track running realised PnL
        net_trade_pnl = pnl_amount - trade.commission
        self._realised_pnl += net_trade_pnl
        self._portfolio_value = self._initial_capital + self._realised_pnl

        self._closed_trades.append(trade)

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

    def _add_on(self, ticker: str, bar: Bar):
        """Add second batch (50% of target size) on price confirmation."""
        trade = self._open_positions.get(ticker)
        if trade is None or trade.confirmed:
            return

        # Add-on = same size as initial entry (both are 50% of target)
        size = trade.size  # match batch 1
        addon_notional = self._portfolio_value * size

        if trade.is_long:
            fill_price = bar.close * (1.0 + self._slippage_rate)
        else:
            fill_price = bar.close * (1.0 - self._slippage_rate)

        entry_commission = max(addon_notional * self._commission_rate, self._min_commission)

        if addon_notional <= 0 or self._cash < addon_notional + entry_commission:
            return

        # Exposure guard
        if self._total_exposure() + addon_notional > self._max_exposure_pct * self._portfolio_value:
            return

        # Weighted average entry price
        old_notional = trade.notional
        total_notional = old_notional + addon_notional
        trade.entry_price = (
            trade.entry_price * old_notional + fill_price * addon_notional
        ) / total_notional
        trade.notional = total_notional
        trade.size += size
        trade.commission += entry_commission

        # Update peak/trough from new avg entry
        if trade.is_long:
            trade.peak_price = max(trade.peak_price or trade.entry_price, bar.close)
        else:
            trade.trough_price = min(trade.trough_price or trade.entry_price, bar.close)

        self._cash -= addon_notional + entry_commission

        logger.debug(
            "Add-on {} {} @ {:.2f} (new_avg={:.2f}, total_notional={:.0f})",
            trade.signal.value, ticker, fill_price,
            trade.entry_price, trade.notional,
        )

    def _close_tranche(
        self,
        ticker: str,
        exit_price: float,
        reason: str,
        exit_ts: Any = None,
        fraction: float = 0.333,
    ):
        """Close a fraction of an open position, keep remainder open."""
        trade = self._open_positions.get(ticker)
        if trade is None or trade.notional <= 0:
            return

        # Apply slippage
        if trade.is_long:
            fill_price = exit_price * (1.0 - self._slippage_rate)
        else:
            fill_price = exit_price * (1.0 + self._slippage_rate)

        tranche_notional = trade.notional * fraction
        tranche_size = trade.size * fraction
        exit_commission = max(tranche_notional * self._commission_rate, self._min_commission)

        # PnL on this tranche
        if trade.is_long:
            tranche_pnl_pct = (fill_price - trade.entry_price) / trade.entry_price
        else:
            tranche_pnl_pct = (trade.entry_price - fill_price) / trade.entry_price
        tranche_pnl_amount = tranche_pnl_pct * tranche_notional

        # Return closed tranche to cash
        self._cash += tranche_notional + tranche_pnl_amount - exit_commission

        # Track realised PnL
        self._realised_pnl += tranche_pnl_amount - exit_commission
        self._portfolio_value = self._initial_capital + self._realised_pnl

        # Shrink the remaining position
        trade.notional -= tranche_notional
        trade.size -= tranche_size
        trade.commission += exit_commission

        # Record tranche as a separate closed trade for reporting
        ts = exit_ts
        if ts is not None and hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        partial_trade = Trade(
            ticker=ticker,
            timestamp=trade.timestamp,
            signal=trade.signal,
            entry_price=trade.entry_price,
            exit_price=fill_price,
            size=tranche_size,
            commission=exit_commission,
            notional=tranche_notional,
            pnl=tranche_pnl_pct,
            exit_reason=reason,
            bars_held=trade.bars_held,
            exit_timestamp=ts,
            conviction_tier=trade.conviction_tier,
            slippage=self._slippage_rate * 2,
        )
        self._closed_trades.append(partial_trade)

        logger.debug(
            "Tranche {} {} @ {:.2f} ({:.0f}%%, pnl={:+.4f}, reason={})",
            trade.signal.value, ticker, fill_price,
            fraction * 100, tranche_pnl_pct, reason,
        )

    def _close_all_positions(self):
        """Force-close all open positions at end of backtest."""
        if not self._open_positions:
            return

        tickers = list(self._open_positions.keys())
        for ticker in tickers:
            data = self._feeds.get(ticker)
            if data is not None and len(data) > 0:
                price = float(data.close[0])
                ts = data.datetime.datetime(0)
                self._close_position(ticker, price, "end_of_backtest", ts)
                logger.info(
                    "Force-closed {} at end of backtest @ {:.2f}",
                    ticker,
                    price,
                )
