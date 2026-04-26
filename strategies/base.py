"""
Strategy base class for ApexQuant.

Inspired by QuantConnect's API style but simplified and AI-signal-native.
Strategies consume ``bar.aggregated_signal`` and ``bar.predictions``
only — they never import from ``predictors/`` directly.

Provides:

- :class:`Signal` — trade direction enum (BUY, SELL, SHORT, COVER, HOLD)
- :class:`OrderType` — order type enum (MARKET, LIMIT)
- :class:`Trade` — tracks an open or closed position
- :class:`BaseStrategy` — abstract strategy with lifecycle hooks

Usage::

    from strategies.base import BaseStrategy, Signal, Trade
    from data.bar import Bar

    class MyStrategy(BaseStrategy):
        name = "my_strategy"

        def on_bar(self, bar: Bar) -> Signal:
            if bar.aggregated_signal and bar.aggregated_signal.is_bullish:
                return Signal.BUY
            return Signal.HOLD
"""

__all__ = ["Signal", "OrderType", "Trade", "BaseStrategy"]

import abc
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from data.bar import Bar


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Signal(Enum):
    """Trade signal enumeration."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Tracks an open or closed position.

    Attributes:
        trade_id: Unique identifier for this trade.
        ticker: Stock symbol.
        timestamp: Entry timestamp.
        signal: Entry signal that opened this trade.
        entry_price: Fill price at entry.
        exit_price: Fill price at exit (``None`` while open).
        size: Position size as fraction of capital.
        commission: Total commission cost.
        slippage: Total slippage cost.
        pnl: Realised P&L (``None`` while open).
        exit_reason: Why the trade was closed.
        bars_held: Number of bars the position has been held.
        exit_timestamp: Timestamp when the trade was closed.
    """

    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ticker: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    signal: Signal = Signal.HOLD
    entry_price: float = 0.0
    exit_price: float | None = None
    size: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float | None = None
    exit_reason: str | None = None
    bars_held: int = 0
    exit_timestamp: datetime | None = None
    conviction_tier: str = ""           # "high" or "mid" (set by strategy)
    notional: float = 0.0              # actual dollar notional at entry
    entry_tp_pct: float | None = None  # TP % locked at entry (set by strategy)
    entry_sl_pct: float | None = None  # SL % locked at entry (set by strategy)
    peak_price: float | None = None    # LONG: highest close since entry
    trough_price: float | None = None  # SHORT: lowest close since entry
    confirmed: bool = False            # price confirmation triggered add-on
    tranches_exited: int = 0           # 0, 1, 2, or 3 (fully closed)
    target_size: float = 0.0          # original full target size (fraction)
    tp_top_hist: deque = field(default_factory=lambda: deque(maxlen=3))
    tp_bottom_hist: deque = field(default_factory=lambda: deque(maxlen=3))
    current_pnl_pct: float = 0.0            # updated each bar by bt_strategy

    @property
    def is_open(self) -> bool:
        """Whether this trade is still open.

        Returns:
            ``True`` if ``exit_price`` is ``None``.
        """
        return self.exit_price is None

    @property
    def is_long(self) -> bool:
        """Whether this is a long position.

        Returns:
            ``True`` if entry signal was BUY.
        """
        return self.signal == Signal.BUY

    @property
    def is_short(self) -> bool:
        """Whether this is a short position.

        Returns:
            ``True`` if entry signal was SHORT.
        """
        return self.signal == Signal.SHORT

    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else f"CLOSED pnl={self.pnl:.4f}"
        return (
            f"Trade({self.trade_id}, {self.ticker}, {self.signal.value}, "
            f"entry={self.entry_price:.2f}, bars={self.bars_held}, {status})"
        )


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------

class BaseStrategy(abc.ABC):
    """Abstract strategy with lifecycle hooks and AI-signal-native interface.

    Subclasses must implement :meth:`on_bar`.  All other methods have
    sensible defaults that can be overridden for custom logic.

    Attributes:
        name: Unique strategy identifier.
        config: Full CONFIG dict.
        positions: Dict mapping ticker to the current open :class:`Trade`.
        signal_buffer: Latest prediction per label (updated asynchronously).
        trade_history: List of all closed trades.
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the strategy.

        Args:
            config: Full CONFIG dict.
        """
        self.config = config
        self.positions: dict[str, Trade] = {}
        self.signal_buffer: dict[str, Any] = {}
        self.trade_history: list[Trade] = []
        self._initialized: bool = False

        logger.info("Strategy '{}' initialised", self.name)

    # ------------------------------------------------------------------
    # Lifecycle hooks (override optionally)
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once before the first bar is processed.

        Override to perform any setup (e.g. pre-compute indicators,
        load saved state).
        """
        self._initialized = True
        logger.debug("Strategy '{}' started", self.name)

    def on_end(self) -> None:
        """Called once after the last bar is processed.

        Override to perform any cleanup or final reporting.
        """
        logger.debug(
            "Strategy '{}' ended — {} trades total",
            self.name,
            len(self.trade_history),
        )

    def on_fill(self, trade: Trade) -> None:
        """Called when a trade is filled (opened or closed).

        Override for custom fill logging, notifications, or
        position-tracking logic.

        Args:
            trade: The trade that was just filled.
        """
        logger.debug(
            "Fill: {} {} @ {:.2f} (size={:.4f})",
            trade.signal.value,
            trade.ticker,
            trade.entry_price,
            trade.size,
        )

    def on_regime_change(self, regime: str) -> None:
        """Called when the market regime changes.

        Override to adjust strategy parameters based on the
        detected regime (e.g. ``"trending_up"``, ``"mean_reverting"``).

        Args:
            regime: New regime label.
        """
        logger.info("Strategy '{}' regime change: {}", self.name, regime)

    # ------------------------------------------------------------------
    # Prediction updates (different cadence from bars)
    # ------------------------------------------------------------------

    def on_prediction(self, pred: Any) -> None:
        """Called when a new prediction arrives.

        Predictions may arrive at a different frequency from bars.
        The default implementation stores them in ``signal_buffer``.

        Args:
            pred: A :class:`PredictionResult` (or any object with a
                ``label`` attribute).
        """
        label = getattr(pred, "label", None)
        if label is not None:
            self.signal_buffer[label] = pred
            logger.debug("Buffered prediction: {}", label)

    # ------------------------------------------------------------------
    # Core method — must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def on_bar(self, bar: Bar) -> Signal:
        """Process a new bar and return a trading signal.

        This is the only method a subclass **must** implement.

        Args:
            bar: The current bar with technical indicators,
                predictions, and aggregated_signal populated.

        Returns:
            A :class:`Signal` indicating the desired action.
        """
        ...

    # ------------------------------------------------------------------
    # Position sizing — override for custom logic
    # ------------------------------------------------------------------

    def get_position_size(self, bar: Bar) -> float:
        """Compute position size for a trade.

        Default logic scales the base size by signal strength and
        inversely by volatility probability.

        Args:
            bar: The current bar.

        Returns:
            Position size as a fraction of capital.
        """
        base = self.config.get("backtest", {}).get("position_size", 0.10)

        if bar.aggregated_signal is not None:
            strength = getattr(bar.aggregated_signal, "strength", 1.0)
            vol_prob = bar.get_prob("vol_prob", 0.5)
            return base * max(0.1, strength) * (1.0 / (1.0 + vol_prob))

        return base

    # ------------------------------------------------------------------
    # TP / SL — override for dynamic logic
    # ------------------------------------------------------------------

    def get_tp_sl(self, bar: Bar) -> tuple[float, float]:
        """Return take-profit and stop-loss levels.

        Default reads from ``config["model"]``.  Override for
        dynamic TP/SL based on ATR, volatility, or signal strength.

        Args:
            bar: The current bar.

        Returns:
            Tuple of ``(take_profit, stop_loss)`` as decimal fractions.
        """
        model_cfg = self.config.get("model", {})
        tp = model_cfg.get("meta_tp", 0.005)
        sl = model_cfg.get("meta_sl", 0.003)
        return tp, sl

    # ------------------------------------------------------------------
    # Max bars — override for dynamic logic
    # ------------------------------------------------------------------

    def get_max_bars(self, bar: Bar) -> int:
        """Return maximum bars to hold a position.

        Default reads from ``config["model"]["meta_mb"]``.

        Args:
            bar: The current bar.

        Returns:
            Maximum number of bars.
        """
        return int(self.config.get("model", {}).get("meta_mb", 48))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def has_position(self, ticker: str) -> bool:
        """Check if there is an open position for a ticker.

        Args:
            ticker: Stock symbol.

        Returns:
            ``True`` if an open position exists.
        """
        return ticker in self.positions

    def get_position(self, ticker: str) -> Trade | None:
        """Get the open position for a ticker, if any.

        Args:
            ticker: Stock symbol.

        Returns:
            The open :class:`Trade`, or ``None``.
        """
        return self.positions.get(ticker)

    def open_position(self, bar: Bar, signal: Signal) -> Trade:
        """Open a new position at the current bar's close price.

        Args:
            bar: The current bar.
            signal: Entry signal (BUY or SHORT).

        Returns:
            The newly created :class:`Trade`.
        """
        trade = Trade(
            ticker=bar.ticker,
            timestamp=bar.timestamp.to_pydatetime()
            if hasattr(bar.timestamp, "to_pydatetime")
            else bar.timestamp,
            signal=signal,
            entry_price=bar.close,
            size=self.get_position_size(bar),
        )
        self.positions[bar.ticker] = trade
        self.on_fill(trade)
        return trade

    def close_position(
        self, ticker: str, exit_price: float, reason: str = "signal"
    ) -> Trade | None:
        """Close an open position.

        Args:
            ticker: Stock symbol.
            exit_price: Fill price at exit.
            reason: Exit reason (``"tp"``, ``"sl"``, ``"max_bars"``,
                ``"signal"``).

        Returns:
            The closed :class:`Trade`, or ``None`` if no position was open.
        """
        trade = self.positions.pop(ticker, None)
        if trade is None:
            return None

        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Compute PnL
        if trade.is_long:
            trade.pnl = (exit_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl = (trade.entry_price - exit_price) / trade.entry_price

        # Subtract costs
        backtest_cfg = self.config.get("backtest", {})
        trade.commission = backtest_cfg.get("commission", 0.001) * 2
        trade.slippage = backtest_cfg.get("slippage", 0.0005) * 2
        trade.pnl -= trade.commission + trade.slippage

        self.trade_history.append(trade)

        logger.debug(
            "Closed {} {} @ {:.2f} (pnl={:+.4f}, reason={})",
            trade.signal.value,
            ticker,
            exit_price,
            trade.pnl,
            reason,
        )
        return trade

    def check_exit_conditions(
        self, bar: Bar, trade: Trade,
    ) -> tuple[str, float] | None:
        """Check if an open position should be closed.

        Uses intrabar HIGH/LOW to detect SL/TP triggers realistically.
        When triggered, the fill price is the SL or TP level itself
        (simulating a stop/limit order), not bar.close.

        For max-bars exits, fills at bar.close (market-on-close).

        Args:
            bar: The current bar.
            trade: The open trade to check.

        Returns:
            Tuple of ``(reason, fill_price)`` or ``None`` if the
            position should remain open.
        """
        trade.bars_held += 1
        # Use TP/SL locked at entry time (if set), otherwise fall back
        # to current-bar calculation.
        if trade.entry_tp_pct is not None and trade.entry_sl_pct is not None:
            tp_pct, sl_pct = trade.entry_tp_pct, trade.entry_sl_pct
        else:
            tp_pct, sl_pct = self.get_tp_sl(bar)
        max_bars = self.get_max_bars(bar)

        entry = trade.entry_price

        if trade.is_long:
            tp_price = entry * (1.0 + tp_pct)
            sl_price = entry * (1.0 - sl_pct)
            sl_hit = bar.low <= sl_price
            tp_hit = bar.high >= tp_price
        else:
            tp_price = entry * (1.0 - tp_pct)
            sl_price = entry * (1.0 + sl_pct)
            sl_hit = bar.high >= sl_price
            tp_hit = bar.low <= tp_price

        if sl_hit and tp_hit:
            return ("sl", sl_price)
        if sl_hit:
            return ("sl", sl_price)
        if tp_hit:
            return ("tp", tp_price)

        if trade.bars_held >= max_bars:
            return ("max_bars", bar.close)

        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"positions={len(self.positions)}, "
            f"history={len(self.trade_history)})"
        )
