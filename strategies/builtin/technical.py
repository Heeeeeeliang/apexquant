"""
Pure EMA + RSI baseline strategy for ApexQuant.

Mirrors the QuantConnect backtest logic (Sharpe = -1.12 reference).
Uses only ``bar.ema_8``, ``bar.ema_21``, ``bar.rsi_14`` — no AI signals.
This serves as the non-AI control baseline for comparison.

Usage::

    from strategies.builtin.technical import TechnicalStrategy
    from config.default import CONFIG

    strat = TechnicalStrategy(CONFIG)
    signal = strat.on_bar(bar)
"""

__all__ = ["TechnicalStrategy"]

from collections import defaultdict, deque
from typing import Any

import numpy as np
from loguru import logger

from data.bar import Bar
from strategies.base import BaseStrategy, Signal, Trade


class TechnicalStrategy(BaseStrategy):
    """Pure EMA crossover + RSI filter baseline strategy.

    Entry logic:

    - **BUY**: EMA-fast crosses above EMA-slow AND RSI < ``rsi_buy``
      (not overbought).
    - **SHORT**: EMA-fast crosses below EMA-slow AND RSI > ``rsi_sell``
      (not oversold).

    Exit logic:

    - **EMA crossover exit** (checked first): close LONG when fast
      crosses below slow; close SHORT when fast crosses above slow.
    - Then TP / SL / max-bars from :class:`BaseStrategy`.

    Attributes:
        name: ``"technical"``.
        ema_fast: Fast EMA period (default 8).
        ema_slow: Slow EMA period (default 21).
        rsi_buy: RSI threshold for buy confirmation.
        rsi_sell: RSI threshold for sell confirmation.
    """

    name: str = "technical"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the technical strategy.

        Args:
            config: Full CONFIG dict.  Reads ``config["technical"]``
                for EMA/RSI parameters.
        """
        super().__init__(config)

        tech_cfg = config.get("technical", {})
        self.ema_fast: int = tech_cfg.get("ema_fast", 8)
        self.ema_slow: int = tech_cfg.get("ema_slow", 21)
        self.rsi_buy: int = tech_cfg.get("rsi_buy", 70)
        self.rsi_sell: int = tech_cfg.get("rsi_sell", 30)

        # Vol-aware position sizing (off by default)
        self._vol_sizing: bool = tech_cfg.get("vol_sizing", False)
        self._vol_lookback: int = tech_cfg.get("vol_lookback", 20)
        self._vol_high_pct: float = tech_cfg.get("vol_high_pct", 0.60)
        self._vol_high_size: float = tech_cfg.get("vol_high_size", 0.20)
        self._vol_low_size: float = tech_cfg.get("vol_low_size", 0.05)

        # Track previous EMA values per ticker for crossover detection
        self._prev_ema_fast: dict[str, float] = {}
        self._prev_ema_slow: dict[str, float] = {}

        # Rolling return buffer for realized vol calculation
        self._return_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._vol_lookback + 1)
        )
        self._prev_close: dict[str, float] = {}

        logger.info(
            "TechnicalStrategy: EMA({}/{}), RSI buy<{} sell>{}, vol_sizing={}",
            self.ema_fast,
            self.ema_slow,
            self.rsi_buy,
            self.rsi_sell,
            self._vol_sizing,
        )

    def on_bar(self, bar: Bar) -> Signal:
        """Process a bar using EMA crossover + RSI filter.

        Requires ``bar.ema_8``, ``bar.ema_21``, and ``bar.rsi_14``
        to be populated.  Returns HOLD if any indicator is missing.
        """
        # Track per-ticker returns for realized vol calculation
        ticker = bar.ticker
        prev_c = self._prev_close.get(ticker)
        self._prev_close[ticker] = bar.close
        if prev_c is not None and prev_c > 0:
            self._return_history[ticker].append(
                (bar.close - prev_c) / prev_c
            )

        ema_f = bar.ema_8
        ema_s = bar.ema_21
        rsi = bar.rsi_14

        if ema_f is None or ema_s is None or rsi is None:
            return Signal.HOLD

        ticker = bar.ticker
        prev_fast = self._prev_ema_fast.get(ticker)
        prev_slow = self._prev_ema_slow.get(ticker)

        # Store current values for next bar's crossover detection
        self._prev_ema_fast[ticker] = ema_f
        self._prev_ema_slow[ticker] = ema_s

        # Need previous values to detect crossover
        if prev_fast is None or prev_slow is None:
            return Signal.HOLD

        signal = Signal.HOLD

        # Bullish crossover: fast crosses above slow
        was_below = prev_fast <= prev_slow
        now_above = ema_f > ema_s

        if was_below and now_above and rsi < self.rsi_buy:
            signal = Signal.BUY
            logger.debug(
                "{}: EMA bullish crossover (fast={:.2f}>slow={:.2f}), "
                "RSI={:.1f}<{} → BUY",
                ticker, ema_f, ema_s, rsi, self.rsi_buy,
            )

        # Bearish crossover: fast crosses below slow
        was_above = prev_fast >= prev_slow
        now_below = ema_f < ema_s

        if was_above and now_below and rsi > self.rsi_sell:
            signal = Signal.SHORT
            logger.debug(
                "{}: EMA bearish crossover (fast={:.2f}<slow={:.2f}), "
                "RSI={:.1f}>{} → SHORT",
                ticker, ema_f, ema_s, rsi, self.rsi_sell,
            )

        return signal

    # ------------------------------------------------------------------
    # EMA crossover exit + TP/SL overrides
    # ------------------------------------------------------------------

    def check_exit_conditions(
        self, bar: Bar, trade: "Trade",
    ) -> tuple[str, float] | None:
        """Check exits: EMA crossover exit first, then TP/SL/max_bars.

        LONG positions close when EMA-fast crosses below EMA-slow.
        SHORT positions close when EMA-fast crosses above EMA-slow.
        """
        ema_f = bar.ema_8
        ema_s = bar.ema_21

        ticker = bar.ticker
        prev_fast = self._prev_ema_fast.get(ticker)
        prev_slow = self._prev_ema_slow.get(ticker)

        # EMA crossover exit (checked before TP/SL)
        if ema_f is not None and ema_s is not None and prev_fast is not None and prev_slow is not None:
            if trade.is_long:
                was_above = prev_fast >= prev_slow
                now_below = ema_f < ema_s
                if was_above and now_below:
                    trade.bars_held += 1
                    return ("ema_cross_exit", bar.close)

            elif trade.is_short:
                was_below = prev_fast <= prev_slow
                now_above = ema_f > ema_s
                if was_below and now_above:
                    trade.bars_held += 1
                    return ("ema_cross_exit", bar.close)

        # Fall through to base TP/SL/max_bars
        return super().check_exit_conditions(bar, trade)

    def get_tp_sl(self, bar: Bar) -> tuple[float, float]:
        """TP at 2% and emergency-only SL at 5%.

        The primary exit is EMA crossover, not SL.  The 5% SL only
        catches catastrophic moves (earnings gaps, flash crashes).
        """
        tech_cfg = self.config.get("technical", {})
        tp = tech_cfg.get("tp", 0.999)
        sl = tech_cfg.get("sl", 0.050)
        return tp, sl

    def _realized_vol(self, ticker: str) -> float | None:
        """Realized vol (std of returns) over lookback window."""
        rets = self._return_history.get(ticker)
        if rets is None or len(rets) < self._vol_lookback:
            return None
        return float(np.std(list(rets)))

    def get_position_size(self, bar: Bar) -> float:
        """Position size for technical strategy.

        Default: fixed ``config["technical"]["position_size"]`` (10%).
        With vol_sizing=True: two-tier sizing based on realized vol.
            High vol (top 40% of cross-sectional vol) → vol_high_size (20%)
            Low vol → vol_low_size (5%)
        """
        if not self._vol_sizing:
            return self.config.get("technical", {}).get("position_size", 0.10)

        # Compute cross-sectional realized vol for all tickers with data
        all_vols = {}
        for tk, rets in self._return_history.items():
            if len(rets) >= self._vol_lookback:
                all_vols[tk] = float(np.std(list(rets)))

        this_vol = all_vols.get(bar.ticker)
        if this_vol is None or len(all_vols) < 2:
            return self._vol_low_size

        # Percentile rank: top 40% = above 60th percentile
        threshold = float(np.percentile(list(all_vols.values()), (1.0 - self._vol_high_pct) * 100))
        if this_vol >= threshold:
            return self._vol_high_size
        return self._vol_low_size
