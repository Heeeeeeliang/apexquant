"""
Default AI strategy for ApexQuant.

Consumes the :class:`AggregatedSignal` produced by
:class:`LearnedAggregator`.  Requires no knowledge of individual
predictors — all signal interpretation is done via three configurable
thresholds on direction, strength, and confidence.

Usage::

    from strategies.builtin.ai_strategy import AIStrategy
    from config.default import CONFIG

    strat = AIStrategy(CONFIG)
    signal = strat.on_bar(bar)

    # Or via factory
    strat = AIStrategy.from_config(CONFIG)
"""

from __future__ import annotations

__all__ = ["AIStrategy"]

from collections import defaultdict, deque
from typing import Any

import numpy as np
from loguru import logger

from data.bar import Bar
from strategies.base import BaseStrategy, Signal, Trade


class AIStrategy(BaseStrategy):
    """Default AI-signal-native strategy.

    Acts only when the aggregated signal passes **all three** gate
    conditions: direction magnitude, strength, and confidence.

    Attributes:
        name: ``"ai_strategy"``.
        direction_threshold: Minimum ``abs(direction)`` to act.
        strength_threshold: Minimum ``strength`` to act.
        confidence_threshold: Minimum ``confidence`` to act.
    """

    name: str = "ai_strategy"

    def __init__(
        self,
        config: dict[str, Any],
        direction_threshold: float = 0.3,
        strength_threshold: float = 0.50,
        confidence_threshold: float = 0.55,
        bottom_threshold: float | None = None,
        use_confirmation: bool = False,
        dynamic_execution: bool = True,
        vol_threshold: float = 0.50,
        trend_bypass_period: int = 80,
        trend_bypass_pct: float = 0.05,
        trend_bypass_min_vol: float = 0.35,
        direction_filter: bool = True,
    ) -> None:
        super().__init__(config)
        self.direction_threshold = direction_threshold
        self.strength_threshold = strength_threshold
        self.confidence_threshold = confidence_threshold
        self.bottom_threshold = bottom_threshold if bottom_threshold is not None else strength_threshold
        self.use_confirmation = use_confirmation
        self.dynamic_execution = dynamic_execution
        self.direction_filter = direction_filter

        # Vol gate parameters
        self.vol_threshold = vol_threshold
        self.trend_bypass_period = trend_bypass_period
        self.trend_bypass_pct = trend_bypass_pct
        self.trend_bypass_min_vol = trend_bypass_min_vol

        # ATR-based exit parameters (from config)
        strat_cfg = config.get("strategy", {})
        self._tp_atr_mult: float = strat_cfg.get("tp_atr_mult", 1.5)
        self._sl_atr_mult: float = strat_cfg.get("sl_atr_mult", 0.75)
        self._max_bars_held: int = strat_cfg.get("max_bars_held", 24)

        # Per-bar state: set during on_bar, read by get_position_size
        self._current_meta_prob: float = 0.0
        self._current_vol_prob: float = 0.5
        self._current_top_prob: float = 0.0
        self._current_bot_prob: float = 0.0

        # ATR lookup — injected by bt_runner before backtest starts
        self._atr_lookup: dict[str, Any] = {}

        # Per-ticker rolling close buffer for trend detection
        self._close_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=trend_bypass_period + 1)
        )

        logger.info(
            "AIStrategy: top>{:.2f}, bot>{:.2f}, dynamic={}, confirm={}, "
            "vol_gate={:.2f}, trend_bypass={}/{:.1%}, "
            "exit: tp_atr={:.2f}, sl_atr={:.2f}, max_bars={}",
            strength_threshold,
            self.bottom_threshold,
            dynamic_execution,
            use_confirmation,
            vol_threshold,
            trend_bypass_period,
            trend_bypass_pct,
            self._tp_atr_mult,
            self._sl_atr_mult,
            self._max_bars_held,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AIStrategy:
        """Create an AIStrategy with thresholds from config."""
        strat_cfg = config.get("strategy", {})
        return cls(
            config=config,
            direction_threshold=strat_cfg.get("direction_threshold", 0.3),
            strength_threshold=strat_cfg.get("strength_threshold", 0.50),
            confidence_threshold=strat_cfg.get("confidence_threshold", 0.55),
            bottom_threshold=strat_cfg.get("bottom_threshold", 0.50),
            use_confirmation=strat_cfg.get("use_confirmation", False),
            dynamic_execution=strat_cfg.get("dynamic_execution", True),
            vol_threshold=strat_cfg.get("vol_threshold", 0.50),
            trend_bypass_period=strat_cfg.get("trend_bypass_period", 80),
            trend_bypass_pct=strat_cfg.get("trend_bypass_pct", 0.05),
            trend_bypass_min_vol=strat_cfg.get("trend_bypass_min_vol", 0.35),
            direction_filter=strat_cfg.get("direction_filter", True),
        )

    # ------------------------------------------------------------------
    # TP / SL — ATR-based with conviction scaling
    # ------------------------------------------------------------------

    def get_tp_sl(self, bar: Bar) -> tuple[float, float]:
        """ATR-based TP/SL.

        Looks up pre-computed ATR (as fraction of price) for the
        ticker/timestamp from ``self._atr_lookup`` (injected by
        bt_runner).  Multiplies by tp_atr_mult / sl_atr_mult.

        Falls back to fixed conviction-tiered levels when ATR is
        unavailable.
        """
        atr_pct = self._get_atr_pct(bar)
        if atr_pct is not None and atr_pct > 0:
            tp = atr_pct * self._tp_atr_mult
            sl = atr_pct * self._sl_atr_mult
            # Clamp to sane range
            tp = max(0.003, min(tp, 0.05))
            sl = max(0.002, min(sl, 0.03))
            return tp, sl

        # Fallback: conviction-tiered fixed levels
        vp = self._current_vol_prob
        if vp > 0.7:
            return 0.020, 0.006
        return 0.008, 0.004

    def _get_atr_pct(self, bar: Bar) -> float | None:
        """Look up pre-computed ATR fraction for this bar's ticker/timestamp."""
        atr_series = self._atr_lookup.get(bar.ticker)
        if atr_series is None:
            return None
        ts = bar.timestamp
        if hasattr(ts, "tz") and ts.tz is not None:
            ts = ts.tz_localize(None)
        try:
            val = atr_series.get(ts)
            if val is not None and not np.isnan(val):
                return float(val)
        except (KeyError, TypeError):
            pass
        # Nearest lookup fallback
        try:
            idx = atr_series.index.get_indexer([ts], method="ffill")[0]
            if idx >= 0:
                val = atr_series.iloc[idx]
                if not np.isnan(val):
                    return float(val)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Trailing stop — conviction-tiered
    # ------------------------------------------------------------------

    def get_trail_params(self, bar: Bar) -> tuple[float, float]:
        """Return (trail_distance_pct, min_profit_lock) for current conviction.

        HIGH conviction (vol_prob > 0.7):
            trail = ATR × 2.0,  min_profit_lock = 0.8%
        MID conviction:
            trail = ATR × 1.5,  min_profit_lock = 0.4%

        Returns fractions of price.
        """
        tier = self.get_conviction_tier()
        if tier == "high":
            atr_mult = 2.0
            min_lock = 0.008
            fallback = 0.010
        else:
            atr_mult = 1.5
            min_lock = 0.004
            fallback = 0.008

        atr_pct = self._get_atr_pct(bar)
        if atr_pct is not None and atr_pct > 0:
            trail = atr_pct * atr_mult
            trail = max(0.003, min(trail, 0.03))
        else:
            trail = fallback

        return trail, min_lock

    def get_trail_atr_pct(self, bar: Bar) -> float:
        """Return trailing stop distance (backward compat for trail_aged)."""
        trail, _ = self.get_trail_params(bar)
        return trail

    # ------------------------------------------------------------------
    # Position sizing (conviction-based)
    # ------------------------------------------------------------------

    def get_conviction_tier(self) -> str:
        """Return conviction tier based on cached vol_prob."""
        return "high" if self._current_vol_prob > 0.7 else "mid"

    def get_position_size(self, bar: Bar) -> float:
        """Conviction-based sizing (fraction of *current* equity).

        vol_prob > 0.7   → 30% of equity (high conviction)
        vol_prob 0.5–0.7 →  8% of equity (mid conviction)
        vol_prob ≤ 0.5   → filtered out in on_bar
        """
        if not self.dynamic_execution:
            return super().get_position_size(bar)

        vp = self._current_vol_prob

        if vp > 0.7:
            return 0.30
        else:
            return 0.08

    def get_max_bars(self, bar: Bar) -> int:
        """Time-based stop: evict zombie positions."""
        return self._max_bars_held

    # ------------------------------------------------------------------
    # Vol gate with trend momentum bypass
    # ------------------------------------------------------------------

    def _vol_gate_passes(self, bar: Bar, vol_prob: float) -> bool:
        """Check whether the vol gate allows this trade.

        Passes if EITHER:
        1. vol_prob > vol_threshold (original condition), OR
        2. Trend bypass: price moved > trend_bypass_pct over the
           last trend_bypass_period bars AND vol_prob exceeds a
           weak minimum (trend_bypass_min_vol) to avoid dead markets.
        """
        # Original gate
        if vol_prob > self.vol_threshold:
            return True

        # Trend bypass — requires enough close history
        history = self._close_history.get(bar.ticker)
        if (
            history is not None
            and len(history) >= self.trend_bypass_period
            and vol_prob > self.trend_bypass_min_vol
        ):
            past_close = history[-self.trend_bypass_period]
            current_close = bar.close
            if past_close > 0:
                price_change = abs(current_close - past_close) / past_close
                if price_change > self.trend_bypass_pct:
                    return True

        return False

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> Signal:
        """Process a bar and return a signal based on AI output.

        Supports two modes:

        1. **Aggregated signal** — if ``bar.aggregated_signal`` is set
           (i.e. the full three-layer cascade ran), use the direction /
           strength / confidence gates.
        2. **Direct predictions** — if only individual model predictions
           are available (``bar.predictions``), interpret them directly.
           Requires both a top-detection and a bottom-detection model;
           if either is missing, returns HOLD (safe default).
        """
        # Track close history for trend bypass
        self._close_history[bar.ticker].append(bar.close)

        # --- Mode 1: aggregated signal (full three-layer cascade) ---
        agg = bar.aggregated_signal
        if agg is not None:
            return self._from_aggregated(bar, agg)

        # --- Mode 2: direct prediction lookup ---
        return self._from_predictions(bar)

    def _from_aggregated(self, bar: Bar, agg) -> Signal:
        """Signal from the learned aggregator output."""
        confidence = getattr(agg, "confidence", 0.0)
        strength = getattr(agg, "strength", 0.0)
        direction = getattr(agg, "direction", 0.0)

        if confidence < self.confidence_threshold:
            return Signal.HOLD
        if strength < self.strength_threshold:
            return Signal.HOLD

        if direction > self.direction_threshold:
            return Signal.BUY
        if direction < -self.direction_threshold:
            return Signal.SHORT
        return Signal.HOLD

    _diag_count: int = 0

    def _from_predictions(self, bar: Bar) -> Signal:
        """Signal from raw model predictions when aggregator is absent.

        Looks for top/bottom turning-point probabilities in
        ``bar.predictions``.  Both must be present to generate a trade
        signal; if either is missing, returns HOLD.
        """
        # Diagnostic: log prediction contents for the first 10 bars
        if AIStrategy._diag_count < 10:
            AIStrategy._diag_count += 1
            preds = getattr(bar, "predictions", None)
            logger.info(
                "[diag] on_bar #{} {}: bar.predictions keys={}, values={}",
                AIStrategy._diag_count,
                bar.ticker,
                list(preds.keys()) if isinstance(preds, dict) else type(preds).__name__,
                {k: (f"{v:.4f}" if isinstance(v, float) else str(v))
                 for k, v in preds.items()}
                if isinstance(preds, dict) else "N/A",
            )

        top_prob = self._find_prediction(bar, "top")
        bot_prob = self._find_prediction(bar, "bottom", "bot")

        # Require both models to avoid one-sided bias
        if top_prob is None or bot_prob is None:
            return Signal.HOLD

        top_threshold = self.strength_threshold
        bot_threshold = self.bottom_threshold

        # Determine raw AI signal
        ai_signal = Signal.HOLD
        meta_prob = 0.0
        if self.direction_filter:
            # Default: require directional dominance (top > bot or bot > top)
            if top_prob > top_threshold and top_prob > bot_prob:
                ai_signal = Signal.SHORT
                meta_prob = top_prob
            elif bot_prob > bot_threshold and bot_prob > top_prob:
                ai_signal = Signal.BUY
                meta_prob = bot_prob
        else:
            # No direction filter: any signal above threshold triggers
            top_fires = top_prob > top_threshold
            bot_fires = bot_prob > bot_threshold
            if top_fires and bot_fires:
                # Both above threshold — take stronger
                if top_prob >= bot_prob:
                    ai_signal = Signal.SHORT
                    meta_prob = top_prob
                else:
                    ai_signal = Signal.BUY
                    meta_prob = bot_prob
            elif top_fires:
                ai_signal = Signal.SHORT
                meta_prob = top_prob
            elif bot_fires:
                ai_signal = Signal.BUY
                meta_prob = bot_prob

        if ai_signal == Signal.HOLD:
            return Signal.HOLD

        # Cache per-bar state for get_position_size
        self._current_meta_prob = meta_prob
        self._current_top_prob = top_prob if top_prob is not None else 0.0
        self._current_bot_prob = bot_prob if bot_prob is not None else 0.0
        vol_prob = self._find_prediction(bar, "vol")
        self._current_vol_prob = vol_prob if vol_prob is not None else 0.5

        # Vol gate with trend momentum bypass
        if self.dynamic_execution and not self._vol_gate_passes(bar, self._current_vol_prob):
            return Signal.HOLD

        # Post-filter: EMA trend + RSI confirmation (if enabled)
        if self.use_confirmation:
            confirmed = self._confirm_signal(bar, ai_signal)
            if not confirmed:
                return Signal.HOLD

        if ai_signal == Signal.SHORT:
            logger.debug(
                "{}: top_prob={:.3f} > {:.2f} → SHORT",
                bar.ticker, top_prob, top_threshold,
            )
        else:
            logger.debug(
                "{}: bot_prob={:.3f} > {:.2f} → BUY",
                bar.ticker, bot_prob, bot_threshold,
            )
        return ai_signal

    @staticmethod
    def _confirm_signal(bar: Bar, signal: Signal) -> bool:
        """Post-filter: EMA trend direction + RSI range confirmation."""
        ema_f = bar.ema_8
        ema_s = bar.ema_21
        rsi = bar.rsi_14

        if ema_f is None or ema_s is None:
            return True

        if signal == Signal.BUY and ema_f <= ema_s:
            return False
        if signal == Signal.SHORT and ema_f >= ema_s:
            return False

        if rsi is not None and (rsi <= 30 or rsi >= 70):
            return False

        return True

    @staticmethod
    def _find_prediction(bar: Bar, *keywords: str) -> float | None:
        """Search bar.predictions for a value matching any keyword."""
        preds = getattr(bar, "predictions", None)
        if not isinstance(preds, dict) or not preds:
            return None
        for key, val in preds.items():
            key_lower = key.lower()
            for kw in keywords:
                if kw in key_lower:
                    return float(val.prob) if hasattr(val, "prob") else float(val)
        return None
