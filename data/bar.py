"""
Bar — the single data carrier between all ApexQuant layers.

Every component (predictors, aggregator, strategies, backtesting)
reads from and writes to Bar instances.  Strategies only ever access
``bar.aggregated_signal`` and ``bar.predictions`` — they never import
from ``predictors/`` directly.

Usage::

    from data.bar import Bar
    bar = Bar(
        timestamp=pd.Timestamp("2023-01-03"),
        ticker="AAPL",
        open=130.0, high=132.0, low=129.5, close=131.5,
        volume=1_000_000,
    )
    bar.predictions["volatility"] = pred_result
    print(bar.get_prob("volatility"))  # 0.82
"""

__all__ = ["Bar"]

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Bar:
    """Single OHLCV bar augmented with technical indicators, predictions, and signals.

    Attributes:
        ticker: Stock symbol (e.g. ``"AAPL"``).
        timestamp: Bar timestamp.
        open: Opening price.
        high: High price.
        low: Low price.
        close: Closing price.
        volume: Trading volume.

        ema_8: 8-period EMA.
        ema_21: 21-period EMA.
        ema_50: 50-period EMA.
        rsi_14: 14-period RSI.
        macd: MACD line.
        macd_signal: MACD signal line.
        macd_hist: MACD histogram.
        atr_14: 14-period ATR.
        bb_upper: Bollinger upper band.
        bb_lower: Bollinger lower band.
        bb_mid: Bollinger middle band.
        volume_ratio: Volume relative to 20-bar SMA.
        vwap: Volume-weighted average price.
        obv: On-balance volume.
        adx_14: 14-period ADX.
        stoch_k: Stochastic %K.
        stoch_d: Stochastic %D.
        willr_14: Williams %R (14-period).
        cci_20: Commodity Channel Index (20-period).
        mfi_14: Money Flow Index (14-period).

        features: Dict of additional computed feature values.
        predictions: Dict of per-predictor outputs keyed by label.
        aggregated_signal: Final combined AggregatedSignal from the aggregator.
        meta: Arbitrary metadata (e.g. split label).
    """

    # Core OHLCV
    ticker: str = ""
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now())
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    # Technical indicators (populated by feature pipeline)
    ema_8: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    bb_mid: float | None = None
    volume_ratio: float | None = None
    vwap: float | None = None
    obv: float | None = None
    adx_14: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    willr_14: float | None = None
    cci_20: float | None = None
    mfi_14: float | None = None

    # AI signal fields (populated by predictor pipeline)
    features: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    aggregated_signal: Any = None
    signals: Any = None  # SignalsProxy instance (set by backtest engine)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- Helper methods -------------------------------------------------------

    def get_prediction(self, label: str) -> Any:
        """Get a PredictionResult by label.

        Args:
            label: Predictor label (e.g. ``"vol_prob"``).

        Returns:
            The PredictionResult, or ``None`` if not present.
        """
        return self.predictions.get(label)

    def get_prob(self, label: str, default: float = 0.5) -> float:
        """Get a predictor's probability, with staleness check.

        If the prediction is missing or stale, returns *default*.

        Args:
            label: Predictor label.
            default: Fallback probability.

        Returns:
            Probability in ``[0, 1]``, or *default*.
        """
        result = self.predictions.get(label)
        if result is None:
            return default
        # Support both raw floats (legacy) and PredictionResult objects
        if isinstance(result, (int, float)):
            return float(result)
        if hasattr(result, "is_stale") and result.is_stale:
            return default
        if hasattr(result, "prob"):
            return float(result.prob)
        return default

    # -- Properties -----------------------------------------------------------

    @property
    def mid(self) -> float:
        """Mid-price (high + low) / 2."""
        return (self.high + self.low) / 2.0

    @property
    def typical(self) -> float:
        """Typical price (high + low + close) / 3."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def returns(self) -> float | None:
        """Return stored in features, if available."""
        return self.features.get("returns")

    def __repr__(self) -> str:
        agg_str = "None"
        if self.aggregated_signal is not None:
            if hasattr(self.aggregated_signal, "direction"):
                agg_str = f"dir={self.aggregated_signal.direction:+.3f}"
            else:
                agg_str = f"{self.aggregated_signal:.3f}"
        return (
            f"Bar({self.ticker}, {self.timestamp.date()}, C={self.close:.2f}, "
            f"agg={agg_str})"
        )
