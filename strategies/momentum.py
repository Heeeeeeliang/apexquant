"""
Momentum strategy — goes long when the aggregated signal is bullish
and strong, short when bearish and strong.

Usage::

    from strategies.momentum import MomentumStrategy
    from config.default import CONFIG

    strat = MomentumStrategy(CONFIG)
    signal = strat.on_bar(bar)
"""

__all__ = ["MomentumStrategy"]

from typing import Any

from loguru import logger

from data.bar import Bar
from strategies.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Threshold-based momentum strategy on aggregated signal.

    Attributes:
        name: ``"momentum"``.
        long_threshold: Signal direction above which to go long.
        short_threshold: Signal direction below which to go short.
    """

    name: str = "momentum"

    def __init__(
        self,
        config: dict[str, Any],
        long_threshold: float = 0.3,
        short_threshold: float = -0.3,
    ) -> None:
        super().__init__(config)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        logger.info(
            "MomentumStrategy: long>{:+.2f}, short<{:+.2f}",
            long_threshold,
            short_threshold,
        )

    def on_bar(self, bar: Bar) -> Signal:
        """Generate a momentum signal from the aggregated signal.

        Args:
            bar: Current bar with ``aggregated_signal`` populated.

        Returns:
            BUY if bullish and strong, SHORT if bearish and strong,
            HOLD otherwise.
        """
        agg = bar.aggregated_signal
        if agg is None:
            return Signal.HOLD

        direction = getattr(agg, "direction", 0.0)

        if direction >= self.long_threshold:
            if self.has_position(bar.ticker):
                return Signal.HOLD
            return Signal.BUY

        if direction <= self.short_threshold:
            if self.has_position(bar.ticker):
                return Signal.SELL
            return Signal.SHORT

        return Signal.HOLD
