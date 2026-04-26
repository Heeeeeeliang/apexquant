"""
Mean-reversion strategy — fades the aggregated signal when it reaches
extremes, expecting a reversion to the mean.

Usage::

    from strategies.mean_reversion import MeanReversionStrategy
    from config.default import CONFIG

    strat = MeanReversionStrategy(CONFIG)
    signal = strat.on_bar(bar)
"""

__all__ = ["MeanReversionStrategy"]

from typing import Any

from loguru import logger

from data.bar import Bar
from strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Contrarian strategy that fades extreme aggregated signals.

    Attributes:
        name: ``"mean_reversion"``.
        overbought: Direction level above which to go short (fade).
        oversold: Direction level below which to go long (fade).
    """

    name: str = "mean_reversion"

    def __init__(
        self,
        config: dict[str, Any],
        overbought: float = 0.6,
        oversold: float = -0.6,
    ) -> None:
        super().__init__(config)
        self.overbought = overbought
        self.oversold = oversold
        logger.info(
            "MeanReversionStrategy: overbought>{:+.2f}, oversold<{:+.2f}",
            overbought,
            oversold,
        )

    def on_bar(self, bar: Bar) -> Signal:
        """Generate a mean-reversion signal from the aggregated signal.

        Args:
            bar: Current bar with ``aggregated_signal`` populated.

        Returns:
            SHORT if overbought (fade), BUY if oversold (fade),
            HOLD otherwise.
        """
        agg = bar.aggregated_signal
        if agg is None:
            return Signal.HOLD

        direction = getattr(agg, "direction", 0.0)

        if direction >= self.overbought:
            if self.has_position(bar.ticker):
                return Signal.HOLD
            return Signal.SHORT

        if direction <= self.oversold:
            if self.has_position(bar.ticker):
                return Signal.HOLD
            return Signal.BUY

        return Signal.HOLD
