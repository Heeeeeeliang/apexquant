"""
External data context for non-price predictors.

Holds supplementary data sources (economic indicators, sentiment,
earnings calendars, etc.) that predictors can optionally consume.
All methods are currently stubs returning neutral defaults — they
will be implemented as data pipelines are connected.

Usage::

    from data.context import MarketContext

    ctx = MarketContext()
    vix = ctx.get_vix()            # stub → 20.0
    sent = ctx.get_sentiment()     # stub → 0.0
    cal = ctx.get_earnings_cal()   # stub → empty DataFrame
"""

__all__ = ["MarketContext"]

import pandas as pd
from loguru import logger


class MarketContext:
    """Container for external (non-price) market data.

    Each method returns a sensible neutral default until the
    corresponding data pipeline is connected.  All stubs raise
    no errors so that callers can safely integrate them without
    gating on availability.

    Attributes:
        ticker: Primary ticker symbol this context relates to.
    """

    def __init__(self, ticker: str = "") -> None:
        self.ticker = ticker
        logger.debug("MarketContext initialised for '{}'", ticker or "global")

    def get_vix(self) -> float:
        """Return current VIX level.

        Returns:
            VIX value (stub returns 20.0 — long-run average).
        """
        raise NotImplementedError("TODO: VIX data integration")

    def get_sentiment(self) -> float:
        """Return aggregate market sentiment score.

        Returns:
            Sentiment in ``[-1.0, 1.0]`` (stub returns 0.0 — neutral).
        """
        raise NotImplementedError("TODO: sentiment data integration")

    def get_earnings_calendar(self) -> pd.DataFrame:
        """Return upcoming earnings dates for the ticker.

        Returns:
            DataFrame with columns ``[date, ticker, estimate_eps]``
            (stub returns empty DataFrame).
        """
        raise NotImplementedError("TODO: earnings calendar integration")

    def get_economic_indicators(self) -> dict[str, float]:
        """Return latest macro-economic indicators.

        Returns:
            Dict mapping indicator name to value (stub returns empty dict).
        """
        raise NotImplementedError("TODO: economic indicators integration")

    def get_sector_momentum(self) -> float:
        """Return the sector momentum score for the ticker's sector.

        Returns:
            Momentum score in ``[-1.0, 1.0]`` (stub returns 0.0).
        """
        raise NotImplementedError("TODO: sector momentum integration")

    def __repr__(self) -> str:
        return f"MarketContext(ticker={self.ticker!r})"
