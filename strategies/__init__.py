"""
ApexQuant Strategies Module
============================

Strategies consume ``bar.aggregated_signal`` and ``bar.predictions``
only.  They **never** import from ``predictors/`` directly — this
decoupling is a core architectural rule.

Usage::

    from strategies import BaseStrategy, Signal, Trade, OrderType
    from strategies import MomentumStrategy, MeanReversionStrategy
    from strategies import AIStrategy, TechnicalStrategy
    from config.default import CONFIG

    ai = AIStrategy(CONFIG)
    tech = TechnicalStrategy(CONFIG)
    signal = ai.on_bar(bar)
"""

__all__ = [
    "Signal",
    "OrderType",
    "Trade",
    "BaseStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "AIStrategy",
    "TechnicalStrategy",
]

from strategies.base import Signal, OrderType, Trade, BaseStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.builtin import AIStrategy, TechnicalStrategy
