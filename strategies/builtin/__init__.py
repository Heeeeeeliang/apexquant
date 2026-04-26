"""
Built-in strategy implementations for ApexQuant.

- :class:`AIStrategy` — default AI-signal-native strategy
- :class:`TechnicalStrategy` — pure EMA + RSI baseline

Usage::

    from strategies.builtin import AIStrategy, TechnicalStrategy
    from config.default import CONFIG

    ai = AIStrategy(CONFIG)
    tech = TechnicalStrategy(CONFIG)
"""

__all__ = ["AIStrategy", "TechnicalStrategy"]

from strategies.builtin.ai_strategy import AIStrategy
from strategies.builtin.technical import TechnicalStrategy
