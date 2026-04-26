"""
ApexQuant Data Module
=====================

Provides the canonical ``Bar`` data carrier, market-data loading,
validation/cleaning, feature engineering, and chronological splitting.

Usage::

    from data import Bar, load_data, load_raw, load_all_tickers
    from data import validate, adjust_splits, get_train_val_test_split
    from data import MarketContext, FeatureEngine
    from data import load_bars, split_data  # legacy
"""

__all__ = [
    "Bar",
    "load_data",
    "load_raw",
    "load_all_tickers",
    "load_bars",
    "split_data",
    "validate",
    "adjust_splits",
    "get_train_val_test_split",
    "MarketContext",
    "FeatureEngine",
]

from data.bar import Bar
from data.loader import load_data, load_raw, load_all_tickers, load_bars, split_data
from data.cleaner import validate, adjust_splits, get_train_val_test_split
from data.context import MarketContext
from data.feature_engine import FeatureEngine
