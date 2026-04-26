"""
Backtrader data feed for ApexQuant.

Only OHLCV columns are registered as Backtrader lines.  All extra
columns (indicators, FeatureEngine outputs) are extracted into a
plain-Python dict keyed by nanosecond timestamp *before* being handed
to Backtrader.  This avoids Backtrader synchronising 160+ Line objects
every bar — the dominant cost in the previous implementation.

Usage::

    from backtest.bt_feeds import make_feed

    feed, features_by_ns, extra_col_names = make_feed(df, name="AAPL")
    cerebro.adddata(feed, name="AAPL")
"""

__all__ = ["make_feed"]

import numpy as np
import pandas as pd
from loguru import logger

try:
    import backtrader as bt
except ImportError:
    bt = None  # type: ignore[assignment]


# Standard OHLCV column names (Backtrader defaults expect lowercase)
_OHLCV = {"open", "high", "low", "close", "volume", "openinterest"}
# Capitalized variants used after _add_indicators in runner.py
_OHLCV_CAP = {"Open", "High", "Low", "Close", "Volume"}


def make_feed(
    df: pd.DataFrame,
    name: str = "",
) -> tuple:
    """Create a Backtrader OHLCV-only feed plus a side-channel features dict.

    Args:
        df: DataFrame with DatetimeIndex and OHLCV + indicator columns.
        name: Ticker name (for logging).

    Returns:
        Tuple of ``(bt_feed, features_by_ns, extra_col_names)``.

        * ``bt_feed`` — a ``bt.feeds.PandasData`` with only OHLCV lines.
        * ``features_by_ns`` — ``dict[int, dict[str, float]]`` mapping
          nanosecond timestamps to feature dicts.  This bypasses
          Backtrader entirely.
        * ``extra_col_names`` — list of the column names stored in
          *features_by_ns* (for diagnostics / logging).
    """
    df = df.copy()

    # Normalise to lowercase for Backtrader
    rename_map = {}
    for cap, low in zip(
        ["Open", "High", "Low", "Close", "Volume"],
        ["open", "high", "low", "close", "volume"],
    ):
        if cap in df.columns and low not in df.columns:
            rename_map[cap] = low
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Strip timezone if present (Backtrader doesn't handle tz well)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Deduplicate columns (FeatureEngine may produce overlapping names)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        logger.warning(
            "Dropping {} duplicate columns in feed {}: {}",
            len(dupes), name, dupes,
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Identify extra (non-OHLCV) columns
    extra_cols = [
        c for c in df.columns
        if c not in _OHLCV and c not in _OHLCV_CAP
    ]

    # ---- Build features dict (side-channel, outside Backtrader) ----
    features_by_ns: dict[int, dict[str, float]] = {}
    if extra_cols:
        # Vectorised extraction: convert extra columns to a NumPy matrix
        extra_df = df[extra_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        vals = extra_df.values  # shape (n_bars, n_extra)
        ts_ns = df.index.asi8   # int64 nanosecond array
        for i in range(len(df)):
            row_dict: dict[str, float] = {}
            row_vals = vals[i]
            for j, col in enumerate(extra_cols):
                v = row_vals[j]
                if v != 0.0 or not np.isnan(v):
                    row_dict[col] = float(v)
            features_by_ns[int(ts_ns[i])] = row_dict

    # ---- Strip extra columns — only OHLCV goes to Backtrader ----
    ohlcv_cols = [c for c in df.columns if c in _OHLCV]
    df_ohlcv = df[ohlcv_cols].copy()

    feed = bt.feeds.PandasData(
        dataname=df_ohlcv,
        datetime=None,  # use index
        open="open"   if "open"   in ohlcv_cols else -1,
        high="high"   if "high"   in ohlcv_cols else -1,
        low="low"     if "low"    in ohlcv_cols else -1,
        close="close" if "close"  in ohlcv_cols else -1,
        volume="volume" if "volume" in ohlcv_cols else -1,
        openinterest=-1,
    )

    logger.debug(
        "Created OHLCV-only feed for {}: {} bars, {} features in side-channel",
        name, len(df_ohlcv), len(extra_cols),
    )
    return feed, features_by_ns, extra_cols
