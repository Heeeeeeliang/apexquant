"""
Feature engineering pipeline using pandas-ta.

Computes technical indicators and derived features, then attaches
them to each Bar's ``features`` dict.

Usage::

    from data.feature_engine import FeatureEngine
    from data.loader import load_bars

    bars = load_bars("AAPL")
    engine = FeatureEngine()
    bars = engine.compute(bars)
    print(bars[-1].features.keys())
"""

__all__ = ["FeatureEngine", "compute_features_df"]

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from config import get
from data.bar import Bar


def compute_features_df(
    df: pd.DataFrame,
    vol_windows: list[int] | None = None,
    ta_indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Compute a comprehensive feature set directly on a DataFrame.

    Produces the same features used during model training:

    1. **Base features** — returns, log-returns, volatility windows,
       pandas-ta indicators.
    2. **Lag features** — shifted values of base features.
    3. **Rolling statistics** — mean, std, min, max over multiple windows.

    All new columns are collected into a dict first and joined once at
    the end to avoid DataFrame fragmentation.

    Args:
        df: DataFrame with ``Close``, ``High``, ``Low``, ``Volume`` columns
            and a DatetimeIndex.
        vol_windows: Volatility rolling windows (default ``[5, 10, 20, 60]``).
        ta_indicators: pandas-ta indicator names
            (default ``["rsi", "macd", "bbands", "atr", "obv"]``).

    Returns:
        The same DataFrame with feature columns added and NaN filled with 0.
    """
    if vol_windows is None:
        vol_windows = get("features.vol_windows", [5, 10, 20, 60])
    if ta_indicators is None:
        ta_indicators = get(
            "features.ta_indicators", ["rsi", "macd", "bbands", "atr", "obv"]
        )

    # Defragment the input up-front so we start from a contiguous block.
    df = df.copy()

    c = df["Close"]
    h = df["High"]
    lo = df["Low"]
    v = df["Volume"]
    o = df["Open"]
    c_safe = c.replace(0, 1e-10)

    # Accumulator for every new column — joined once at the very end.
    new: dict[str, pd.Series] = {}

    # ---------------------------------------------------------------
    # 1. Base features (returns, volatility, pandas-ta)
    # ---------------------------------------------------------------
    returns = c.pct_change()
    log_returns = np.log(c / c.shift(1))
    new["returns"] = returns
    new["log_returns"] = log_returns

    for w in vol_windows:
        new[f"vol_{w}"] = log_returns.rolling(w).std() * np.sqrt(252)

    # pandas-ta indicators — each may return a Series or multi-col DataFrame
    for name in ta_indicators:
        try:
            result = df.ta.__getattribute__(name)()
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    new[col] = result[col]
            elif isinstance(result, pd.Series):
                new[name] = result
        except Exception as exc:
            logger.warning("compute_features_df: failed to compute '{}': {}", name, exc)

    # ---------------------------------------------------------------
    # 2. Price-derived features
    # ---------------------------------------------------------------
    new["high_low_range"] = (h - lo) / c_safe
    new["close_open_range"] = (c - o) / c_safe
    new["upper_shadow"] = (h - np.maximum(c, o)) / c_safe
    new["lower_shadow"] = (np.minimum(c, o) - lo) / c_safe
    new["body_ratio"] = abs(c - o) / (h - lo).replace(0, 1e-10)
    new["vol_price_ratio"] = v / c_safe
    new["typical_price"] = (h + lo + c) / 3.0
    new["median_price"] = (h + lo) / 2.0
    new["price_momentum"] = c - c.shift(1)

    # ---------------------------------------------------------------
    # 3. Lag features (shifts of key series)
    # ---------------------------------------------------------------
    # We need the series from `new` that were just computed above.
    _lag_sources = {
        "returns": returns,
        "log_returns": log_returns,
        "high_low_range": new["high_low_range"],
        "vol_price_ratio": new["vol_price_ratio"],
    }
    for col_name, series in _lag_sources.items():
        for lag in [1, 2, 3, 5, 10]:
            new[f"{col_name}_lag{lag}"] = series.shift(lag)

    # ---------------------------------------------------------------
    # 4. Rolling statistics over multiple windows
    # ---------------------------------------------------------------
    _roll_windows = [5, 10, 20, 60]
    _roll_fns = {
        "mean": lambda s, w: s.rolling(w).mean(),
        "std": lambda s, w: s.rolling(w).std(),
        "min": lambda s, w: s.rolling(w).min(),
        "max": lambda s, w: s.rolling(w).max(),
        "skew": lambda s, w: s.rolling(w).skew(),
    }

    for col_name, series in _lag_sources.items():
        for w in _roll_windows:
            for stat_name, stat_fn in _roll_fns.items():
                new[f"{col_name}_r{w}_{stat_name}"] = stat_fn(series, w)

    # Close-price rolling features
    for w in _roll_windows:
        sma = c.rolling(w).mean()
        sma_safe = sma.replace(0, 1e-10)
        new[f"close_sma{w}_ratio"] = c / sma_safe
        new[f"close_sma{w}_dist"] = (c - sma) / sma_safe

    # Volume rolling features
    for w in _roll_windows:
        vol_sma = v.rolling(w).mean().replace(0, 1e-10)
        new[f"volume_r{w}_ratio"] = v / vol_sma

    # ---------------------------------------------------------------
    # 5. Cross-sectional / interaction features
    # ---------------------------------------------------------------
    new["return_x_volume"] = returns * v
    new["range_x_volume"] = new["high_low_range"] * v
    rsi_series = new.get("rsi") if "rsi" in new else new.get("RSI_14")
    # pandas-ta may name the column differently; check common variants
    if rsi_series is None:
        for key in new:
            if key.lower().startswith("rsi"):
                rsi_series = new[key]
                break
    if rsi_series is not None:
        new["rsi_detrended"] = rsi_series - 50.0
        new["rsi_extreme"] = ((rsi_series > 70) | (rsi_series < 30)).astype(float)

    # ---------------------------------------------------------------
    # 6. Single concat + fill NaN
    # ---------------------------------------------------------------
    features_df = pd.DataFrame(new, index=df.index)
    # Coerce non-numeric to NaN, then fill with 0
    features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df = pd.concat([df, features_df], axis=1)

    logger.info(
        "compute_features_df: {} feature columns, {} rows",
        len(features_df.columns),
        len(df),
    )
    return df


class FeatureEngine:
    """Computes technical-analysis features on a list of Bar objects.

    Features are written into each bar's ``features`` dict.  The engine
    converts bars to a DataFrame internally, runs pandas-ta, then maps
    results back.

    Attributes:
        lookback: Number of bars used for rolling calculations.
        vol_windows: Window sizes for volatility features.
        indicators: List of TA indicator names to compute.
    """

    def __init__(
        self,
        lookback: int | None = None,
        vol_windows: list[int] | None = None,
        indicators: list[str] | None = None,
    ) -> None:
        self.lookback = lookback or get("features.lookback_window", 20)
        self.vol_windows = vol_windows or get("features.vol_windows", [5, 10, 20, 60])
        self.indicators = indicators or get(
            "features.ta_indicators", ["rsi", "macd", "bbands", "atr", "obv"]
        )
        logger.info(
            "FeatureEngine initialised: lookback={}, indicators={}",
            self.lookback,
            self.indicators,
        )

    def compute(self, bars: list[Bar]) -> list[Bar]:
        """Compute all features and attach them to bars.

        Args:
            bars: Chronologically ordered list of Bar instances.

        Returns:
            The same list of bars with ``features`` populated.
        """
        df = self._bars_to_dataframe(bars)
        df = self._add_returns(df)
        df = self._add_volatility(df)
        df = self._add_ta_indicators(df)
        df = df.fillna(0.0)

        self._write_features_to_bars(df, bars)
        logger.info("Computed {} features across {} bars", len(df.columns) - 5, len(bars))
        return bars

    def _bars_to_dataframe(self, bars: list[Bar]) -> pd.DataFrame:
        """Convert bars to a DataFrame for vectorised feature computation.

        Args:
            bars: List of Bar instances.

        Returns:
            DataFrame with OHLCV columns.
        """
        records = [
            {
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume,
            }
            for b in bars
        ]
        df = pd.DataFrame(records, index=[b.timestamp for b in bars])
        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns and simple returns.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with return columns added.
        """
        new = {
            "returns": df["Close"].pct_change(),
            "log_returns": np.log(df["Close"] / df["Close"].shift(1)),
        }
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realised volatility over multiple windows.

        Args:
            df: DataFrame with log_returns column.

        Returns:
            DataFrame with volatility columns added.
        """
        new: dict[str, pd.Series] = {}
        for w in self.vol_windows:
            new[f"vol_{w}"] = df["log_returns"].rolling(w).std() * np.sqrt(252)
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    def _add_ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pandas-ta technical indicators.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with TA indicator columns added.
        """
        new: dict[str, pd.Series] = {}
        for name in self.indicators:
            try:
                result = df.ta.__getattribute__(name)()
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        new[col] = result[col]
                elif isinstance(result, pd.Series):
                    new[name] = result
            except Exception as exc:
                logger.warning("Failed to compute indicator '{}': {}", name, exc)
        if new:
            return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
        return df

    def _write_features_to_bars(
        self, df: pd.DataFrame, bars: list[Bar]
    ) -> None:
        """Map computed DataFrame columns back onto Bar.features.

        Args:
            df: DataFrame with all feature columns.
            bars: Original Bar list (mutated in-place).
        """
        feature_cols = [c for c in df.columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
        for i, bar in enumerate(bars):
            bar.features = {col: float(df.iloc[i][col]) for col in feature_cols}
