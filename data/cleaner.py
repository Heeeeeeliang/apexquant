"""
Data validation and cleaning for ApexQuant.

Provides OHLCV integrity checks, forward-fill for small gaps,
time-continuity warnings, and known stock-split adjustments.

Usage::

    from data.cleaner import validate, adjust_splits, get_train_val_test_split
    from config.default import CONFIG

    df = validate(df, "AAPL")
    df = adjust_splits(df, "TSLA")
    train, val, test = get_train_val_test_split(df, CONFIG)
"""

__all__ = ["validate", "adjust_splits", "get_train_val_test_split"]

from typing import Any

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Known stock splits: {ticker: {date_str: ratio}}
# Prices BEFORE the split date are divided by ratio; volume is multiplied.
# ---------------------------------------------------------------------------
_KNOWN_SPLITS: dict[str, dict[str, float]] = {
    "TSLA": {"2020-08-31": 5.0},      # 5-for-1
    "GOOGL": {"2022-07-18": 20.0},     # 20-for-1
    "GOOG": {"2022-07-18": 20.0},      # 20-for-1
}

# OHLCV columns that must be present
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Maximum consecutive NaN rows to forward-fill
_MAX_FFILL = 3


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate OHLCV integrity and apply light cleaning.

    Checks performed:

    1. **OHLC relationships** — High >= max(Open, Close) and
       Low <= min(Open, Close).  Violations are logged as warnings
       but rows are **not** dropped.
    2. **NaN check** — forward-fills up to :data:`_MAX_FFILL`
       consecutive NaNs in OHLCV columns.  Remaining NaNs are
       logged.
    3. **Time continuity** — warns if any gap exceeds 2x the
       median interval.

    Args:
        df: DataFrame with OHLCV columns and a DatetimeIndex.
        ticker: Ticker symbol (used in log messages).

    Returns:
        The cleaned DataFrame (same object, mutated in-place where
        appropriate).
    """
    logger.info("Validating {} ({} rows)", ticker, len(df))

    # --- Ensure OHLCV columns exist ---
    present = [c for c in _OHLCV if c in df.columns]
    missing = [c for c in _OHLCV if c not in df.columns]
    if missing:
        logger.warning("{}: missing OHLCV columns: {}", ticker, missing)

    # --- 1. OHLC relationship check ---
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        oc_max = df[["Open", "Close"]].max(axis=1)
        oc_min = df[["Open", "Close"]].min(axis=1)

        high_violations = (df["High"] < oc_max).sum()
        low_violations = (df["Low"] > oc_min).sum()

        if high_violations > 0:
            logger.warning(
                "{}: {} rows where High < max(Open, Close)",
                ticker,
                high_violations,
            )
        if low_violations > 0:
            logger.warning(
                "{}: {} rows where Low > min(Open, Close)",
                ticker,
                low_violations,
            )

    # --- 2. NaN handling ---
    for col in present:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            logger.warning("{}: {} NaN values in '{}'", ticker, n_nan, col)

    if present:
        df[present] = df[present].ffill(limit=_MAX_FFILL)
        remaining_nan = int(df[present].isna().sum().sum())
        if remaining_nan > 0:
            logger.warning(
                "{}: {} NaN values remain after forward-fill (limit={})",
                ticker,
                remaining_nan,
                _MAX_FFILL,
            )

    # --- 3. Time continuity ---
    if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
        deltas = df.index.to_series().diff().dropna()
        if len(deltas) > 0:
            median_delta = deltas.median()
            large_gaps = deltas[deltas > median_delta * 2]
            if len(large_gaps) > 0:
                logger.warning(
                    "{}: {} time gaps > 2x median interval ({}).  "
                    "Largest: {} at {}",
                    ticker,
                    len(large_gaps),
                    median_delta,
                    large_gaps.max(),
                    large_gaps.idxmax(),
                )

    logger.info("Validation complete for {} ({} rows)", ticker, len(df))
    return df


# ---------------------------------------------------------------------------
# adjust_splits
# ---------------------------------------------------------------------------

def adjust_splits(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Adjust historical prices for known stock splits.

    For each known split of *ticker*, all OHLC prices **before** the
    split date are divided by the split ratio, and volume is multiplied
    by the ratio.  This produces a split-adjusted series that is
    continuous across the event.

    Args:
        df: DataFrame with OHLCV columns and a DatetimeIndex.
        ticker: Ticker symbol.

    Returns:
        The adjusted DataFrame (mutated in-place).
    """
    splits = _KNOWN_SPLITS.get(ticker, {})
    if not splits:
        logger.debug("No known splits for {}", ticker)
        return df

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]

    for date_str, ratio in splits.items():
        split_date = pd.Timestamp(date_str, tz=df.index.tz)
        mask = df.index < split_date
        n_adjusted = int(mask.sum())

        if n_adjusted == 0:
            logger.debug(
                "{}: split {} (ratio {}) — no rows before split date",
                ticker,
                date_str,
                ratio,
            )
            continue

        for col in price_cols:
            df.loc[mask, col] = df.loc[mask, col] / ratio

        if "Volume" in df.columns:
            df.loc[mask, "Volume"] = df.loc[mask, "Volume"] * ratio

        logger.info(
            "{}: adjusted {} rows for {}-for-1 split on {}",
            ticker,
            n_adjusted,
            int(ratio),
            date_str,
        )

    return df


# ---------------------------------------------------------------------------
# get_train_val_test_split
# ---------------------------------------------------------------------------

def get_train_val_test_split(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train / val / test sets.

    Strictly chronological — **no shuffling**, ever.  Ratios are read
    from ``config["data"]``.

    Args:
        df: Full DataFrame with a DatetimeIndex (must be sorted).
        config: Full CONFIG dict containing
            ``train_ratio``, ``val_ratio``, ``test_ratio`` under
            ``config["data"]``.

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: If the ratios do not sum to 1.0 (within tolerance).
    """
    data_cfg = config.get("data", {})
    train_ratio = data_cfg.get("train_ratio", 0.70)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    def _date_range(d: pd.DataFrame) -> str:
        if len(d) == 0:
            return "empty"
        return f"{d.index.min().date()} to {d.index.max().date()}"

    logger.info(
        "Chronological split: train={} [{}], val={} [{}], test={} [{}]",
        len(train_df),
        _date_range(train_df),
        len(val_df),
        _date_range(val_df),
        len(test_df),
        _date_range(test_df),
    )
    return train_df, val_df, test_df
