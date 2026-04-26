"""
Market-data loading with source routing and chronological splitting.

Routes to the correct data source based on ``config["data"]["source"]``:

- ``"csv"``           — load from ``{features_dir}/{ticker}_{freq}_features.csv``
- ``"yfinance"``      — TODO (raises :class:`NotImplementedError`)
- ``"databento_api"`` — TODO (raises :class:`NotImplementedError`)

All splits are strictly chronological — no shuffling, ever.

Usage::

    from data.loader import load_data, load_raw, load_all_tickers
    from config.default import CONFIG

    df = load_data("AAPL", "15min", CONFIG)
    raw = load_raw("AAPL", "15min", CONFIG)
    all_dfs = load_all_tickers("15min", CONFIG)
"""

__all__ = [
    "load_data",
    "load_raw",
    "load_all_tickers",
    "load_bars",
    "split_data",
]

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from data.bar import Bar
from data.cleaner import validate

# Streamlit-aware CSV cache: avoids re-reading unchanged files on page reruns.
try:
    import streamlit as _st_cache

    @_st_cache.cache_data(show_spinner=False)
    def _cached_read_csv(path_str: str, _mtime: float) -> pd.DataFrame:
        return pd.read_csv(path_str)
except Exception:
    def _cached_read_csv(path_str: str, _mtime: float) -> pd.DataFrame:
        return pd.read_csv(path_str)


# ---------------------------------------------------------------------------
# Source-routed loaders
# ---------------------------------------------------------------------------

def load_data(ticker: str, freq: str, config: dict[str, Any]) -> pd.DataFrame:
    """Load OHLCV data with pre-computed features for a single ticker.

    Routes to the backend specified by ``config["data"]["source"]``.

    Args:
        ticker: Stock symbol (e.g. ``"AAPL"``).
        freq: Bar frequency (e.g. ``"15min"``, ``"1hour"``).
        config: Full CONFIG dict.

    Returns:
        DataFrame with DatetimeIndex and OHLCV + feature columns.

    Raises:
        NotImplementedError: For yfinance or Databento sources.
        FileNotFoundError: If the CSV file does not exist.
    """
    source = config.get("data", {}).get("source", "csv")

    if source == "csv":
        return _load_csv(ticker, freq, config)
    elif source == "yfinance":
        raise NotImplementedError("TODO: yfinance integration")
    elif source == "databento_api":
        raise NotImplementedError("TODO: Databento API integration")
    else:
        raise ValueError(f"Unknown data source: {source!r}")


def load_raw(ticker: str, freq: str, config: dict[str, Any]) -> pd.DataFrame:
    """Load raw OHLCV data without pre-computed features.

    Same source routing as :func:`load_data`, but loads from the
    ``raw_dir`` instead of ``features_dir`` and only returns the
    core OHLCV columns.

    Args:
        ticker: Stock symbol.
        freq: Bar frequency.
        config: Full CONFIG dict.

    Returns:
        DataFrame with DatetimeIndex and columns
        ``[Open, High, Low, Close, Volume]``.

    Raises:
        NotImplementedError: For yfinance or Databento sources.
        FileNotFoundError: If the CSV file does not exist.
    """
    source = config.get("data", {}).get("source", "csv")

    if source == "csv":
        raw_dir = config.get("data", {}).get("raw_dir", "data/raw/")
        return _load_csv(ticker, freq, config, base_dir=raw_dir, suffix="")
    elif source == "yfinance":
        raise NotImplementedError("TODO: yfinance integration")
    elif source == "databento_api":
        raise NotImplementedError("TODO: Databento API integration")
    else:
        raise ValueError(f"Unknown data source: {source!r}")


def load_all_tickers(
    freq: str, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Load data for every ticker listed in the config.

    Tickers that fail to load are logged as warnings and skipped.

    Args:
        freq: Bar frequency.
        config: Full CONFIG dict.

    Returns:
        Dict mapping ticker symbol to its DataFrame.
    """
    tickers: list[str] = config.get("data", {}).get("tickers", [])
    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            df = load_data(ticker, freq, config)
            result[ticker] = df
        except Exception as exc:
            logger.warning(
                "Skipping ticker '{}': {} — {}",
                ticker,
                type(exc).__name__,
                exc,
            )

    logger.info(
        "Loaded {}/{} tickers for freq={}",
        len(result),
        len(tickers),
        freq,
    )
    return result


# ---------------------------------------------------------------------------
# CSV backend
# ---------------------------------------------------------------------------

_TIMESTAMP_CANDIDATES = ["ts_event", "timestamp", "datetime", "date"]


def _load_csv(
    ticker: str,
    freq: str,
    config: dict[str, Any],
    base_dir: str | None = None,
    suffix: str = "_features",
) -> pd.DataFrame:
    """Load a CSV file, parse timestamps, validate, and return.

    Handles multiple timestamp column names:
    ``ts_event``, ``timestamp``, ``datetime``, ``date``.

    Args:
        ticker: Stock symbol.
        freq: Bar frequency.
        config: Full CONFIG dict.
        base_dir: Override directory (defaults to ``features_dir``).
        suffix: Filename suffix before ``.csv`` (default ``"_features"``).

    Returns:
        Cleaned DataFrame with UTC DatetimeIndex.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no recognised timestamp column is found.
    """
    if base_dir is None:
        base_dir = config.get("data", {}).get("features_dir", "data/features/")

    base = Path(base_dir)
    csv_path = base / f"{ticker}_{freq}{suffix}.csv"

    # Try alternate filename patterns if primary doesn't exist
    if not csv_path.exists():
        alternates = [
            base / f"{ticker}{suffix}.csv",              # AAPL_features.csv
            base / f"{ticker}_{freq}.csv",                # AAPL_15min.csv
            base / f"{ticker}.csv",                       # AAPL.csv
            base / f"{ticker.lower()}_{freq}{suffix}.csv",
            base / f"{ticker.lower()}.csv",
        ]
        for alt in alternates:
            if alt.exists():
                csv_path = alt
                logger.info("Using alternate CSV path: {}", csv_path)
                break

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {base / f'{ticker}_{freq}{suffix}.csv'} "
            f"(also tried alternate patterns in {base})"
        )

    df = _cached_read_csv(str(csv_path), csv_path.stat().st_mtime)

    # Normalise lowercase OHLCV columns to Title case expected downstream.
    # Only renames the 5 core columns; leaves indicator columns untouched.
    _OHLCV_MAP = {"open": "Open", "high": "High", "low": "Low",
                  "close": "Close", "volume": "Volume"}
    df.rename(columns={k: v for k, v in _OHLCV_MAP.items()
                       if k in df.columns}, inplace=True)

    logger.info("Read {} rows from {}", len(df), csv_path)

    # --- Detect and parse timestamp column ---
    ts_col = _find_timestamp_column(df)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col)
    df.index.name = "timestamp"

    # --- Sort and deduplicate ---
    df = df.sort_index()
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning("Dropped {} duplicate timestamps for {}", n_dupes, ticker)

    # --- Validate ---
    df = validate(df, ticker)

    logger.info(
        "Loaded {}: {} rows, {} to {}",
        ticker,
        len(df),
        df.index.min().date() if len(df) > 0 else "N/A",
        df.index.max().date() if len(df) > 0 else "N/A",
    )
    return df


def _find_timestamp_column(df: pd.DataFrame) -> str:
    """Detect the timestamp column from known candidates.

    Args:
        df: Raw DataFrame.

    Returns:
        The name of the detected timestamp column.

    Raises:
        ValueError: If no candidate column is found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in cols_lower:
            return cols_lower[candidate]

    raise ValueError(
        f"No timestamp column found. Expected one of "
        f"{_TIMESTAMP_CANDIDATES}, got {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_bars(
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
) -> list[Bar]:
    """Download OHLCV data via yfinance and convert to Bar objects.

    This is the legacy interface retained for backward compatibility
    with ``run_all.py``.  New code should prefer :func:`load_data`.

    Args:
        ticker: Stock symbol (default from config).
        start: Start date string (default from config).
        end: End date string (default from config).
        interval: Bar interval (default from config).

    Returns:
        Chronologically ordered list of Bar instances.
    """
    from config.default import get

    ticker = ticker or get("data.tickers", ["AAPL"])[0]
    start = start or get("data.start_date", "2015-01-01")
    end = end or get("data.end_date", "2024-01-01")
    interval = interval or get("data.interval", "1d")

    cache_dir = Path(get("data.cache_dir", "data/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}_{interval}.parquet"

    if cache_file.exists():
        logger.info("Loading cached data: {}", cache_file)
        df = pd.read_parquet(cache_file)
    else:
        import yfinance as yf

        logger.info(
            "Downloading {} data: {} to {} ({})", ticker, start, end, interval
        )
        df = yf.download(ticker, start=start, end=end, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_parquet(cache_file)
        logger.info("Cached {} rows to {}", len(df), cache_file)

    bars = _dataframe_to_bars(df, ticker)
    logger.info("Loaded {} bars for {}", len(bars), ticker)
    return bars


def _dataframe_to_bars(df: pd.DataFrame, ticker: str) -> list[Bar]:
    """Convert a pandas OHLCV DataFrame into a list of Bar objects.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns.
        ticker: Ticker symbol for metadata.

    Returns:
        List of Bar instances.
    """
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bar = Bar(
            timestamp=pd.Timestamp(ts),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            meta={"ticker": ticker},
        )
        bars.append(bar)
    return bars


def split_data(
    bars: list[Bar],
    train: float | None = None,
    val: float | None = None,
    test: float | None = None,
) -> tuple[list[Bar], list[Bar], list[Bar]]:
    """Split bars chronologically into train / val / test sets.

    Args:
        bars: Full list of bars (must be in chronological order).
        train: Fraction for training (default from config).
        val: Fraction for validation (default from config).
        test: Fraction for test (default from config).

    Returns:
        Tuple of (train_bars, val_bars, test_bars).

    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    from config.default import get

    train = train or get("data.train_ratio", 0.7)
    val = val or get("data.val_ratio", 0.15)
    test = test or get("data.test_ratio", 0.15)

    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train + val + test:.4f}")

    n = len(bars)
    train_end = int(n * train)
    val_end = int(n * (train + val))

    train_bars = bars[:train_end]
    val_bars = bars[train_end:val_end]
    test_bars = bars[val_end:]

    logger.info(
        "Split: train={}, val={}, test={} (total={})",
        len(train_bars),
        len(val_bars),
        len(test_bars),
        n,
    )
    return train_bars, val_bars, test_bars
