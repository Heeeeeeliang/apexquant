"""
Volatility model feature pipeline — replicates training-time transformations.

Builds the 119-feature vector that ``lightgbm_v3_flat`` (Model D) expects:

1. Load 1-hour OHLCV → compute 40 indicators matching ``create_features()``
2. Aggregate into 12-bar blocks (mean of numeric columns, block RV)
3. Aggregate 1-hour bars into daily bars (daily_rv, range, return, vol_change)
4. Assemble per-block features: block history, RV stats, tech, SPY, daily

Usage::

    from data.vol_features import build_vol_features
    vol_df = build_vol_features("AAPL", config)  # DataFrame with 119 cols
"""

__all__ = ["build_vol_features"]

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ── Constants matching training notebook ──────────────────────────────
BLOCK_SIZE = 12
SEQ_LEN = 10
DAILY_LB = 5

# The 45 numeric columns from the original 1-hour feature CSV.
# Order matters — it defines the ``fc_list`` used for ``tech_`` prefix features
# and for fuzzy-matching ``close``/``volume`` in block features.
_FC_LIST = [
    "open", "high", "low", "close", "volume",
    "returns", "log_returns", "price_change", "price_range", "price_range_pct",
    "ma_5", "ma_ratio_5", "ma_10", "ma_ratio_10",
    "ma_20", "ma_ratio_20", "ma_60", "ma_ratio_60",
    "ema_12", "ema_26", "macd", "macd_signal",
    "volatility_20", "volatility_60", "rsi",
    "bb_middle", "bb_std", "bb_upper", "bb_lower", "bb_width",
    "volume_ma_20", "relative_volume", "volume_change",
    "momentum_5", "roc_5", "momentum_10", "roc_10", "momentum_20", "roc_20",
    "hour", "day_of_week", "month",
    "high_low_ratio", "close_to_high", "close_to_low",
]

# Columns used for per-block key features (fuzzy-matched in training code)
_KEY_COLS = ["close", "volume"]

# Daily columns used in d{0-4}_ features
_DAILY_COLS = ["daily_rv", "daily_range", "daily_return", "vol_change"]


# ═══════════════════════════════════════════════════════════════════════
#  1-hour indicator computation (replicates Colab create_features())
# ═══════════════════════════════════════════════════════════════════════

def create_1h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 40 indicators that were in the training 1-hour feature CSV.

    Expects Title-case OHLCV columns (Open/High/Low/Close/Volume) with
    a DatetimeIndex.  Returns a DataFrame with 45 lowercase columns
    (5 OHLCV + 40 indicators) matching ``_FC_LIST``.
    """
    out = pd.DataFrame(index=df.index)
    c = df["Close"].astype(np.float64)
    h = df["High"].astype(np.float64)
    lo = df["Low"].astype(np.float64)
    o = df["Open"].astype(np.float64)
    v = df["Volume"].astype(np.float64)

    c_safe = c.replace(0, 1e-10)
    lo_safe = lo.replace(0, 1e-10)
    h_lo = (h - lo).replace(0, 1e-10)

    out["open"] = o
    out["high"] = h
    out["low"] = lo
    out["close"] = c
    out["volume"] = v

    # Returns
    out["returns"] = c.pct_change()
    out["log_returns"] = np.log(c / c.shift(1))
    out["price_change"] = c - c.shift(1)
    out["price_range"] = h - lo
    out["price_range_pct"] = (h - lo) / c_safe

    # Moving averages
    for w in [5, 10, 20, 60]:
        ma = c.rolling(w).mean()
        out[f"ma_{w}"] = ma
        out[f"ma_ratio_{w}"] = c / ma.replace(0, 1e-10)

    # EMA / MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["ema_12"] = ema12
    out["ema_26"] = ema26
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    # Volatility
    lr = out["log_returns"]
    out["volatility_20"] = lr.rolling(20).std()
    out["volatility_60"] = lr.rolling(60).std()

    # RSI (Wilder's smoothing, period 14)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_middle"] = bb_mid
    out["bb_std"] = bb_std
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (4 * bb_std) / bb_mid.replace(0, 1e-10)

    # Volume indicators
    vol_ma = v.rolling(20).mean().replace(0, 1e-10)
    out["volume_ma_20"] = vol_ma
    out["relative_volume"] = v / vol_ma
    out["volume_change"] = v.pct_change()

    # Momentum / ROC
    for w in [5, 10, 20]:
        out[f"momentum_{w}"] = c - c.shift(w)
        out[f"roc_{w}"] = (c - c.shift(w)) / c.shift(w).replace(0, 1e-10)

    # Time features
    if isinstance(df.index, pd.DatetimeIndex):
        out["hour"] = df.index.hour
        out["day_of_week"] = df.index.dayofweek
        out["month"] = df.index.month
    else:
        out["hour"] = 0
        out["day_of_week"] = 0
        out["month"] = 0

    # Price ratios
    out["high_low_ratio"] = h / lo_safe
    out["close_to_high"] = (c - h) / h_lo
    out["close_to_low"] = (c - lo) / h_lo

    # Fill leading NaNs with 0 (matches training: np.nan_to_num)
    out = out.fillna(0.0)

    # Ensure column order matches _FC_LIST
    return out[_FC_LIST]


# ═══════════════════════════════════════════════════════════════════════
#  Block aggregation
# ═══════════════════════════════════════════════════════════════════════

def compute_blocks(
    df_1h: pd.DataFrame, block_size: int = BLOCK_SIZE
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Group 1-hour bars into non-overlapping blocks.

    Returns:
        (block_meta, block_feats, block_rvs)
        - block_meta: DataFrame with bar_start, bar_end, ts_start, ts_end
        - block_feats: (n_blocks, 45) array — mean of numeric cols per block
        - block_rvs: (n_blocks,) array — realized volatility per block
    """
    close = df_1h["close"].values.astype(np.float64)
    n_blocks = len(close) // block_size

    # Log returns for RV computation
    lr = np.concatenate([[0.0], np.diff(np.log(close + 1e-10))])

    # Numeric feature values for block-mean aggregation
    fv = np.nan_to_num(df_1h[_FC_LIST].values.astype(np.float32), nan=0.0)

    ts = df_1h.index  # DatetimeIndex

    rows = []
    block_feats = np.zeros((n_blocks, len(_FC_LIST)), dtype=np.float32)
    block_rvs = np.zeros(n_blocks, dtype=np.float64)

    for b in range(n_blocks):
        s = b * block_size
        e = (b + 1) * block_size
        rv = np.std(lr[s:e]) * np.sqrt(block_size)
        block_rvs[b] = rv
        block_feats[b] = np.mean(fv[s:e], axis=0)
        rows.append({
            "block_idx": b,
            "bar_start": s,
            "bar_end": e,
            "ts_start": ts[s],
            "ts_end": ts[min(e - 1, len(close) - 1)],
        })

    block_meta = pd.DataFrame(rows)
    return block_meta, block_feats, block_rvs


# ═══════════════════════════════════════════════════════════════════════
#  Daily bar aggregation
# ═══════════════════════════════════════════════════════════════════════

def compute_daily_bars(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-hour bars into daily bars with derived features.

    Matches the training notebook's ``compute_daily_bars()`` exactly.
    """
    df = df_1h.copy()
    # Strip tz for date grouping
    if hasattr(df.index, "tz") and df.index.tz is not None:
        idx_notz = df.index.tz_localize(None)
    else:
        idx_notz = df.index
    df["_date"] = idx_notz.date

    agg = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
    }
    if "volume" in df.columns:
        agg["volume"] = ("volume", "sum")

    daily = df.groupby("_date").agg(**agg).reset_index()
    daily.rename(columns={"_date": "date"}, inplace=True)

    c = daily["close"].values.astype(np.float64)
    h = daily["high"].values.astype(np.float64)
    lo = daily["low"].values.astype(np.float64)
    o = daily["open"].values.astype(np.float64)
    n = len(daily)

    daily["daily_return"] = (c - o) / (o + 1e-10)
    daily["daily_range"] = (h - lo) / (c + 1e-10)

    if "volume" in daily.columns:
        v = daily["volume"].values.astype(np.float64)
        vc = np.concatenate([[1.0], v[1:] / (v[:-1] + 1e-10)])
        daily["vol_change"] = vc
    else:
        daily["vol_change"] = 1.0

    # Daily RV from intraday log returns
    drv = []
    for _, grp in df.groupby("_date"):
        cc = grp["close"].values.astype(np.float64)
        if len(cc) > 1:
            lr = np.diff(np.log(cc + 1e-10))
            drv.append(np.std(lr) * np.sqrt(len(cc)))
        else:
            drv.append(0.0)
    daily["daily_rv"] = drv

    return daily


# ═══════════════════════════════════════════════════════════════════════
#  Model D feature assembly (119 features)
# ═══════════════════════════════════════════════════════════════════════

def build_block_features(
    block_meta: pd.DataFrame,
    block_feats: np.ndarray,
    block_rvs: np.ndarray,
    daily: pd.DataFrame,
    spy_rvs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the 119 Model D features for each valid block position.

    Returns a DataFrame with one row per block (starting from SEQ_LEN)
    and 119 named columns matching ``feature_names.json``.
    """
    n_blocks = len(block_rvs)
    fc_list = _FC_LIST

    # Indices of close/volume in fc_list
    close_idx = fc_list.index("close")
    volume_idx = fc_list.index("volume")

    # Daily lookup arrays
    daily_dates = daily["date"].values
    daily_vals = np.nan_to_num(
        daily[_DAILY_COLS].values.astype(np.float32), nan=0.0
    )

    log_rvs = np.log(block_rvs + 1e-10)

    rows = []
    timestamps = []

    for i in range(SEQ_LEN, n_blocks):
        feats: dict[str, float] = {}

        # ── A) Per-block features (10 blocks × 4) = 40 ──
        for off in range(SEQ_LEN):
            bi = i - SEQ_LEN + off
            px = f"b{off}"
            feats[f"{px}_rv"] = float(block_rvs[bi])
            feats[f"{px}_log_rv"] = float(log_rvs[bi])
            feats[f"{px}_close"] = float(block_feats[bi, close_idx])
            feats[f"{px}_volume"] = float(block_feats[bi, volume_idx])

        # ── B) RV statistics = 9 ──
        rv_win = block_rvs[max(0, i - SEQ_LEN):i]
        rv4 = block_rvs[max(0, i - 4):i]

        feats["rv_mean_4"] = float(np.mean(rv4)) if len(rv4) > 0 else 0.0
        feats["rv_std_4"] = float(np.std(rv4)) if len(rv4) > 1 else 0.0
        feats["rv_mean_8"] = float(np.mean(rv_win)) if len(rv_win) > 0 else 0.0
        feats["rv_std_8"] = float(np.std(rv_win)) if len(rv_win) > 1 else 0.0
        feats["rv_ratio"] = float(
            block_rvs[i - 1] / (feats["rv_mean_8"] + 1e-10)
        )

        if len(rv_win) >= 3:
            feats["rv_trend"] = float(
                np.polyfit(np.arange(len(rv_win)), rv_win, 1)[0]
            )
        else:
            feats["rv_trend"] = 0.0

        su = 0
        for k in range(i - 1, max(0, i - SEQ_LEN) - 1, -1):
            if k > 0 and block_rvs[k] > block_rvs[k - 1]:
                su += 1
            else:
                break
        feats["rv_streak_up"] = float(su)

        sd = 0
        for k in range(i - 1, max(0, i - SEQ_LEN) - 1, -1):
            if k > 0 and block_rvs[k] < block_rvs[k - 1]:
                sd += 1
            else:
                break
        feats["rv_streak_down"] = float(sd)
        feats["current_log_rv"] = float(log_rvs[i - 1])

        # ── C) Tech features (block-mean of current block) = 45 ──
        for j, cn in enumerate(fc_list):
            feats[f"tech_{cn}"] = float(block_feats[i - 1, j])

        # ── D) SPY cross-asset = 2 ──
        if spy_rvs is not None and i < len(spy_rvs):
            feats["spy_rv_cur"] = float(spy_rvs[max(0, i - 1)])
            s4 = spy_rvs[max(0, i - 4):i]
            feats["spy_rv_mean_4"] = float(np.mean(s4)) if len(s4) > 0 else 0.0
        else:
            feats["spy_rv_cur"] = np.nan
            feats["spy_rv_mean_4"] = np.nan

        # ── E) Daily features = 23 ──
        block_end_ts = block_meta.loc[i, "ts_end"]
        bd = block_end_ts.date() if hasattr(block_end_ts, "date") else block_end_ts

        # Find most recent daily row <= block date
        day_idx = -1
        for di in range(len(daily_dates) - 1, -1, -1):
            if daily_dates[di] <= bd:
                day_idx = di
                break

        if day_idx >= DAILY_LB:
            for d_off in range(DAILY_LB):
                di2 = day_idx - DAILY_LB + 1 + d_off
                for ci, cn in enumerate(_DAILY_COLS):
                    feats[f"d{d_off}_{cn}"] = float(daily_vals[di2, ci])

            drv_w = daily_vals[day_idx - DAILY_LB + 1 : day_idx + 1, 0]
            feats["daily_rv_mean_5"] = float(np.mean(drv_w))
            feats["daily_rv_std_5"] = float(np.std(drv_w))
            if len(drv_w) >= 3:
                feats["daily_rv_trend"] = float(
                    np.polyfit(np.arange(len(drv_w)), drv_w, 1)[0]
                )
            else:
                feats["daily_rv_trend"] = 0.0
        else:
            # Not enough daily history — fill with NaN
            for d_off in range(DAILY_LB):
                for cn in _DAILY_COLS:
                    feats[f"d{d_off}_{cn}"] = np.nan
            feats["daily_rv_mean_5"] = np.nan
            feats["daily_rv_std_5"] = np.nan
            feats["daily_rv_trend"] = np.nan

        rows.append(feats)
        timestamps.append(block_end_ts)

    result = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
    result.index.name = "timestamp"
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════

def build_vol_features(
    ticker: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the 119 Model D features for a single ticker.

    Loads the 1-hour OHLCV data, computes indicators, builds blocks,
    daily bars, and assembles the full feature vector.

    Returns a DataFrame with 119 columns and one row per valid block.
    The column order matches ``models/lightgbm_v3_flat/feature_names.json``.
    """
    from data.loader import load_data

    # ── Load 1-hour data ──
    df_1h = load_data(ticker, "1hour", config)
    logger.info(
        "Vol features for {}: loaded {} 1h bars", ticker, len(df_1h),
    )

    # ── Compute indicators ──
    df_ind = create_1h_indicators(df_1h)

    # ── Compute blocks ──
    block_meta, block_feats, block_rvs = compute_blocks(df_ind)
    logger.info(
        "Vol features for {}: {} blocks from {} bars",
        ticker, len(block_rvs), len(df_ind),
    )

    # ── Compute daily bars ──
    daily = compute_daily_bars(df_ind)

    # ── Load SPY blocks (cross-asset feature) ──
    spy_rvs = None
    if ticker != "SPY":
        try:
            df_spy = load_data("SPY", "1hour", config)
            spy_ind = create_1h_indicators(df_spy)
            _, _, spy_rvs = compute_blocks(spy_ind)
        except Exception as exc:
            logger.warning(
                "Could not load SPY for cross-asset features: {}", exc,
            )

    # ── Build features ──
    result = build_block_features(
        block_meta, block_feats, block_rvs, daily, spy_rvs,
    )

    # ── Reorder columns to match training feature order ──
    feat_json = (
        Path("models") / "lightgbm_v3_flat" / "feature_names.json"
    )
    if feat_json.exists():
        with open(feat_json) as f:
            expected_names = json.load(f)["features"]
        # Add any missing columns as NaN
        for col in expected_names:
            if col not in result.columns:
                result[col] = np.nan
        result = result[expected_names]

    logger.info(
        "Vol features for {}: {} rows × {} cols",
        ticker, len(result), result.shape[1],
    )
    return result
