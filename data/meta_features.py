"""
Meta-label feature pipeline — replicates training-time CNN window construction.

Builds the 490-feature vector that ``lgb_bottom_v1`` / ``lgb_top_v1`` expect:

1. Load 15-min OHLCV → compute 61 manual indicators (66 total with OHLCV)
2. Load 1-hour  OHLCV → compute 40 indicators + 8 dynamics (53 total with OHLCV)
3. For each 15-min bar, build a short window (30 bars) and a long window
   (48 1-hour bars).
4. Run the Multi-Scale CNN to produce ``cnn_prob`` for each window.
5. Flatten last 5 short bars + last 3 long bars + cnn_prob = 490 features.

Usage::

    from data.meta_features import build_meta_features
    meta_df = build_meta_features("AAPL", config, cnn_task="bottom")
"""

__all__ = ["build_meta_features"]

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ── Window parameters (from training CFG) ─────────────────────
SHORT_WIN = 30   # 30 × 15-min bars
LONG_WIN = 48    # 48 × 1-hour bars
SHORT_TAIL = 5   # last 5 short bars flattened
LONG_TAIL = 3    # last 3 long bars flattened

# ── Short feature column names (OHLCV + 61 indicators) ────────
# Order matches _compute_manual_indicators from training Cell 9,
# preceded by OHLCV.  Training used numeric_cols(df_s) which
# returns columns in DataFrame order.
_SHORT_COLS = [
    "open", "high", "low", "close", "volume",
    "sma_5", "ema_5", "sma_10", "ema_10", "sma_20", "ema_20",
    "sma_50", "ema_50", "sma_100", "ema_100", "sma_200", "ema_200",
    "macd", "macd_signal", "macd_hist",
    "rsi_7", "rsi_14", "rsi_21",
    "bb_upper_20", "bb_middle_20", "bb_lower_20", "bb_bandwidth_20", "bb_percent_20",
    "bb_upper", "bb_middle", "bb_lower",
    "atr_7", "atr_14", "atr_21", "true_range",
    "stoch_k_14", "stoch_d_14",
    "adx_14", "plus_di_14", "minus_di_14",
    "cci_14", "cci_20", "willr_14", "mfi_14",
    "obv", "vwap",
    "volatility_5", "return_5", "log_return_5",
    "volatility_10", "return_10", "log_return_10",
    "volatility_20", "return_20", "log_return_20",
    "return_1", "log_return_1",
    "hl_spread", "oc_spread",
    "volume_sma_5", "volume_sma_10", "volume_sma_20", "volume_ratio",
    "hour", "day_of_week", "minute",
]
assert len(_SHORT_COLS) == 66, f"Expected 66 short cols, got {len(_SHORT_COLS)}"

# ── Long feature column names (OHLCV + 40 indicators + 8 dynamics) ──
# Reuses the vol_features._FC_LIST (45 cols) + 8 dynamics columns.
_LONG_BASE_COLS = [
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

_DYN_COLS = [
    "dyn_buyer_seller_ratio", "dyn_rsi_slope_12", "dyn_macd_accel",
    "dyn_volume_trend", "dyn_atr_change_12", "dyn_lower_shadow_ratio",
    "dyn_price_position_48", "dyn_consecutive_down",
]

_LONG_COLS = _LONG_BASE_COLS + _DYN_COLS
assert len(_LONG_COLS) == 53, f"Expected 53 long cols, got {len(_LONG_COLS)}"


# ═══════════════════════════════════════════════════════════════
#  15-min indicator computation (replicates _compute_manual_indicators)
# ═══════════════════════════════════════════════════════════════

def compute_15min_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 61 indicators on 15-min OHLCV data.

    Expects columns: open, high, low, close, volume (lowercase)
    with a DatetimeIndex. Returns DataFrame with 66 columns
    matching ``_SHORT_COLS``.
    """
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(np.float64)
    h = df["high"].astype(np.float64)
    lo = df["low"].astype(np.float64)
    o = df["open"].astype(np.float64)
    v = df["volume"].astype(np.float64)

    out["open"] = o
    out["high"] = h
    out["low"] = lo
    out["close"] = c
    out["volume"] = v

    # Moving Averages
    for w in [5, 10, 20, 50, 100, 200]:
        out[f"sma_{w}"] = c.rolling(w).mean()
        out[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI
    for w in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"rsi_{w}"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    out["bb_upper_20"] = mid + 2 * std
    out["bb_middle_20"] = mid
    out["bb_lower_20"] = mid - 2 * std
    bb_range = (out["bb_upper_20"] - out["bb_lower_20"])
    out["bb_bandwidth_20"] = bb_range / mid
    out["bb_percent_20"] = (c - out["bb_lower_20"]) / bb_range.replace(0, np.nan)
    out["bb_upper"] = out["bb_upper_20"]
    out["bb_middle"] = out["bb_middle_20"]
    out["bb_lower"] = out["bb_lower_20"]

    # ATR
    for w in [7, 14, 21]:
        tr = pd.concat([
            h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        out[f"atr_{w}"] = tr.rolling(w).mean()
    out["true_range"] = pd.concat([
        h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Stochastic
    low_min = lo.rolling(14).min()
    high_max = h.rolling(14).max()
    out["stoch_k_14"] = 100 * (c - low_min) / (high_max - low_min).replace(0, np.nan)
    out["stoch_d_14"] = out["stoch_k_14"].rolling(3).mean()

    # ADX
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-lo.diff()).clip(upper=0)
    tr14 = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr14.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    out["adx_14"] = dx.rolling(14).mean()
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di

    # CCI
    for w in [14, 20]:
        tp = (h + lo + c) / 3
        out[f"cci_{w}"] = (tp - tp.rolling(w).mean()) / (0.015 * tp.rolling(w).std())

    # Williams %R
    high_max14 = h.rolling(14).max()
    low_min14 = lo.rolling(14).min()
    out["willr_14"] = -100 * (high_max14 - c) / (high_max14 - low_min14).replace(0, np.nan)

    # MFI
    tp = (h + lo + c) / 3
    mf = tp * v
    pos_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=df.index).rolling(14).sum()
    neg_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=df.index).rolling(14).sum()
    out["mfi_14"] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

    # OBV
    out["obv"] = pd.Series(
        np.where(c > c.shift(1), v, np.where(c < c.shift(1), -v, 0)),
        index=df.index,
    ).cumsum()

    # VWAP
    tp_v = (h + lo + c) / 3
    out["vwap"] = (tp_v * v).cumsum() / v.cumsum().replace(0, np.nan)

    # Volatility / momentum
    for w in [5, 10, 20]:
        out[f"volatility_{w}"] = c.pct_change().rolling(w).std()
        out[f"return_{w}"] = c.pct_change(w)
        out[f"log_return_{w}"] = np.log(c / c.shift(w))

    out["return_1"] = c.pct_change(1)
    out["log_return_1"] = np.log(c / c.shift(1))

    # Price ratios / spreads
    out["hl_spread"] = (h - lo) / c.replace(0, np.nan)
    out["oc_spread"] = (c - o) / o.replace(0, np.nan)

    # Volume indicators
    for w in [5, 10, 20]:
        out[f"volume_sma_{w}"] = v.rolling(w).mean()
    out["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)

    # Time features
    if isinstance(df.index, pd.DatetimeIndex):
        out["hour"] = df.index.hour
        out["day_of_week"] = df.index.dayofweek
        out["minute"] = df.index.minute
    else:
        out["hour"] = 0
        out["day_of_week"] = 0
        out["minute"] = 0

    out = out.fillna(0.0)
    return out[_SHORT_COLS]


# ═══════════════════════════════════════════════════════════════
#  1-hour dynamics features (replicates add_dynamics_features)
# ═══════════════════════════════════════════════════════════════

def add_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 dynamics columns to a 1-hour DataFrame (in-place + return).

    Replicates ``add_dynamics_features()`` from training Cell 5 exactly.
    """
    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    n = len(df)

    # 1. buyer_seller_ratio (rolling 12)
    up_bar = (close > open_).astype(np.float64)
    down_bar = (close < open_).astype(np.float64)
    bsr = np.full(n, 1.0)
    for i in range(12, n):
        seg_up = up_bar[i - 12:i]
        seg_dn = down_bar[i - 12:i]
        seg_vol = vol[i - 12:i]
        up_vol = np.mean(seg_vol[seg_up == 1]) if seg_up.sum() > 0 else 0
        dn_vol = np.mean(seg_vol[seg_dn == 1]) if seg_dn.sum() > 0 else 1e-10
        bsr[i] = up_vol / (dn_vol + 1e-10)
    df["dyn_buyer_seller_ratio"] = bsr

    # 2. rsi_slope_12
    rsi_col = None
    for c_name in df.columns:
        if c_name.lower() in ("rsi_14", "rsi"):
            rsi_col = c_name
            break
    rsi_slope = np.zeros(n)
    if rsi_col is not None:
        rsi = df[rsi_col].values.astype(np.float64)
        x = np.arange(12, dtype=np.float64)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        for i in range(12, n):
            seg = rsi[i - 12:i]
            if np.any(np.isnan(seg)):
                continue
            rsi_slope[i] = np.sum((x - x_mean) * (seg - seg.mean())) / (x_var + 1e-10)
    df["dyn_rsi_slope_12"] = rsi_slope

    # 3. macd_accel
    macd_hist_col = None
    for c_name in df.columns:
        cl = c_name.lower()
        if "macd" in cl and ("hist" in cl or "diff" in cl):
            macd_hist_col = c_name
            break
    macd_accel = np.zeros(n)
    if macd_hist_col is not None:
        mh = df[macd_hist_col].values.astype(np.float64)
        for i in range(3, n):
            if not np.isnan(mh[i]) and not np.isnan(mh[i - 3]):
                macd_accel[i] = mh[i] - mh[i - 3]
    df["dyn_macd_accel"] = macd_accel

    # 4. volume_trend (slope of log(vol) over 12 bars)
    vol_trend = np.zeros(n)
    log_vol = np.log(vol + 1)
    x12 = np.arange(12, dtype=np.float64)
    x12m = x12.mean()
    x12v = ((x12 - x12m) ** 2).sum()
    for i in range(12, n):
        seg = log_vol[i - 12:i]
        vol_trend[i] = np.sum((x12 - x12m) * (seg - seg.mean())) / (x12v + 1e-10)
    df["dyn_volume_trend"] = vol_trend

    # 5. atr_change_12
    atr_col = None
    for c_name in df.columns:
        if c_name.lower() in ("atr_14", "atr"):
            atr_col = c_name
            break
    atr_change = np.zeros(n)
    if atr_col is not None:
        atr = df[atr_col].values.astype(np.float64)
        for i in range(12, n):
            if atr[i - 12] > 0 and not np.isnan(atr[i]) and not np.isnan(atr[i - 12]):
                atr_change[i] = (atr[i] - atr[i - 12]) / (atr[i - 12] + 1e-10)
    df["dyn_atr_change_12"] = atr_change

    # 6. lower_shadow_ratio (rolling 6)
    body_low = np.minimum(open_, close)
    full_range = high - low + 1e-10
    lower_shadow = (body_low - low) / full_range
    lsr = np.zeros(n)
    for i in range(6, n):
        lsr[i] = np.mean(lower_shadow[i - 6:i])
    df["dyn_lower_shadow_ratio"] = lsr

    # 7. price_position_48
    pp48 = np.full(n, 0.5)
    for i in range(48, n):
        hh = np.max(high[i - 48:i])
        ll = np.min(low[i - 48:i])
        rng = hh - ll
        pp48[i] = (close[i] - ll) / (rng + 1e-10) if rng > 0 else 0.5
    df["dyn_price_position_48"] = pp48

    # 8. consecutive_down
    consec = np.zeros(n)
    for i in range(1, n):
        if close[i] < close[i - 1]:
            consec[i] = consec[i - 1] + 1
        else:
            consec[i] = 0
    df["dyn_consecutive_down"] = consec

    return df


# ═══════════════════════════════════════════════════════════════
#  1-hour indicator computation (reuses vol_features logic + dynamics)
# ═══════════════════════════════════════════════════════════════

def compute_1h_with_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 45 base indicators + 8 dynamics = 53 columns on 1-hour data.

    Accepts either Title-case or lowercase OHLCV columns with DatetimeIndex.
    Returns DataFrame with 53 columns matching ``_LONG_COLS``.
    """
    from data.vol_features import create_1h_indicators

    # create_1h_indicators expects Title-case; ensure that
    df_input = df.copy()
    for low, cap in [("open", "Open"), ("high", "High"), ("low", "Low"),
                     ("close", "Close"), ("volume", "Volume")]:
        if low in df_input.columns and cap not in df_input.columns:
            df_input.rename(columns={low: cap}, inplace=True)

    # Get the 45 base indicators (same as vol pipeline)
    df_base = create_1h_indicators(df_input)

    # Add dynamics (needs lowercase OHLCV + rsi/macd columns)
    df_base = add_dynamics_features(df_base)

    # Select and order columns
    return df_base[_LONG_COLS]


# ═══════════════════════════════════════════════════════════════
#  15-min ↔ 1-hour timestamp alignment
# ═══════════════════════════════════════════════════════════════

def _build_hour_index_map(ts_15min: pd.DatetimeIndex, ts_1hour: pd.DatetimeIndex) -> np.ndarray:
    """For each 15-min timestamp, find the index of the latest 1-hour bar
    at or before that time. Returns array of indices (or -1 if none)."""
    ts_h = ts_1hour.asi8
    ts_s = ts_15min.asi8
    hour_idx = np.full(len(ts_s), -1, dtype=np.intp)
    j = 0
    for i in range(len(ts_s)):
        while j < len(ts_h) - 1 and ts_h[j + 1] <= ts_s[i]:
            j += 1
        if ts_h[j] <= ts_s[i]:
            hour_idx[i] = j
    return hour_idx


# ═══════════════════════════════════════════════════════════════
#  Feature name construction
# ═══════════════════════════════════════════════════════════════

def _build_feature_names() -> list[str]:
    """Build the 490 flattened feature names: s0_* ... s4_* l0_* ... l2_* cnn_prob."""
    names = []
    for si in range(SHORT_TAIL):
        for col in _SHORT_COLS:
            names.append(f"s{si}_{col}")
    for li in range(LONG_TAIL):
        for col in _LONG_COLS:
            names.append(f"l{li}_{col}")
    names.append("cnn_prob")
    return names


META_FEATURE_NAMES = _build_feature_names()
assert len(META_FEATURE_NAMES) == 490, f"Expected 490, got {len(META_FEATURE_NAMES)}"


# ═══════════════════════════════════════════════════════════════
#  Multi-Scale CNN model (replicates training architecture)
# ═══════════════════════════════════════════════════════════════

# Checkpoint paths for each task
_CNN_PATHS: dict[str, str] = {
    "bottom": "models/layer2/tp_bottom/cnn_bottom_v1/weights.pt",
    "top": "models/layer2/tp_top/cnn_top_v1/weights.pt",
}

# Module-level cache so each CNN is loaded only once
_cnn_cache: dict[str, tuple] = {}


def _load_cnn(task: str) -> tuple | None:
    """Load the CNN checkpoint for *task* ('bottom' or 'top').

    Returns ``(model, s_mu, s_sigma, l_mu, l_sigma)`` or *None* if
    PyTorch is unavailable or the checkpoint is missing.
    """
    if task in _cnn_cache:
        return _cnn_cache[task]

    path = Path(_CNN_PATHS.get(task, ""))
    if not path.exists():
        logger.warning("CNN checkpoint not found for task '{}': {}", task, path)
        _cnn_cache[task] = None  # type: ignore[assignment]
        return None

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        logger.warning("PyTorch not available — CNN inference disabled: {}", exc)
        _cnn_cache[task] = None  # type: ignore[assignment]
        return None

    class MultiScaleCNN(nn.Module):
        def __init__(self, n_short_feat: int, n_long_feat: int):
            super().__init__()
            self.short_branch = nn.Sequential(
                nn.Conv1d(n_short_feat, 32, kernel_size=5, padding=2),
                nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(), nn.BatchNorm1d(64),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.long_branch = nn.Sequential(
                nn.Conv1d(n_long_feat, 32, kernel_size=7, padding=3),
                nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(), nn.BatchNorm1d(64),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 1),
            )

        def forward(self, x_short, x_long):
            s = self.short_branch(x_short.transpose(1, 2))
            l_out = self.long_branch(x_long.transpose(1, 2))
            fused = torch.cat([s, l_out], dim=1)
            return self.head(fused).squeeze(-1)

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    n_sf = ckpt["n_short_feat"]
    n_lf = ckpt["n_long_feat"]

    model = MultiScaleCNN(n_sf, n_lf)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s_mu = ckpt["s_mu"].numpy()      # (n_short_feat,)
    s_sigma = ckpt["s_sigma"].numpy()
    l_mu = ckpt["l_mu"].numpy()      # (n_long_feat,)
    l_sigma = ckpt["l_sigma"].numpy()

    result = (model, s_mu, s_sigma, l_mu, l_sigma)
    _cnn_cache[task] = result
    logger.info(
        "Loaded CNN for '{}': n_short={}, n_long={} from {}",
        task, n_sf, n_lf, path,
    )
    return result


def _run_cnn_inference(
    cnn_tuple: tuple,
    short_windows: np.ndarray,
    long_windows: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Run CNN on paired windows, return sigmoid probabilities.

    Args:
        cnn_tuple: ``(model, s_mu, s_sigma, l_mu, l_sigma)`` from ``_load_cnn``.
        short_windows: ``(N, SHORT_WIN, 66)`` float32 array.
        long_windows:  ``(N, LONG_WIN, 53)`` float32 array.
        batch_size: Inference batch size.

    Returns:
        ``(N,)`` float32 array of CNN probabilities in [0, 1].
    """
    import torch

    model, s_mu, s_sigma, l_mu, l_sigma = cnn_tuple

    # Normalise using training-time statistics
    s_normed = (short_windows - s_mu) / (s_sigma + 1e-8)
    l_normed = (long_windows - l_mu) / (l_sigma + 1e-8)
    np.nan_to_num(s_normed, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(l_normed, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(short_windows)
    all_probs = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            s_t = torch.from_numpy(s_normed[start:end].astype(np.float32))
            l_t = torch.from_numpy(l_normed[start:end].astype(np.float32))
            logits = model(s_t, l_t)
            probs = torch.sigmoid(logits).numpy()
            all_probs[start:end] = probs

    return all_probs


# ═══════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════

def build_meta_features(
    ticker: str,
    config: dict[str, Any],
    cnn_task: str | None = None,
) -> pd.DataFrame:
    """Build the 490 meta-label features for a single ticker.

    Loads 15-min and 1-hour OHLCV data, computes indicators, builds
    paired windows, runs the CNN model (if available), flattens, and
    returns a DataFrame with 490 columns aligned to each 15-min bar
    where sufficient history exists.

    Args:
        ticker: Stock symbol.
        config: Full CONFIG dict.
        cnn_task: ``"bottom"`` or ``"top"`` — which CNN checkpoint to use
            for the ``cnn_prob`` feature.  If *None*, ``cnn_prob`` is NaN.

    Returns a DataFrame with 490 columns and a DatetimeIndex (15-min bars).
    """
    from data.loader import load_data

    # ── Load data ──
    df_15 = load_data(ticker, "15min", config)
    df_1h = load_data(ticker, "1hour", config)
    logger.info(
        "Meta features for {}: {} 15-min bars, {} 1-hour bars",
        ticker, len(df_15), len(df_1h),
    )

    # ── Normalise column names to lowercase ──
    for df in [df_15, df_1h]:
        rename = {}
        for cap, low in [("Open", "open"), ("High", "high"), ("Low", "low"),
                         ("Close", "close"), ("Volume", "volume")]:
            if cap in df.columns and low not in df.columns:
                rename[cap] = low
        if rename:
            df.rename(columns=rename, inplace=True)

    # ── Ensure DatetimeIndex ──
    for df in [df_15, df_1h]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # ── Strip timezone for consistency ──
    tz_15 = df_15.index.tz
    if df_15.index.tz is not None:
        df_15.index = df_15.index.tz_localize(None)
    if df_1h.index.tz is not None:
        df_1h.index = df_1h.index.tz_localize(None)

    # ── Compute indicators ──
    s_feat_df = compute_15min_indicators(df_15)
    l_feat_df = compute_1h_with_dynamics(df_1h)
    logger.info(
        "Meta features for {}: short={} cols, long={} cols",
        ticker, s_feat_df.shape[1], l_feat_df.shape[1],
    )

    # ── Convert to numpy (NaN → 0 like training) ──
    s_feat = np.nan_to_num(s_feat_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    l_feat = np.nan_to_num(l_feat_df.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # ── Build 15-min → 1-hour alignment map ──
    hour_idx_map = _build_hour_index_map(s_feat_df.index, l_feat_df.index)

    # ── Collect full windows + flattened features ──
    n_short = len(s_feat_df)
    n_feat = len(META_FEATURE_NAMES)
    n_short_flat = SHORT_TAIL * len(_SHORT_COLS)  # 5 × 66 = 330
    n_long_flat = LONG_TAIL * len(_LONG_COLS)      # 3 × 53 = 159

    flat_rows = []
    short_wins_full = []  # full 30-bar windows for CNN
    long_wins_full = []   # full 48-bar windows for CNN
    timestamps = []

    for i in range(SHORT_WIN, n_short):
        hi = hour_idx_map[i]
        if hi < LONG_WIN:
            continue  # not enough 1-hour history

        # Full windows for CNN
        sw_full = s_feat[i - SHORT_WIN:i]          # (30, 66)
        lw_full = l_feat[hi - LONG_WIN + 1:hi + 1]  # (48, 53)

        if sw_full.shape[0] != SHORT_WIN or lw_full.shape[0] != LONG_WIN:
            continue

        # Flattened tail for LGB features
        row = np.empty(n_feat, dtype=np.float32)
        row[:n_short_flat] = sw_full[-SHORT_TAIL:].ravel()       # last 5
        row[n_short_flat:n_short_flat + n_long_flat] = lw_full[-LONG_TAIL:].ravel()  # last 3
        row[-1] = np.nan  # placeholder for cnn_prob

        flat_rows.append(row)
        short_wins_full.append(sw_full)
        long_wins_full.append(lw_full)
        timestamps.append(s_feat_df.index[i])

    if not flat_rows:
        logger.warning("Meta features for {}: no valid windows", ticker)
        return pd.DataFrame(columns=META_FEATURE_NAMES)

    flat_arr = np.array(flat_rows)  # (N, 490)

    # ── CNN inference for cnn_prob ──
    cnn_tuple = _load_cnn(cnn_task) if cnn_task else None
    if cnn_tuple is not None:
        X_short = np.array(short_wins_full, dtype=np.float32)  # (N, 30, 66)
        X_long = np.array(long_wins_full, dtype=np.float32)    # (N, 48, 53)
        cnn_probs = _run_cnn_inference(cnn_tuple, X_short, X_long)
        flat_arr[:, -1] = cnn_probs
        logger.info(
            "CNN '{}' for {}: prob range [{:.4f}, {:.4f}], mean={:.4f}",
            cnn_task, ticker,
            float(cnn_probs.min()), float(cnn_probs.max()),
            float(cnn_probs.mean()),
        )
    else:
        if cnn_task:
            logger.warning(
                "CNN for '{}' not available — cnn_prob will be NaN", cnn_task,
            )

    result = pd.DataFrame(
        flat_arr,
        index=pd.DatetimeIndex(timestamps),
        columns=META_FEATURE_NAMES,
    )
    result.index.name = "timestamp"

    # Restore timezone if original data had one
    if tz_15 is not None:
        result.index = result.index.tz_localize(tz_15)

    logger.info(
        "Meta features for {}: {} rows × {} cols",
        ticker, len(result), result.shape[1],
    )
    return result
