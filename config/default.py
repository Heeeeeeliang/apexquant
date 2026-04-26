"""
ApexQuant — Single-source configuration with helper utilities.

All tuneable parameters live in the ``CONFIG`` dict below.  Per-environment
overrides are applied via ``config/local.py`` or ``config/cloud.py`` using
:func:`deep_merge`.  Runtime overrides can be loaded from / saved to JSON
via :func:`load_config` and :func:`save_config`.

Helper functions
----------------
- ``get(key_path, default)`` — dot-notation access into CONFIG
- ``save_config(path)``      — serialise CONFIG to a JSON file
- ``load_config(path)``      — deserialise a JSON file and deep-merge into CONFIG
- ``deep_merge(base, over)`` — recursive dict merge (pure, returns new dict)

Usage::

    from config.default import CONFIG, get, save_config, load_config

    # Dot-notation access
    use_ai = get("strategy.use_ai")          # True
    lr     = get("model.meta_threshold")     # 0.45
    missing = get("foo.bar.baz", "fallback") # "fallback"

    # Persist / restore
    save_config("runs/my_experiment.json")
    load_config("runs/my_experiment.json")
"""

__all__ = ["CONFIG", "get", "save_config", "load_config", "deep_merge"]

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Master configuration dictionary
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    # ------------------------------------------------------------------
    # compute — where and how to run training / inference
    # ------------------------------------------------------------------
    "compute": {
        "backend": "colab",                         # "local" | "colab" | "gcloud" | "aws"
        "colab": {
            "drive_path": "/gdrive/MyDrive/apexquant/",
            "poll_interval": 30,                    # seconds
        },
        "gcloud": {
            "project": "",
            "region": "us-central1",
            "machine_type": "n1-standard-8",
            "accelerator": "NVIDIA_TESLA_T4",
        },
        "aws": {
            "instance_type": "ml.p3.2xlarge",
            "role_arn": "",
        },
        "local": {
            "device": "auto",                       # "auto" | "cpu" | "cuda" | "mps"
        },
    },

    # ------------------------------------------------------------------
    # data — sources, paths, tickers, split ratios
    # ------------------------------------------------------------------
    "data": {
        "source": "csv",                            # "csv" | "yfinance" | "databento_api"
        "raw_dir": "data/raw/",
        "features_dir": "data/features/",
        "tickers": [
            "AAPL", "MSFT", "GOOGL", "GOOG",
            "NVDA", "TSLA", "SPY", "QQQ",
        ],
        "freq_short": "15min",
        "freq_long": "1hour",
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },

    # ------------------------------------------------------------------
    # predictors — which layers to enable, calibration, staleness
    # ------------------------------------------------------------------
    "predictors": {
        "enabled": [],           # populated dynamically from REGISTRY
        "calibration": True,
        "staleness_check": True,
    },

    # ------------------------------------------------------------------
    # model — hyper-parameters shared across predictor layers
    # ------------------------------------------------------------------
    "model": {
        "vol_block_size": 12,
        "cnn_short_win": 30,
        "cnn_long_win": 48,
        "cnn_threshold": 0.5,
        "meta_threshold": 0.45,                     # auto-optimized on val set
        "meta_tp": 0.005,
        "meta_sl": 0.003,
        "meta_mb": 48,
    },

    # ------------------------------------------------------------------
    # aggregator — how predictor signals are combined
    # ------------------------------------------------------------------
    "aggregator": {
        "model_type": "lightgbm",                   # "lightgbm" | "logistic" | "weighted_vote"
        "retrain_on_new_predictor": True,
        "regime_aware": False,                      # future work
    },

    # ------------------------------------------------------------------
    # strategy — top-level strategy controls
    # ------------------------------------------------------------------
    "strategy": {
        "use_ai": True,
        "signal_mode": "ai",                        # "ai" | "technical" | "hybrid"
        "vol_threshold": 0.50,                      # primary vol gate
        "trend_bypass_period": 80,                  # bars to look back (~1.25 trading days at 15min)
        "trend_bypass_pct": 0.05,                   # 5% move in 80 bars = strong trend
        "trend_bypass_min_vol": 0.35,               # floor vol_prob even in trend bypass
        "bottom_threshold": 0.48,                   # tp_bottom outputs lower than tp_top; 0.48 = selective BUY signals
        "tp_atr_mult": 2.5,                         # TP = ATR_pct * 2.5
        "sl_atr_mult": 0.75,                        # SL = ATR_pct * 0.75
        "max_bars_held": 24,                        # time stop: evict zombie positions
    },

    # ------------------------------------------------------------------
    # backtest — execution simulation parameters
    # ------------------------------------------------------------------
    "backtest": {
        "initial_capital": 100_000,
        "commission": 0.001,
        "slippage": 0.0005,
        "min_commission": 1.0,                      # $1 minimum per trade side
        "position_size": 0.25,
        "max_positions": 6,
    },

    # ------------------------------------------------------------------
    # technical — classical indicator parameters
    # ------------------------------------------------------------------
    "technical": {
        "ema_fast": 8,
        "ema_slow": 21,
        "rsi_buy": 70,
        "rsi_sell": 30,
        "tp": 0.999,
        "sl": 0.050,
        "position_size": 0.10,
    },

    # ------------------------------------------------------------------
    # drive — Google Drive sync settings
    # ------------------------------------------------------------------
    "drive": {
        "data_folder_url": "",
        "models_folder_url": "",
        "cache_dir": "data/features",
        "models_dir": "models",
    },

    # ------------------------------------------------------------------
    # Google Drive API key (for public folder listing via Drive API v3).
    # Do NOT commit a real key. Set via env var and read with
    # os.getenv("GDRIVE_API_KEY", "") at the call site, or override in
    # a gitignored config/local_override.json.
    # ------------------------------------------------------------------
    "GDRIVE_API_KEY": "",

    # ------------------------------------------------------------------
    # llm — language-model provider settings
    # ------------------------------------------------------------------
    "llm": {
        "provider": "anthropic",                    # "openai" | "anthropic" | "local"
        "model": "claude-sonnet-4-20250514",
        "local_url": "http://localhost:11434/v1",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get(path: str, default: Any = None) -> Any:
    """Retrieve a CONFIG value via dot-separated key path.

    Args:
        path: Dot-separated key path, e.g. ``"strategy.use_ai"``.
        default: Value returned when the path does not exist.

    Returns:
        The resolved value, or *default* if any key segment is missing.

    Examples::

        >>> get("strategy.use_ai")
        True
        >>> get("compute.colab.poll_interval")
        30
        >>> get("nonexistent.deep.key", -1)
        -1
    """
    node: Any = CONFIG
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*.

    Nested dicts are merged key-by-key; all other types are replaced.

    Args:
        base: The base dictionary.
        override: Values that take precedence over *base*.

    Returns:
        A **new** dictionary — neither input is mutated.

    Example::

        >>> deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9}})
        {'a': {'x': 1, 'y': 9}}
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def save_config(path: str) -> Path:
    """Serialise the current CONFIG to a JSON file.

    Args:
        path: Destination file path (will be created / overwritten).

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Example::

        >>> save_config("results/runs/experiment_01.json")
        PosixPath('results/runs/experiment_01.json')
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(CONFIG, fh, indent=2, default=str)

    logger.info("Config saved to {}", file_path)
    return file_path.resolve()


def load_config(path: str) -> dict[str, Any]:
    """Load a JSON file and deep-merge it into the live CONFIG.

    The on-disk file acts as an override layer: keys present in the
    JSON replace the corresponding defaults, while keys absent from
    the JSON keep their current values.

    Args:
        path: Path to a JSON config file.

    Returns:
        The merged CONFIG dict (same object as the module-level ``CONFIG``).

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Example::

        >>> load_config("results/runs/experiment_01.json")
        {...}
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as fh:
        overrides: dict[str, Any] = json.load(fh)

    merged = deep_merge(CONFIG, overrides)
    CONFIG.clear()
    CONFIG.update(merged)

    logger.info("Config loaded and merged from {} ({} top-level keys)", file_path, len(overrides))
    return CONFIG
