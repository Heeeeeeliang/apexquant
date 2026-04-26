"""
Named configuration presets.

Each preset is a partial CONFIG dict that gets deep-merged on top of
the defaults.  Presets only override the keys they care about.

Usage::

    from config.presets import PRESETS, apply_preset
    from config.default import CONFIG

    new_cfg = apply_preset(CONFIG, "conservative")
"""

__all__ = ["PRESETS", "apply_preset", "list_presets"]

import warnings
from typing import Any

from config.default import deep_merge


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "_meta": {
            "name": "Default",
            "description": "Factory defaults — balanced parameters for general use.",
        },
        # No overrides — uses CONFIG as-is
    },

    "run1_baseline": {
        "_meta": {
            "name": "Run 1 (EMA+RSI Baseline)",
            "description": (
                "Technical-only baseline: EMA cross + RSI, no AI signals. "
                "Used as the control in ablation studies."
            ),
        },
        "strategy": {
            "signal_mode": "technical",
            "use_ai": False,
        },
        "backtest": {
            "position_size": 0.10,
            "max_positions": 6,
        },
    },

    "conservative": {
        "_meta": {
            "name": "Conservative",
            "description": (
                "Lower position sizing, tighter stops, higher vol gate. "
                "Prioritises capital preservation over returns."
            ),
        },
        "strategy": {
            "vol_threshold": 0.60,
        },
        "backtest": {
            "position_size": 0.08,
            "max_positions": 4,
        },
        "model": {
            "meta_tp": 0.006,
            "meta_sl": 0.003,
            "meta_mb": 24,
        },
    },

    "run8_tranche_exit": {
        "_meta": {
            "name": "Run 8 (Tranche Exit)",
            "description": (
                "Tranche exit configuration with larger positions, wider stops, "
                "lower vol gate. Achieved +68.5% / Sharpe 2.14."
            ),
        },
        "strategy": {
            "vol_threshold": 0.40,
        },
        "backtest": {
            "position_size": 0.35,
            "max_positions": 8,
        },
        "model": {
            "meta_tp": 0.020,
            "meta_sl": 0.006,
            "meta_mb": 48,
        },
    },

    "research": {
        "_meta": {
            "name": "Research",
            "description": (
                "All tickers enabled, technical signal mode, "
                "small positions for exploratory backtests."
            ),
        },
        "strategy": {
            "signal_mode": "technical",
            "vol_threshold": 0.30,
        },
        "data": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "TSLA", "SPY", "QQQ"],
        },
        "backtest": {
            "position_size": 0.05,
            "max_positions": 6,
        },
    },

    "run9_trailstop": {
        "_meta": {
            "name": "Run 9 (Trail Stop)",
            "description": (
                "Run 9 configuration: trail stop exits, wider ATR params, "
                "dynamic execution. Achieved +67% / Sharpe 2.14."
            ),
        },
        "strategy": {
            "signal_mode": "ai",
            "use_ai": True,
            "vol_threshold": 0.50,
            "tp_atr_mult": 3.0,
            "sl_atr_mult": 1.0,
            "dynamic_execution": True,
            "max_bars_held": 24,
        },
        "backtest": {
            "position_size": 0.25,
            "max_positions": 6,
            "commission": 0.001,
            "slippage": 0.0005,
            "min_commission": 1.0,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_presets() -> list[dict[str, str]]:
    """Return list of {id, name, description} for all presets."""
    result = []
    for preset_id, preset_dict in PRESETS.items():
        meta = preset_dict.get("_meta", {})
        result.append({
            "id": preset_id,
            "name": meta.get("name", preset_id),
            "description": meta.get("description", ""),
        })
    return result


# Deprecated aliases → canonical names
_DEPRECATED_ALIASES: dict[str, str] = {
    "aggressive": "run8_tranche_exit",
    "prod_trail_stop": "run9_trailstop",
}


def apply_preset(base_config: dict[str, Any], preset_id: str) -> dict[str, Any]:
    """Deep-merge a preset on top of a base config.

    Args:
        base_config: The base CONFIG dict.
        preset_id: Key into :data:`PRESETS`. Deprecated aliases
            (``"aggressive"`` → ``"run8_tranche_exit"``,
            ``"prod_trail_stop"`` → ``"run9_trailstop"``) are accepted
            with a :class:`DeprecationWarning`.

    Returns:
        New merged config dict. Does not mutate inputs.

    Raises:
        KeyError: If preset_id is not found.
    """
    if preset_id in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[preset_id]
        warnings.warn(
            f"Preset '{preset_id}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        preset_id = canonical

    if preset_id not in PRESETS:
        available = list(PRESETS.keys())
        raise KeyError(f"Unknown preset '{preset_id}'. Available: {available}")

    preset = PRESETS[preset_id]
    # Strip _meta before merging
    overrides = {k: v for k, v in preset.items() if k != "_meta"}
    return deep_merge(base_config, overrides)
