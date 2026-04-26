"""
Tests for config schema, presets, guards, and preset_io.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from config.default import CONFIG, deep_merge
from config.presets import PRESETS, apply_preset, list_presets
from config.preset_io import (
    delete_user_preset,
    list_user_presets,
    load_user_preset,
    save_user_preset,
)
from config.schema import (
    KNOWN_TICKERS,
    GuardViolation,
    validate_config,
)


# =========================================================================
# Schema / Guards
# =========================================================================

class TestValidateConfig:
    def test_default_config_is_valid(self):
        violations = validate_config(CONFIG)
        errors = [v for v in violations if v.level == "error"]
        assert errors == [], f"Default CONFIG has errors: {errors}"

    def test_empty_tickers_is_error(self):
        cfg = deep_merge(CONFIG, {"data": {"tickers": []}})
        vs = validate_config(cfg)
        assert any(v.field == "data.tickers" and v.level == "error" for v in vs)

    def test_bad_signal_mode_is_error(self):
        cfg = deep_merge(CONFIG, {"strategy": {"signal_mode": "invalid"}})
        vs = validate_config(cfg)
        assert any(v.field == "strategy.signal_mode" for v in vs)

    def test_zero_capital_is_error(self):
        cfg = deep_merge(CONFIG, {"backtest": {"initial_capital": 0}})
        vs = validate_config(cfg)
        assert any(v.field == "backtest.initial_capital" for v in vs)

    def test_negative_commission_is_error(self):
        cfg = deep_merge(CONFIG, {"backtest": {"commission": -0.01}})
        vs = validate_config(cfg)
        assert any(v.field == "backtest.commission" for v in vs)

    def test_negative_slippage_is_error(self):
        cfg = deep_merge(CONFIG, {"backtest": {"slippage": -0.001}})
        vs = validate_config(cfg)
        assert any(v.field == "backtest.slippage" for v in vs)

    def test_bad_split_ratios_is_error(self):
        cfg = deep_merge(CONFIG, {
            "data": {"train_ratio": 0.9, "val_ratio": 0.5, "test_ratio": 0.5}
        })
        vs = validate_config(cfg)
        assert any("ratio" in v.field for v in vs)

    def test_inverted_tp_sl_is_warning(self):
        cfg = deep_merge(CONFIG, {"model": {"meta_tp": 0.001, "meta_sl": 0.010}})
        vs = validate_config(cfg)
        warnings = [v for v in vs if v.level == "warning" and "inverted" in v.message]
        assert len(warnings) == 1

    def test_bad_position_size_is_warning(self):
        cfg = deep_merge(CONFIG, {"backtest": {"position_size": 1.5}})
        vs = validate_config(cfg)
        assert any(v.field == "backtest.position_size" and v.level == "warning" for v in vs)

    def test_zero_tp_is_error(self):
        cfg = deep_merge(CONFIG, {"model": {"meta_tp": 0}})
        vs = validate_config(cfg)
        assert any(v.field == "model.meta_tp" for v in vs)

    def test_bad_data_source_is_error(self):
        cfg = deep_merge(CONFIG, {"data": {"source": "yahoo"}})
        vs = validate_config(cfg)
        assert any(v.field == "data.source" for v in vs)


class TestKnownTickers:
    def test_eight_tickers(self):
        assert len(KNOWN_TICKERS) == 8

    def test_contains_expected(self):
        for t in ("AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "TSLA", "SPY", "QQQ"):
            assert t in KNOWN_TICKERS


# =========================================================================
# Presets
# =========================================================================

class TestPresets:
    def test_all_presets_produce_valid_config(self):
        for preset_id in PRESETS:
            cfg = apply_preset(CONFIG, preset_id)
            errors = [v for v in validate_config(cfg) if v.level == "error"]
            assert errors == [], f"Preset '{preset_id}' produces errors: {errors}"

    def test_default_preset_is_identity(self):
        cfg = apply_preset(CONFIG, "default")
        # Should be identical since default has no overrides
        assert cfg["backtest"]["initial_capital"] == CONFIG["backtest"]["initial_capital"]

    def test_conservative_lower_position_size(self):
        cfg = apply_preset(CONFIG, "conservative")
        assert cfg["backtest"]["position_size"] == 0.08

    def test_run8_tranche_exit_higher_position_size(self):
        cfg = apply_preset(CONFIG, "run8_tranche_exit")
        assert cfg["backtest"]["position_size"] == 0.35

    def test_research_uses_technical_mode(self):
        cfg = apply_preset(CONFIG, "research")
        assert cfg["strategy"]["signal_mode"] == "technical"

    def test_run9_trailstop_uses_ai(self):
        cfg = apply_preset(CONFIG, "run9_trailstop")
        assert cfg["strategy"]["signal_mode"] == "ai"
        assert cfg["strategy"]["tp_atr_mult"] == 3.0

    def test_run1_baseline_uses_technical(self):
        cfg = apply_preset(CONFIG, "run1_baseline")
        assert cfg["strategy"]["signal_mode"] == "technical"
        assert cfg["strategy"]["use_ai"] is False

    def test_deprecated_aggressive_alias(self):
        with pytest.warns(DeprecationWarning, match="aggressive.*run8_tranche_exit"):
            cfg = apply_preset(CONFIG, "aggressive")
        assert cfg["backtest"]["position_size"] == 0.35

    def test_deprecated_prod_trail_stop_alias(self):
        with pytest.warns(DeprecationWarning, match="prod_trail_stop.*run9_trailstop"):
            cfg = apply_preset(CONFIG, "prod_trail_stop")
        assert cfg["strategy"]["tp_atr_mult"] == 3.0

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            apply_preset(CONFIG, "nonexistent")

    def test_does_not_mutate_base(self):
        original = deepcopy(CONFIG)
        apply_preset(CONFIG, "run8_tranche_exit")
        assert CONFIG["backtest"]["position_size"] == original["backtest"]["position_size"]

    def test_list_presets_has_all(self):
        presets = list_presets()
        ids = {p["id"] for p in presets}
        assert ids == set(PRESETS.keys())
        for p in presets:
            assert "name" in p
            assert "description" in p


# =========================================================================
# Preset I/O
# =========================================================================

class TestPresetIO:
    @pytest.fixture(autouse=True)
    def _use_tmp_dir(self, tmp_path, monkeypatch):
        """Redirect user presets to a temp directory."""
        import config.preset_io as pio
        monkeypatch.setattr(pio, "_USER_PRESETS_DIR", tmp_path / "user_presets")

    def test_save_and_load(self):
        cfg = {"backtest": {"initial_capital": 50_000}}
        path = save_user_preset("my_test", cfg)
        assert path.exists()

        loaded = load_user_preset("my_test")
        assert loaded["backtest"]["initial_capital"] == 50_000

    def test_cannot_overwrite_builtin(self):
        with pytest.raises(ValueError, match="built-in"):
            save_user_preset("conservative", {"x": 1})

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_user_preset("does_not_exist")

    def test_delete_preset(self):
        save_user_preset("deleteme", {"x": 1})
        assert delete_user_preset("deleteme") is True
        assert delete_user_preset("deleteme") is False

    def test_list_user_presets(self):
        save_user_preset("alpha", {"a": 1})
        save_user_preset("beta", {"b": 2})
        presets = list_user_presets()
        ids = {p["id"] for p in presets}
        assert ids == {"alpha", "beta"}

    def test_list_empty(self):
        presets = list_user_presets()
        assert presets == []
