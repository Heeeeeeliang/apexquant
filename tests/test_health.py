"""
Tests for analytics.health preflight checks.

Covers:
- check_models: GREEN (all ready), YELLOW (some missing), RED (none registered)
- check_data: GREEN (all tickers found), YELLOW (partial), RED (none/no dir)
- check_config: GREEN (valid), YELLOW (odd position_size), RED (missing sections)
- check_dependencies: GREEN (all importable)
- run_preflight: overall aggregation, can_run gate
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from analytics.health import (
    CheckStatus,
    HealthReport,
    Segment,
    check_config,
    check_data,
    check_dependencies,
    check_models,
    run_preflight,
)


# =========================================================================
# check_config
# =========================================================================

class TestCheckConfig:
    """Config validation — pure dict checks, no filesystem."""

    _GOOD_CONFIG = {
        "data": {
            "tickers": ["AAPL"],
            "features_dir": "data/features",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "strategy": {"signal_mode": "ai"},
        "backtest": {"initial_capital": 100_000, "position_size": 0.25},
    }

    def test_green_valid_config(self):
        seg = check_config(self._GOOD_CONFIG)
        assert seg.status == CheckStatus.GREEN
        assert seg.name == "Config"

    def test_green_details_populated(self):
        seg = check_config(self._GOOD_CONFIG)
        assert any("ai" in d for d in seg.details)
        assert any("100,000" in d for d in seg.details)

    def test_red_missing_section(self):
        cfg = {"data": {"tickers": ["AAPL"], "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}}
        seg = check_config(cfg)
        assert seg.status == CheckStatus.RED
        assert any("Missing" in d for d in seg.details)

    def test_red_bad_split_ratios(self):
        cfg = {
            **self._GOOD_CONFIG,
            "data": {**self._GOOD_CONFIG["data"], "train_ratio": 0.9, "val_ratio": 0.5, "test_ratio": 0.5},
        }
        seg = check_config(cfg)
        assert seg.status == CheckStatus.RED

    def test_red_invalid_signal_mode(self):
        cfg = {
            **self._GOOD_CONFIG,
            "strategy": {"signal_mode": "invalid_mode"},
        }
        seg = check_config(cfg)
        assert seg.status == CheckStatus.RED

    def test_red_zero_capital(self):
        cfg = {
            **self._GOOD_CONFIG,
            "backtest": {"initial_capital": 0, "position_size": 0.25},
        }
        seg = check_config(cfg)
        assert seg.status == CheckStatus.RED

    def test_yellow_bad_position_size(self):
        cfg = {
            **self._GOOD_CONFIG,
            "backtest": {"initial_capital": 100_000, "position_size": 1.5},
        }
        seg = check_config(cfg)
        assert seg.status == CheckStatus.YELLOW


# =========================================================================
# check_data
# =========================================================================

class TestCheckData:
    def test_red_no_tickers(self):
        seg = check_data({"data": {"tickers": [], "features_dir": "data/features"}})
        assert seg.status == CheckStatus.RED

    def test_red_missing_dir(self, tmp_path):
        seg = check_data({
            "data": {
                "tickers": ["AAPL"],
                "features_dir": str(tmp_path / "nonexistent"),
            }
        })
        assert seg.status == CheckStatus.RED

    def test_green_all_found(self, tmp_path):
        # Create mock data files
        (tmp_path / "AAPL_1hour.csv").write_text("open,high,low,close,volume\n")
        (tmp_path / "TSLA_1hour.csv").write_text("open,high,low,close,volume\n")
        seg = check_data({
            "data": {
                "tickers": ["AAPL", "TSLA"],
                "features_dir": str(tmp_path),
            }
        })
        assert seg.status == CheckStatus.GREEN

    def test_yellow_partial(self, tmp_path):
        (tmp_path / "AAPL_1hour.csv").write_text("open,high,low,close,volume\n")
        seg = check_data({
            "data": {
                "tickers": ["AAPL", "TSLA"],
                "features_dir": str(tmp_path),
            }
        })
        assert seg.status == CheckStatus.YELLOW
        assert any("TSLA" in d and "missing" in d for d in seg.details)

    def test_red_none_found(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        seg = check_data({
            "data": {
                "tickers": ["AAPL", "TSLA"],
                "features_dir": str(tmp_path),
            }
        })
        assert seg.status == CheckStatus.RED


# =========================================================================
# check_models
# =========================================================================

class TestCheckModels:
    def test_red_empty_registry(self):
        with patch("predictors.registry.REGISTRY") as mock_reg:
            mock_reg.list_all.return_value = []
            seg = check_models({})
            assert seg.status == CheckStatus.RED

    def test_green_all_ready(self):
        pred1 = MagicMock()
        pred1.is_ready.return_value = True
        pred2 = MagicMock()
        pred2.is_ready.return_value = True

        with patch("predictors.registry.REGISTRY") as mock_reg:
            mock_reg.list_all.return_value = ["vol", "cnn"]
            mock_reg._predictors = {"vol": pred1, "cnn": pred2}
            seg = check_models({})
            assert seg.status == CheckStatus.GREEN

    def test_yellow_some_missing(self):
        pred1 = MagicMock()
        pred1.is_ready.return_value = True
        pred2 = MagicMock()
        pred2.is_ready.return_value = False

        with patch("predictors.registry.REGISTRY") as mock_reg:
            mock_reg.list_all.return_value = ["vol", "cnn"]
            mock_reg._predictors = {"vol": pred1, "cnn": pred2}
            seg = check_models({})
            assert seg.status == CheckStatus.YELLOW

    def test_red_all_missing(self):
        pred1 = MagicMock()
        pred1.is_ready.return_value = False

        with patch("predictors.registry.REGISTRY") as mock_reg:
            mock_reg.list_all.return_value = ["vol"]
            mock_reg._predictors = {"vol": pred1}
            seg = check_models({})
            assert seg.status == CheckStatus.RED


# =========================================================================
# check_dependencies
# =========================================================================

class TestCheckDependencies:
    def test_green_in_test_env(self):
        """In our test env, core deps (pandas, numpy, etc.) are installed."""
        seg = check_dependencies()
        assert seg.status in (CheckStatus.GREEN, CheckStatus.YELLOW)
        assert seg.name == "Dependencies"
        assert any("pandas: ok" in d for d in seg.details)
        assert any("numpy: ok" in d for d in seg.details)

    def test_red_if_core_missing(self):
        """Simulate a core package import failure."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def fake_import(name, *args, **kwargs):
            if name == "lightgbm":
                raise ImportError("No module named 'lightgbm'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            seg = check_dependencies()
            assert seg.status == CheckStatus.RED
            assert any("lightgbm" in d for d in seg.details)


# =========================================================================
# run_preflight (integration)
# =========================================================================

class TestRunPreflight:
    def test_returns_six_segments(self):
        from config.default import CONFIG
        report = run_preflight(CONFIG)
        assert isinstance(report, HealthReport)
        assert len(report.segments) == 6
        names = {s.name for s in report.segments}
        assert names == {"Models", "Data", "Config", "Dependencies",
                         "FeatureAlignment", "VolProbDistribution"}

    def test_overall_is_worst(self):
        report = HealthReport(segments=[
            Segment(name="A", status=CheckStatus.GREEN),
            Segment(name="B", status=CheckStatus.YELLOW),
            Segment(name="C", status=CheckStatus.GREEN),
        ])
        assert report.overall == CheckStatus.YELLOW

    def test_overall_red_if_any_red(self):
        report = HealthReport(segments=[
            Segment(name="A", status=CheckStatus.GREEN),
            Segment(name="B", status=CheckStatus.RED),
        ])
        assert report.overall == CheckStatus.RED

    def test_can_run_false_if_red(self):
        report = HealthReport(segments=[
            Segment(name="A", status=CheckStatus.GREEN),
            Segment(name="B", status=CheckStatus.RED),
        ])
        assert report.can_run is False

    def test_can_run_true_if_yellow(self):
        report = HealthReport(segments=[
            Segment(name="A", status=CheckStatus.GREEN),
            Segment(name="B", status=CheckStatus.YELLOW),
        ])
        assert report.can_run is True

    def test_get_segment(self):
        report = HealthReport(segments=[
            Segment(name="Models", status=CheckStatus.GREEN),
            Segment(name="Data", status=CheckStatus.YELLOW),
        ])
        assert report.get_segment("Data").status == CheckStatus.YELLOW
        assert report.get_segment("Nonexistent") is None

    def test_empty_report_is_red(self):
        report = HealthReport()
        assert report.overall == CheckStatus.RED
        assert report.can_run is True  # no RED segments, just empty
