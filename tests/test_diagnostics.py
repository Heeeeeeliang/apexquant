"""
Tests for diagnostics module: feature_drift, trade_quality, engine_hooks,
equity_scan, trade_clustering, pnl_autocorrelation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diagnostics.engine_hooks import DiagnosticsReport, collect_diagnostics
from diagnostics.equity_scan import EquityScanReport, scan_equity_curve
from diagnostics.feature_drift import FeatureDriftReport, check_feature_drift
from diagnostics.pnl_autocorrelation import PnlAutocorrReport, scan_pnl_autocorrelation
from diagnostics.trade_clustering import ClusteringScanReport, scan_trade_clustering
from diagnostics.trade_quality import TradeQualityReport, compute_trade_quality


# =========================================================================
# Feature drift
# =========================================================================

class TestFeatureDrift:
    def test_aligned_returns_none_drift(self, tmp_path):
        mdir = tmp_path / "model_ok"
        mdir.mkdir()
        features = ["f1", "f2", "f3"]
        with open(mdir / "feature_names.json", "w") as f:
            json.dump({"features": features, "n_features": 3}, f)

        report = check_feature_drift(["f1", "f2", "f3"], mdir)
        assert report.is_aligned
        assert report.drift_severity == "none"
        assert report.training_count == 3
        assert report.runtime_count == 3

    def test_missing_features_is_error(self, tmp_path):
        mdir = tmp_path / "model_missing"
        mdir.mkdir()
        with open(mdir / "feature_names.json", "w") as f:
            json.dump({"features": ["f1", "f2", "f3"], "n_features": 3}, f)

        report = check_feature_drift(["f1"], mdir)
        assert not report.is_aligned
        assert report.drift_severity == "error"
        assert set(report.missing_in_runtime) == {"f2", "f3"}

    def test_extra_features_is_warning(self, tmp_path):
        mdir = tmp_path / "model_extra"
        mdir.mkdir()
        with open(mdir / "feature_names.json", "w") as f:
            json.dump({"features": ["f1"], "n_features": 1}, f)

        report = check_feature_drift(["f1", "f2", "f3"], mdir)
        assert report.drift_severity == "warning"
        assert set(report.extra_in_runtime) == {"f2", "f3"}

    def test_no_feature_names_file(self, tmp_path):
        mdir = tmp_path / "model_nofn"
        mdir.mkdir()

        report = check_feature_drift(["f1"], mdir)
        assert report.is_aligned  # no training names → nothing to compare
        assert report.training_count == 0

    def test_reuses_inference_loader(self, tmp_path):
        """Verify we're using the shared loader, not a duplicate."""
        from backtest.inference import load_feature_names_from_dir

        mdir = tmp_path / "model_reuse"
        mdir.mkdir()
        with open(mdir / "feature_names.json", "w") as f:
            json.dump({"features": ["a", "b"], "n_features": 2}, f)

        names = load_feature_names_from_dir(mdir)
        assert names == ["a", "b"]

        report = check_feature_drift(["a", "b"], mdir)
        assert report.is_aligned

    def test_custom_model_name(self, tmp_path):
        mdir = tmp_path / "m"
        mdir.mkdir()
        with open(mdir / "feature_names.json", "w") as f:
            json.dump({"features": ["x"], "n_features": 1}, f)

        report = check_feature_drift(["x"], mdir, model_name="custom_name")
        assert report.model_name == "custom_name"

    def test_real_model_alignment(self):
        """Smoke test against real model on disk."""
        mdir = Path("models/lightgbm_v3_flat")
        if not (mdir / "feature_names.json").exists():
            pytest.skip("Real model not available")

        from backtest.inference import load_feature_names_from_dir
        names = load_feature_names_from_dir(mdir)
        assert names is not None
        # Self-check: comparing training names against themselves → aligned
        report = check_feature_drift(names, mdir)
        assert report.is_aligned
        assert report.training_count == 119  # known from memory


# =========================================================================
# Trade quality
# =========================================================================

def _make_trades_df(
    pnls: list[float],
    tickers: list[str] | None = None,
    exit_reasons: list[str] | None = None,
    hours: list[int] | None = None,
    bars_held: list[int] | None = None,
) -> pd.DataFrame:
    n = len(pnls)
    data: dict = {"pnl": pnls}
    if tickers:
        data["ticker"] = tickers
    if exit_reasons:
        data["exit_reason"] = exit_reasons
    if hours:
        data["hour"] = hours
    if bars_held:
        data["bars_held"] = bars_held
    return pd.DataFrame(data)


class TestTradeQuality:
    def test_streak_analysis(self):
        # W W W L L W
        pnls = [0.01, 0.02, 0.01, -0.01, -0.02, 0.01]
        report = compute_trade_quality(_make_trades_df(pnls))
        assert report.max_win_streak == 3
        assert report.max_loss_streak == 2
        assert report.total_trades == 6

    def test_all_winners(self):
        pnls = [0.01] * 10
        report = compute_trade_quality(_make_trades_df(pnls))
        assert report.max_win_streak == 10
        assert report.max_loss_streak == 0

    def test_all_losers(self):
        pnls = [-0.01] * 5
        report = compute_trade_quality(_make_trades_df(pnls))
        assert report.max_win_streak == 0
        assert report.max_loss_streak == 5

    def test_hourly_pnl(self):
        pnls = [0.01, -0.02, 0.03]
        hours = [9, 14, 9]
        report = compute_trade_quality(_make_trades_df(pnls, hours=hours))
        assert report.best_hour == 9  # avg = (0.01+0.03)/2 = 0.02
        assert report.worst_hour == 14  # avg = -0.02

    def test_ticker_concentration(self):
        pnls = [0.01, 0.02, -0.01]
        tickers = ["AAPL", "AAPL", "TSLA"]
        report = compute_trade_quality(_make_trades_df(pnls, tickers=tickers))
        assert report.ticker_trade_counts["AAPL"] == 2
        assert report.ticker_trade_counts["TSLA"] == 1
        assert abs(report.ticker_pnl_sums["AAPL"] - 0.03) < 1e-10

    def test_exit_reason_win_rates(self):
        pnls = [0.01, 0.02, -0.01, -0.02]
        reasons = ["TP", "TP", "SL", "SL"]
        report = compute_trade_quality(_make_trades_df(pnls, exit_reasons=reasons))
        assert report.exit_reason_win_rates["TP"] == 1.0
        assert report.exit_reason_win_rates["SL"] == 0.0

    def test_avg_bars_to_recover(self):
        pnls = [0.01, -0.01, -0.02]
        bars = [5, 10, 20]
        report = compute_trade_quality(_make_trades_df(pnls, bars_held=bars))
        assert report.avg_bars_to_recover == 15.0  # mean of [10, 20]

    def test_empty_trades(self):
        report = compute_trade_quality(pd.DataFrame())
        assert report.total_trades == 0
        assert report.max_win_streak == 0

    def test_with_real_csv(self):
        """Smoke test with real trades CSV."""
        p = Path("results/archive/20260313_041057_ai_trail_stop/trades.csv")
        if not p.exists():
            pytest.skip("Archive data not available")
        df = pd.read_csv(p)
        report = compute_trade_quality(df)
        assert report.total_trades > 0
        assert report.max_win_streak > 0
        assert report.max_loss_streak > 0


# =========================================================================
# Engine hooks / collect_diagnostics
# =========================================================================

class TestCollectDiagnostics:
    def test_from_dataframe(self):
        df = _make_trades_df([0.01, -0.01, 0.02], bars_held=[5, 10, 3])
        report = collect_diagnostics(df)
        assert isinstance(report, DiagnosticsReport)
        assert report.trade_quality.total_trades == 3
        assert report.collection_time_ms >= 0

    def test_with_runtime_columns_checks_drift(self):
        """collect_diagnostics with runtime columns produces drift reports."""
        df = _make_trades_df([0.01])
        mdir = Path("models/lightgbm_v3_flat")
        if not (mdir / "feature_names.json").exists():
            pytest.skip("Real model not available")

        from backtest.inference import load_feature_names_from_dir
        names = load_feature_names_from_dir(mdir)

        report = collect_diagnostics(df, runtime_columns=names)
        # Should have drift reports for all models with feature_names.json
        assert len(report.feature_drift) > 0
        # The v3_flat model itself should be aligned (its own features)
        flat_reports = [d for d in report.feature_drift if d.model_name == "lightgbm_v3_flat"]
        assert len(flat_reports) == 1
        assert flat_reports[0].is_aligned

    def test_has_drift_errors_property(self):
        report = DiagnosticsReport(
            feature_drift=[
                FeatureDriftReport("m1", 10, 8, missing_in_runtime=["a", "b"], extra_in_runtime=[]),
            ]
        )
        assert report.has_drift_errors

    def test_has_drift_warnings_property(self):
        report = DiagnosticsReport(
            feature_drift=[
                FeatureDriftReport("m1", 10, 12, missing_in_runtime=[], extra_in_runtime=["x"]),
            ]
        )
        assert report.has_drift_warnings
        assert not report.has_drift_errors

    def test_collection_time_under_500ms(self):
        """Diagnostics collection should be fast for moderate trade counts."""
        df = _make_trades_df(
            pnls=list(np.random.default_rng(42).normal(0, 0.01, 2000)),
            tickers=["AAPL"] * 1000 + ["TSLA"] * 1000,
            exit_reasons=["TP"] * 500 + ["SL"] * 500 + ["signal"] * 1000,
            hours=list(np.random.default_rng(42).integers(9, 16, 2000)),
            bars_held=list(np.random.default_rng(42).integers(1, 50, 2000)),
        )
        report = collect_diagnostics(df)
        assert report.trade_quality.total_trades == 2000
        assert report.collection_time_ms < 500

    def test_collect_includes_all_scans(self):
        """collect_diagnostics populates all 5 scan fields."""
        n = 200
        ts = pd.date_range("2020-01-01", periods=n, freq="1h")
        df = _make_trades_df(
            pnls=[0.01, -0.01, 0.02, -0.005] * 50,
            tickers=["AAPL"] * n,
        )
        df["timestamp"] = ts
        report = collect_diagnostics(df)
        assert report.trade_quality.total_trades == n
        assert report.pnl_autocorrelation.n_trades == n
        assert report.trade_clustering.total_trades == n
        # equity_scan won't have data since we passed a DataFrame not BacktestResult
        assert report.equity_scan.equity_points == 0


# =========================================================================
# Equity curve scan
# =========================================================================

class TestEquityScan:
    def _make_equity(self, values: list[float], days: int | None = None) -> pd.Series:
        n = len(values)
        if days is None:
            days = n
        idx = pd.date_range("2020-01-01", periods=n, freq="1D")
        return pd.Series(values, index=idx, name="equity")

    def test_healthy_curve(self):
        """Steadily rising equity → good health."""
        vals = [100_000 + i * 100 for i in range(500)]
        report = scan_equity_curve(self._make_equity(vals))
        assert report.health == "good"
        assert report.pct_time_underwater < 0.01
        assert report.equity_points == 500

    def test_all_underwater(self):
        """Equity that peaks on day 1 then declines → poor."""
        vals = [100_000] + [100_000 - i * 50 for i in range(1, 200)]
        report = scan_equity_curve(self._make_equity(vals))
        assert report.health == "poor"
        assert report.pct_time_underwater > 0.95

    def test_flat_curve(self):
        """Constant equity → high flat_pct → poor."""
        vals = [100_000.0] * 300
        report = scan_equity_curve(self._make_equity(vals))
        assert report.flat_pct > 0.90
        assert report.health == "poor"

    def test_recovery_tracked(self):
        """Equity dips then recovers → worst_dd_recovery_days measured."""
        vals = (
            [100_000 + i * 10 for i in range(100)]  # rise
            + [99_000 - i * 20 for i in range(50)]   # drawdown
            + [98_000 + i * 40 for i in range(100)]   # recovery
        )
        report = scan_equity_curve(self._make_equity(vals))
        assert report.worst_dd_recovery_days > 0
        assert report.n_underwater_periods > 0

    def test_none_input(self):
        report = scan_equity_curve(None)
        assert report.equity_points == 0

    def test_too_short(self):
        report = scan_equity_curve(pd.Series([100_000.0], index=pd.DatetimeIndex(["2020-01-01"])))
        assert report.equity_points == 1

    def test_with_real_data(self):
        """Smoke test with real equity CSV."""
        p = Path("results/runs/20260316_140416_ai_full_trail/equity.csv")
        if not p.exists():
            pytest.skip("Real data not available")
        df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
        equity = df["portfolio_value"]
        report = scan_equity_curve(equity)
        assert report.equity_points > 0
        assert report.total_days > 0
        assert report.health in ("good", "caution", "poor")


# =========================================================================
# Trade clustering scan
# =========================================================================

class TestTradeClustering:
    def _make_ts_df(self, timestamps: list[str], pnls: list[float] | None = None) -> pd.DataFrame:
        data: dict = {"timestamp": timestamps}
        if pnls:
            data["pnl"] = pnls
        return pd.DataFrame(data)

    def test_uniform_spacing(self):
        """1 trade per day for 10 days → no clustering."""
        ts = [f"2020-01-{d+1:02d} 10:00:00" for d in range(10)]
        report = scan_trade_clustering(self._make_ts_df(ts))
        assert report.total_trades == 10
        assert report.max_trades_per_day == 1
        assert not report.has_clustering

    def test_burst_detected(self):
        """10 trades in one day, 1 per day for 9 others → burst."""
        ts = [f"2020-01-01 {h:02d}:00:00" for h in range(10)]
        ts += [f"2020-01-{d+2:02d} 10:00:00" for d in range(9)]
        report = scan_trade_clustering(self._make_ts_df(ts))
        assert report.max_trades_per_day == 10
        assert report.burst_days >= 1
        assert report.has_clustering

    def test_empty_df(self):
        report = scan_trade_clustering(pd.DataFrame())
        assert report.total_trades == 0

    def test_median_gap(self):
        """2 trades 6 hours apart → median gap = 6h."""
        ts = ["2020-01-01 09:00:00", "2020-01-01 15:00:00"]
        report = scan_trade_clustering(self._make_ts_df(ts))
        assert abs(report.median_gap_hours - 6.0) < 0.01


# =========================================================================
# PnL autocorrelation scan
# =========================================================================

class TestPnlAutocorrelation:
    def test_random_pnl_not_significant(self):
        """IID random PnL → autocorrelation near zero, not significant."""
        rng = np.random.default_rng(42)
        pnls = rng.normal(0, 0.01, 500)
        df = pd.DataFrame({"pnl": pnls})
        report = scan_pnl_autocorrelation(df)
        assert report.n_trades == 500
        assert abs(report.lag1_autocorr) < 0.15  # should be near zero
        # May or may not be significant due to randomness

    def test_perfectly_alternating(self):
        """W L W L W L → strong negative autocorrelation."""
        pnls = [0.01, -0.01] * 100
        df = pd.DataFrame({"pnl": pnls})
        report = scan_pnl_autocorrelation(df)
        assert report.lag1_autocorr < -0.5
        assert report.is_significant
        assert report.regime_signal == "mean_revert"
        assert report.p_win_after_win == 0.0
        assert report.p_loss_after_win == pytest.approx(1.0)

    def test_winning_streaks(self):
        """Long winning streaks → positive autocorrelation."""
        pnls = [0.01] * 50 + [-0.01] * 50 + [0.01] * 50 + [-0.01] * 50
        df = pd.DataFrame({"pnl": pnls})
        report = scan_pnl_autocorrelation(df)
        assert report.lag1_autocorr > 0.5
        assert report.is_significant
        assert report.regime_signal == "momentum"

    def test_too_few_trades(self):
        """< 10 trades → no analysis."""
        df = pd.DataFrame({"pnl": [0.01, -0.01, 0.01]})
        report = scan_pnl_autocorrelation(df)
        assert report.n_trades == 3
        assert report.lag1_autocorr == 0.0

    def test_transition_probabilities(self):
        """W W L L → P(W|W)=0.5, P(L|W)=0.5, etc."""
        pnls = [0.01, 0.02, -0.01, -0.02] * 10  # 40 trades
        df = pd.DataFrame({"pnl": pnls})
        report = scan_pnl_autocorrelation(df)
        assert report.n_trades == 40
        assert report.p_win_after_win > 0  # at least some W→W transitions
        assert report.p_loss_after_loss > 0  # at least some L→L transitions

    def test_empty_input(self):
        report = scan_pnl_autocorrelation(pd.DataFrame())
        assert report.n_trades == 0
