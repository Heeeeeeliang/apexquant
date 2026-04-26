"""
Tests for analytics.verdict and analytics.attribution.

Covers:
- Verdict: 2+ fixture results per level (GREEN, YELLOW, RED)
- Attribution: by_exit_reason and by_conviction_tier
- Acceptance criteria for Run 9, Phase 12, and SHORT_BIAS scenarios
"""

import pandas as pd
import pytest

from analytics.attribution import by_conviction_tier, by_exit_reason
from analytics.verdict import Verdict, VerdictLevel, compute_verdict


# =========================================================================
# Verdict fixtures
# =========================================================================

# --- GREEN: Run 9 (trail stop) ---
_RUN9_METRICS = {
    "total_return": 0.6711,
    "sharpe_ratio": 2.14,
    "max_drawdown": -0.0990,
    "win_rate": 0.501,
    "total_trades": 1908,
    "long_trades": 192,
    "short_trades": 1716,
    "profit_factor": 2.17,
}

# --- GREEN: strong hypothetical ---
_GREEN_STRONG = {
    "total_return": 1.20,
    "sharpe_ratio": 3.5,
    "max_drawdown": -0.05,
    "win_rate": 0.60,
    "total_trades": 500,
    "long_trades": 250,
    "short_trades": 250,
    "profit_factor": 3.0,
}

# --- RED: Phase 12 buggy config ---
_PHASE12_BUGGY = {
    "total_return": -0.30,
    "sharpe_ratio": -0.50,
    "max_drawdown": -0.516,
    "win_rate": 0.402,
    "total_trades": 12744,
    "long_trades": 1200,
    "short_trades": 11544,
    "profit_factor": 0.60,
}

# --- RED: negative Sharpe, mild DD ---
_RED_NEG_SHARPE = {
    "total_return": -0.05,
    "sharpe_ratio": -0.30,
    "max_drawdown": -0.12,
    "win_rate": 0.45,
    "total_trades": 300,
    "long_trades": 150,
    "short_trades": 150,
    "profit_factor": 0.90,
}

# --- RED: catastrophic drawdown ---
_RED_CATASTROPHIC = {
    "total_return": 0.10,
    "sharpe_ratio": 0.20,
    "max_drawdown": -0.55,
    "win_rate": 0.35,
    "total_trades": 1000,
    "long_trades": 500,
    "short_trades": 500,
    "profit_factor": 1.10,
}

# --- YELLOW: SHORT bias (97.7% SHORT, but profitable) ---
_YELLOW_SHORT_BIAS = {
    "total_return": 0.50,
    "sharpe_ratio": 2.00,
    "max_drawdown": -0.10,
    "win_rate": 0.50,
    "total_trades": 2000,
    "long_trades": 46,
    "short_trades": 1954,
    "profit_factor": 2.0,
}

# --- YELLOW: marginal (profitable but low Sharpe) ---
_YELLOW_MARGINAL = {
    "total_return": 0.15,
    "sharpe_ratio": 0.80,
    "max_drawdown": -0.12,
    "win_rate": 0.48,
    "total_trades": 600,
    "long_trades": 300,
    "short_trades": 300,
    "profit_factor": 1.20,
}

# --- YELLOW: LONG bias ---
_YELLOW_LONG_BIAS = {
    "total_return": 0.40,
    "sharpe_ratio": 1.80,
    "max_drawdown": -0.08,
    "win_rate": 0.52,
    "total_trades": 1000,
    "long_trades": 950,
    "short_trades": 50,
    "profit_factor": 1.80,
}


# =========================================================================
# Verdict tests
# =========================================================================

class TestVerdictGreen:
    def test_run9_is_green(self):
        v = compute_verdict(_RUN9_METRICS)
        assert v.level == VerdictLevel.GREEN
        assert v.label == "PRODUCTION_READY"

    def test_strong_green(self):
        v = compute_verdict(_GREEN_STRONG)
        assert v.level == VerdictLevel.GREEN
        assert v.label == "PRODUCTION_READY"


class TestVerdictRed:
    def test_phase12_buggy_is_red(self):
        v = compute_verdict(_PHASE12_BUGGY)
        assert v.level == VerdictLevel.RED
        assert v.label == "LOSING"

    def test_negative_sharpe(self):
        v = compute_verdict(_RED_NEG_SHARPE)
        assert v.level == VerdictLevel.RED
        assert v.label == "LOSING"

    def test_catastrophic_drawdown(self):
        v = compute_verdict(_RED_CATASTROPHIC)
        assert v.level == VerdictLevel.RED

    def test_phase12_has_short_bias_in_details(self):
        v = compute_verdict(_PHASE12_BUGGY)
        assert any("SHORT" in d for d in v.details)


class TestVerdictYellow:
    def test_short_bias(self):
        v = compute_verdict(_YELLOW_SHORT_BIAS)
        assert v.level == VerdictLevel.YELLOW
        assert v.label == "SHORT_BIAS"

    def test_marginal(self):
        v = compute_verdict(_YELLOW_MARGINAL)
        assert v.level == VerdictLevel.YELLOW
        assert v.label == "MARGINAL"

    def test_long_bias(self):
        v = compute_verdict(_YELLOW_LONG_BIAS)
        assert v.level == VerdictLevel.YELLOW
        assert v.label == "LONG_BIAS"


class TestVerdictEdgeCases:
    def test_empty_metrics(self):
        v = compute_verdict({})
        assert v.level == VerdictLevel.RED

    def test_zero_trades(self):
        v = compute_verdict({"total_trades": 0, "total_return": -0.01})
        assert v.level == VerdictLevel.RED


# =========================================================================
# Attribution fixtures
# =========================================================================

def _make_trades_df(
    exit_reasons: list[str],
    pnls: list[float],
    conviction_tiers: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal trades DataFrame."""
    data: dict = {
        "exit_reason": exit_reasons,
        "pnl": pnls,
        "signal": ["BUY"] * len(pnls),
    }
    if conviction_tiers is not None:
        data["conviction_tier"] = conviction_tiers
    return pd.DataFrame(data)


# Mimics Run 8 exit reasons
_RUN8_EXIT_REASONS = (
    ["TrancheExit"] * 50
    + ["SignalReversal"] * 30
    + ["VolCollapse"] * 20
    + ["TP"] * 40
    + ["SL"] * 60
)
_RUN8_PNLS = (
    [0.005] * 50         # TrancheExit: mostly winners
    + [0.003] * 15 + [-0.002] * 15  # SignalReversal: mixed
    + [-0.001] * 20      # VolCollapse: mostly losers
    + [0.008] * 40       # TP: all winners
    + [-0.004] * 60      # SL: all losers
)


# =========================================================================
# Attribution tests
# =========================================================================

class TestByExitReason:
    def test_run8_has_expected_reasons(self):
        df = _make_trades_df(_RUN8_EXIT_REASONS, _RUN8_PNLS)
        result = by_exit_reason(df)
        reasons = set(result["exit_reason"])
        assert {"TrancheExit", "SignalReversal", "VolCollapse", "TP", "SL"} <= reasons

    def test_all_rows_nonzero(self):
        df = _make_trades_df(_RUN8_EXIT_REASONS, _RUN8_PNLS)
        result = by_exit_reason(df)
        for _, row in result.iterrows():
            assert row["trade_count"] > 0
            assert row["avg_pnl_pct"] != 0.0 or row["sum_pnl_pct"] != 0.0

    def test_contribution_sums_to_one(self):
        df = _make_trades_df(_RUN8_EXIT_REASONS, _RUN8_PNLS)
        result = by_exit_reason(df)
        total = result["contribution_pct"].sum()
        assert abs(total - 1.0) < 1e-10

    def test_avg_pnl_pct_is_percentage(self):
        """avg_pnl_pct should be the raw pnl mean (already a fraction)."""
        df = _make_trades_df(["TP", "TP"], [0.01, 0.02])
        result = by_exit_reason(df)
        assert abs(result.iloc[0]["avg_pnl_pct"] - 0.015) < 1e-10

    def test_empty_trades(self):
        result = by_exit_reason(pd.DataFrame())
        assert len(result) == 0

    def test_sorted_by_contribution(self):
        df = _make_trades_df(_RUN8_EXIT_REASONS, _RUN8_PNLS)
        result = by_exit_reason(df)
        contribs = result["contribution_pct"].tolist()
        assert contribs == sorted(contribs, reverse=True)


class TestByConvictionTier:
    def test_both_tiers_populated(self):
        tiers = ["high"] * 100 + ["mid"] * 200
        pnls = [0.005] * 100 + [0.002] * 200
        df = _make_trades_df(
            ["TP"] * 300, pnls, conviction_tiers=tiers
        )
        result = by_conviction_tier(df)
        tier_set = set(result["conviction_tier"])
        assert tier_set == {"high", "mid"}

    def test_high_tier_higher_avg_pnl(self):
        """High conviction should capture larger avg % moves."""
        tiers = ["high"] * 100 + ["mid"] * 200
        # High tier: larger avg pnl
        pnls = [0.008] * 100 + [0.002] * 200
        df = _make_trades_df(
            ["TP"] * 300, pnls, conviction_tiers=tiers
        )
        result = by_conviction_tier(df)
        high_row = result[result["conviction_tier"] == "high"].iloc[0]
        mid_row = result[result["conviction_tier"] == "mid"].iloc[0]
        assert high_row["avg_pnl_pct"] > mid_row["avg_pnl_pct"]

    def test_contribution_sums_to_one(self):
        tiers = ["high"] * 50 + ["mid"] * 50
        pnls = [0.01] * 50 + [0.005] * 50
        df = _make_trades_df(["TP"] * 100, pnls, conviction_tiers=tiers)
        result = by_conviction_tier(df)
        total = result["contribution_pct"].sum()
        assert abs(total - 1.0) < 1e-10

    def test_missing_conviction_tier_returns_empty(self):
        df = _make_trades_df(["TP", "SL"], [0.01, -0.01])
        result = by_conviction_tier(df)
        assert len(result) == 0

    def test_empty_string_tiers_ignored(self):
        df = _make_trades_df(
            ["TP", "SL"], [0.01, -0.01],
            conviction_tiers=["", ""],
        )
        result = by_conviction_tier(df)
        assert len(result) == 0

    def test_high_first_in_sort_order(self):
        tiers = ["mid"] * 50 + ["high"] * 50
        pnls = [0.005] * 100
        df = _make_trades_df(["TP"] * 100, pnls, conviction_tiers=tiers)
        result = by_conviction_tier(df)
        assert result.iloc[0]["conviction_tier"] == "high"


# =========================================================================
# Integration: real CSV loading
# =========================================================================

class TestWithRealData:
    """Smoke tests that load actual run data if available."""

    _ARCHIVE_TRAIL = "results/archive/20260313_041057_ai_trail_stop"

    @pytest.fixture
    def archive_trades(self):
        from pathlib import Path
        p = Path(self._ARCHIVE_TRAIL) / "trades.csv"
        if not p.exists():
            pytest.skip("Archive data not available")
        return pd.read_csv(p)

    def test_archive_has_conviction_tier(self, archive_trades):
        assert "conviction_tier" in archive_trades.columns

    def test_by_exit_reason_on_archive(self, archive_trades):
        result = by_exit_reason(archive_trades)
        assert len(result) > 0
        assert result["contribution_pct"].sum() == pytest.approx(1.0, abs=1e-8)

    def test_by_conviction_tier_on_archive(self, archive_trades):
        result = by_conviction_tier(archive_trades)
        assert len(result) > 0
