"""End-to-end smoke test for the bundled demo.

Runs the same code path the Streamlit UI "Run AI Backtest" button takes:

    REGISTRY.reload(models_dir=examples/sample_checkpoints)
    generate_predictions(config)        # uses the bundled checkpoints
    run_backtest(config, "ai")          # reads predictions, runs engine
    BacktestReporter(result)            # just constructs, no save

Uses the bundled `examples/sample_data/` (AAPL + SPY, 15-min,
2022-01-03 → 2022-06-30) and `examples/sample_checkpoints/` (the minimal
5-model set the `run9_trailstop` preset consumes).

Assertions — deliberately minimal:
  (a) the pipeline completes without raising
  (b) at least one trade is produced
  (c) the metrics dict exposes the four headline keys
      (total_return, sharpe, max_dd, n_trades — or their canonical aliases
      sharpe_ratio / max_drawdown / total_trades as returned by
      backtest.metrics.compute_metrics)

This is a **smoke test for the bundled demo only**. The sample window is
intentionally narrow (~6 months, 2 tickers) — the resulting metrics are
not comparable to the thesis headline numbers (+67.2% return, Sharpe
2.14, 1,908 trades over the full 2020-01 → 2022-09 window, 8 tickers).
Do not add assertions on specific metric values here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT   = Path(__file__).resolve().parents[1]
SAMPLE_DATA = REPO_ROOT / "examples" / "sample_data"
SAMPLE_CKPT = REPO_ROOT / "examples" / "sample_checkpoints"


@pytest.fixture
def demo_config():
    """Build a CONFIG dict pointing at the bundled demo assets only."""
    from config.default import CONFIG
    from config.presets import apply_preset

    cfg = deepcopy(CONFIG)
    cfg = apply_preset(cfg, "run9_trailstop")

    cfg.setdefault("data", {})
    cfg["data"]["tickers"]      = ["AAPL", "SPY"]
    cfg["data"]["freq_short"]   = "15min"
    cfg["data"]["features_dir"] = str(SAMPLE_DATA)
    cfg["data"]["raw_dir"]      = str(SAMPLE_DATA)
    cfg["data"]["models_dir"]   = str(SAMPLE_CKPT)
    cfg["data"]["source"]       = "csv"
    return cfg


def _has_key(d: dict, canonical: str, *aliases: str) -> bool:
    return canonical in d or any(a in d for a in aliases)


def test_demo_e2e_full_pipeline(demo_config, tmp_path):
    """UI-equivalent flow: reload registry on bundled checkpoints, generate
    predictions, run backtest, assert on structural outputs."""
    from backtest.inference import generate_predictions
    from backtest.reporter import BacktestReporter
    from backtest.runner import run_backtest
    from predictors.registry import REGISTRY

    # --- Point model discovery at the bundled checkpoint bundle ---
    assert SAMPLE_CKPT.exists(), f"sample checkpoints not found: {SAMPLE_CKPT}"
    REGISTRY.reload(models_dir=str(SAMPLE_CKPT))
    adapters = REGISTRY.list_all()
    assert adapters, (
        f"no adapters were auto-registered from {SAMPLE_CKPT}. "
        f"Check that each model dir contains a valid meta.json."
    )

    # --- Direct predictions into a fresh tmp dir so runs don't leak state ---
    cfg = deepcopy(demo_config)
    cfg.setdefault("data", {})
    cfg["data"]["predictions_dir"] = str(tmp_path / "predictions")

    # Stage 1: generate_predictions — exercises the checkpoint loading
    # path that the "🔮 Generate Predictions" UI button calls.
    summary = generate_predictions(cfg)
    assert summary and summary.get("tickers_processed", 0) > 0, (
        f"generate_predictions did not process any tickers; summary={summary!r}"
    )

    # Stage 2: run_backtest — exercises the "▶ Run AI Backtest" UI button.
    result = run_backtest(cfg, strategy_name="ai", save_results=False)

    # --- (a) completion is implicit (no exception raised) ---

    # --- (b) non-empty trade list ---
    trades = getattr(result, "trades", None) or []
    assert len(trades) > 0, (
        "zero trades produced on the bundled sample window. "
        "If this starts failing, widen the date slice in "
        "examples/sample_data/ rather than patching the assertion."
    )

    # --- (c) metrics dict exposes the required headline keys ---
    metrics = getattr(result, "metrics", None) or {}
    assert _has_key(metrics, "total_return"), (
        f"metrics missing 'total_return'; keys={sorted(metrics.keys())}"
    )
    assert _has_key(metrics, "sharpe", "sharpe_ratio"), (
        f"metrics missing 'sharpe'/'sharpe_ratio'; keys={sorted(metrics.keys())}"
    )
    assert _has_key(metrics, "max_dd", "max_drawdown", "maxdd"), (
        f"metrics missing 'max_dd'/'max_drawdown'; keys={sorted(metrics.keys())}"
    )
    assert _has_key(metrics, "n_trades", "total_trades", "num_trades"), (
        f"metrics missing 'n_trades'/'total_trades'; keys={sorted(metrics.keys())}"
    )

    # Reporter constructs cleanly on the result (read-only check, no save)
    reporter = BacktestReporter(result, run_id="demo_e2e_smoke")
    assert reporter is not None
