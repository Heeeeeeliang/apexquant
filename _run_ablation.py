"""Ablation study: 9 backtest runs isolating each optimization layer.

Each run saves full results (metrics.json, trades.csv, equity.csv, charts)
to results/runs/{config_name}/ via BacktestReporter.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import glob
from copy import deepcopy

# Suppress verbose logging
os.environ["LOGURU_LEVEL"] = "WARNING"
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from config.loader import load_config
base_config = load_config()

from backtest.runner import _load_bars, _load_predictions, _run_engine
from backtest.metrics import compute_metrics
from backtest.reporter import BacktestReporter
from strategies.builtin.ai_strategy import AIStrategy
from strategies.builtin.technical import TechnicalStrategy

# ---------------------------------------------------------------------------
# Load data ONCE (shared across all runs)
# ---------------------------------------------------------------------------
print("Loading data...")
bars_by_ticker = _load_bars(base_config)
predictions_by_ticker = _load_predictions(base_config, bars_by_ticker)
print(f"Loaded {len(bars_by_ticker)} tickers, "
      f"{len(predictions_by_ticker)} prediction sets\n")

# ---------------------------------------------------------------------------
# Helper: run one ablation variant and save via reporter
# ---------------------------------------------------------------------------
def run_variant(label, run_id, strategy, config):
    """Run a single backtest variant, save to results/runs/{run_id}/, return metrics dict."""
    # Clear cache
    for f in glob.glob("backtest/cache/*.pkl"):
        os.remove(f)

    result = _run_engine(
        strategy=strategy,
        config=config,
        bars_by_ticker=bars_by_ticker,
        predictions_by_ticker=predictions_by_ticker,
    )
    compute_metrics(result)

    # Set strategy name for reporter
    result.strategy_name = label
    result.config_snapshot = deepcopy(config)

    # Save full results via reporter
    reporter = BacktestReporter(result, run_id=run_id)
    out_dir = reporter.save_all()

    m = result.metrics or {}

    ret = m.get("total_return", 0.0)
    sh = m.get("sharpe_ratio", 0.0)
    wr = m.get("win_rate", 0.0)
    dd = m.get("max_drawdown", 0.0)
    pf = m.get("profit_factor", 0.0)
    trades = m.get("total_trades", 0)
    ab = m.get("avg_bars_held", 0.0)

    print(f"  {label}: Return={ret*100:+.2f}%  Sharpe={sh:.2f}  "
          f"WR={wr*100:.1f}%  MaxDD={dd*100:.2f}%  Trades={trades}  "
          f"-> {out_dir}")

    return {
        "label": label,
        "run_id": run_id,
        "total_return": ret,
        "sharpe": sh,
        "win_rate": wr,
        "max_dd": dd,
        "profit_factor": pf,
        "trades": trades,
        "avg_bars": ab,
    }


# ---------------------------------------------------------------------------
# Run 1: Pure Technical (NO AI)
# ---------------------------------------------------------------------------
cfg0 = deepcopy(base_config)
cfg0["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg0["_ablation"] = {
    "enable_tranches": False, "enable_signal_reversal": False,
    "enable_vol_collapse": False, "enable_preemption": False,
}
strat0 = TechnicalStrategy(cfg0)
r0 = run_variant("Pure Technical", "pure_technical", strat0, cfg0)


# ---------------------------------------------------------------------------
# Run 2: Pure Technical + Advanced Position Management
# ---------------------------------------------------------------------------
cfg0b = deepcopy(base_config)
cfg0b["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg0b["_ablation"] = {
    "enable_tranches": True, "enable_signal_reversal": True,
    "enable_vol_collapse": True, "enable_preemption": False,
}
strat0b = TechnicalStrategy(cfg0b)
r0b = run_variant("Tech + Position Mgmt", "tech_position_mgmt", strat0b, cfg0b)


# ---------------------------------------------------------------------------
# Run 3: Pure Technical + Conviction Sizing only
# ---------------------------------------------------------------------------
cfg0c = deepcopy(base_config)
cfg0c["backtest"].update({"position_size": 0.30, "max_positions": 6})
cfg0c["_ablation"] = {
    "enable_tranches": False, "enable_signal_reversal": False,
    "enable_vol_collapse": False, "enable_preemption": False,
}
strat0c = TechnicalStrategy(cfg0c)
# Patch sizing to 30% (mimic high-conviction sizing without vol_prob)
def _size_30(bar):
    return 0.30
strat0c.get_position_size = _size_30
r0c = run_variant("Tech + Sizing Only", "tech_sizing_only", strat0c, cfg0c)


# ---------------------------------------------------------------------------
# Run 4: AI Baseline (no optimizations, original sizing)
# ---------------------------------------------------------------------------
cfg1 = deepcopy(base_config)
cfg1["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg1["strategy"]["trend_bypass_pct"] = 999.0  # disable trend bypass
cfg1["_ablation"] = {
    "enable_tranches": False, "enable_signal_reversal": False,
    "enable_vol_collapse": False, "enable_preemption": False,
}
strat1 = AIStrategy.from_config(cfg1)
# Patch to original sizing (20%/5%)
strat1._original_get_size = True
def _size_20_5(bar, _s=strat1):
    if not _s.dynamic_execution:
        return 0.10
    return 0.20 if _s._current_vol_prob > 0.7 else 0.05
strat1.get_position_size = _size_20_5
r1 = run_variant("AI Baseline", "ai_baseline", strat1, cfg1)


# ---------------------------------------------------------------------------
# Run 5: +Trend bypass
# ---------------------------------------------------------------------------
cfg2 = deepcopy(base_config)
cfg2["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg2["strategy"]["trend_bypass_pct"] = 0.05  # enable
cfg2["_ablation"] = {
    "enable_tranches": False, "enable_signal_reversal": False,
    "enable_vol_collapse": False, "enable_preemption": False,
}
strat2 = AIStrategy.from_config(cfg2)
def _size_20_5_v2(bar, _s=strat2):
    if not _s.dynamic_execution:
        return 0.10
    return 0.20 if _s._current_vol_prob > 0.7 else 0.05
strat2.get_position_size = _size_20_5_v2
r2 = run_variant("AI + Trend Bypass", "ai_trend_bypass", strat2, cfg2)


# ---------------------------------------------------------------------------
# Run 6: +3-tranche exit
# ---------------------------------------------------------------------------
cfg3 = deepcopy(base_config)
cfg3["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg3["strategy"]["trend_bypass_pct"] = 0.05
cfg3["_ablation"] = {
    "enable_tranches": True, "enable_signal_reversal": False,
    "enable_vol_collapse": False, "enable_preemption": False,
}
strat3 = AIStrategy.from_config(cfg3)
def _size_20_5_v3(bar, _s=strat3):
    if not _s.dynamic_execution:
        return 0.10
    return 0.20 if _s._current_vol_prob > 0.7 else 0.05
strat3.get_position_size = _size_20_5_v3
r3 = run_variant("AI + Tranche Exit", "ai_tranche_exit", strat3, cfg3)


# ---------------------------------------------------------------------------
# Run 7: +Signal reversal
# ---------------------------------------------------------------------------
cfg4 = deepcopy(base_config)
cfg4["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg4["strategy"]["trend_bypass_pct"] = 0.05
cfg4["_ablation"] = {
    "enable_tranches": True, "enable_signal_reversal": True,
    "enable_vol_collapse": False, "enable_preemption": False,
}
strat4 = AIStrategy.from_config(cfg4)
def _size_20_5_v4(bar, _s=strat4):
    if not _s.dynamic_execution:
        return 0.10
    return 0.20 if _s._current_vol_prob > 0.7 else 0.05
strat4.get_position_size = _size_20_5_v4
r4 = run_variant("AI + Signal Reversal", "ai_signal_reversal", strat4, cfg4)


# ---------------------------------------------------------------------------
# Run 8: +Vol collapse
# ---------------------------------------------------------------------------
cfg5 = deepcopy(base_config)
cfg5["backtest"].update({"position_size": 0.10, "max_positions": 8})
cfg5["strategy"]["trend_bypass_pct"] = 0.05
cfg5["_ablation"] = {
    "enable_tranches": True, "enable_signal_reversal": True,
    "enable_vol_collapse": True, "enable_preemption": False,
}
strat5 = AIStrategy.from_config(cfg5)
def _size_20_5_v5(bar, _s=strat5):
    if not _s.dynamic_execution:
        return 0.10
    return 0.20 if _s._current_vol_prob > 0.7 else 0.05
strat5.get_position_size = _size_20_5_v5
r5 = run_variant("AI + Vol Collapse", "ai_vol_collapse", strat5, cfg5)


# ---------------------------------------------------------------------------
# Run 9: +Conviction sizing (30%/8%) + Preemption (FULL)
# ---------------------------------------------------------------------------
cfg6 = deepcopy(base_config)
cfg6["backtest"].update({"position_size": 0.25, "max_positions": 6})
cfg6["strategy"]["trend_bypass_pct"] = 0.05
cfg6["_ablation"] = {
    "enable_tranches": True, "enable_signal_reversal": True,
    "enable_vol_collapse": True, "enable_preemption": True,
}
strat6 = AIStrategy.from_config(cfg6)
# Uses current production sizing: 30%/8% (already in ai_strategy.py)
r6 = run_variant("AI Full (Production)", "ai_full", strat6, cfg6)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
all_results = [r0, r0b, r0c, r1, r2, r3, r4, r5, r6]

print("\n")
print("=" * 110)
print("ABLATION STUDY — SUMMARY")
print("=" * 110)
print(f"{'#':<4s} {'Run ID':<22s} {'Label':<24s} {'Return':>8s} {'Sharpe':>7s} {'WinRate':>8s} "
      f"{'MaxDD':>8s} {'PF':>6s} {'Trades':>7s} {'AvgBars':>8s}")
print("-" * 110)

for i, row in enumerate(all_results):
    print(f"{i+1:<4d} {row['run_id']:<22s} {row['label']:<24s} "
          f"{row['total_return']*100:>+7.2f}% "
          f"{row['sharpe']:>7.2f} "
          f"{row['win_rate']*100:>7.1f}% "
          f"{row['max_dd']*100:>7.2f}% "
          f"{row['profit_factor']:>6.2f} "
          f"{row['trades']:>7d} "
          f"{row['avg_bars']:>8.1f}")

print("=" * 110)

# Incremental contribution
print("\nINCREMENTAL CONTRIBUTION OF EACH LAYER:")
print("-" * 70)
for i in range(1, len(all_results)):
    prev = all_results[i - 1]
    curr = all_results[i]
    d_ret = (curr["total_return"] - prev["total_return"]) * 100
    d_sh = curr["sharpe"] - prev["sharpe"]
    d_dd = (curr["max_dd"] - prev["max_dd"]) * 100
    print(f"  {curr['label']:<26s}  "
          f"Return: {d_ret:>+6.2f}%  "
          f"Sharpe: {d_sh:>+5.2f}  "
          f"MaxDD: {d_dd:>+6.2f}%")

print("\nAll 9 runs saved to results/runs/")
