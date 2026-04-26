"""Compare old AI baseline vs redesigned execution layer."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import glob
from pathlib import Path

from config.loader import load_config
config = load_config()

for f in glob.glob("backtest/cache/*.pkl"):
    Path(f).unlink()

from backtest.runner import run_backtest
from strategies.builtin.ai_strategy import AIStrategy

# ── 1. Old baseline: fixed 10% size, threshold 0.55, static TP=0.5% ──
print("=" * 60)
print("Running: Old Baseline (threshold=0.55, size=10%, static TP)")
print("=" * 60)

class OldBaseline(AIStrategy):
    name = "old_baseline"
    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            strength_threshold=0.55,
            bottom_threshold=0.50,
            dynamic_execution=False,
            **kwargs,
        )

old = run_backtest(config, strategy_name='ai', strategy_class=OldBaseline,
                   save_results=False)

for f in glob.glob("backtest/cache/*.pkl"):
    Path(f).unlink()

# ── 2. New design: threshold 0.50, vol-scaled sizing, dynamic TP ──
print("\n" + "=" * 60)
print("Running: Redesigned (threshold=0.50, vol-sized, dynamic TP)")
print("=" * 60)

# AIStrategy defaults are already the new design (0.50, dynamic=True)
new = run_backtest(config, strategy_name='ai', save_results=True)

# ── Report ──
def fmt(v, is_pct=False):
    if v is None:
        return "--"
    if is_pct:
        return f"{v:.2%}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

results = {"Old (0.55/10%/static)": old, "New (0.50/vol-sized/dyn)": new}

keys = [
    ("total_trades", False),
    ("long_trades", False),
    ("short_trades", False),
    ("total_return", True),
    ("annualized_return", True),
    ("sharpe_ratio", False),
    ("sortino_ratio", False),
    ("max_drawdown", True),
    ("win_rate", True),
    ("profit_factor", False),
    ("avg_trade_pnl", True),
    ("avg_win", True),
    ("avg_loss", True),
    ("avg_bars_held", False),
    ("calmar_ratio", False),
]

col_w = 28
metric_w = 22
header = f"{'Metric':<{metric_w}}"
for name in results:
    header += f" {name:>{col_w}}"
print("\n\n" + "=" * (metric_w + (col_w + 1) * len(results)))
print(header)
print("-" * (metric_w + (col_w + 1) * len(results)))
for key, is_pct in keys:
    row = f"{key:<{metric_w}}"
    for name, res in results.items():
        val = res.metrics.get(key)
        row += f" {fmt(val, is_pct):>{col_w}}"
    print(row)

print(f"\n{'exit_reasons':<{metric_w}}", end="")
for name, res in results.items():
    er = res.metrics.get("exit_reasons", {})
    s = f"SL:{er.get('sl',0)} TP:{er.get('tp',0)} MB:{er.get('max_bars',0)}"
    print(f" {s:>{col_w}}", end="")
print()

print(f"\n--- Direction Split ---")
for name, res in results.items():
    if not res.trades:
        print(f"  {name}: No trades")
        continue
    buys = [t for t in res.trades if t.signal.value == 'BUY']
    shorts = [t for t in res.trades if t.signal.value == 'SHORT']
    buy_wins = sum(1 for t in buys if t.pnl and t.pnl > 0)
    short_wins = sum(1 for t in shorts if t.pnl and t.pnl > 0)
    buy_pnl = sum(t.pnl for t in buys if t.pnl) / len(buys) if buys else 0
    short_pnl = sum(t.pnl for t in shorts if t.pnl) / len(shorts) if shorts else 0
    print(f"  {name}:")
    print(f"    BUY:   {len(buys):3d} trades, win={buy_wins}/{len(buys)}, avg_pnl={buy_pnl*100:+.3f}%")
    print(f"    SHORT: {len(shorts):3d} trades, win={short_wins}/{len(shorts)}, avg_pnl={short_pnl*100:+.3f}%")

# Position size distribution for new design
print(f"\n--- New Design: Position Size Distribution ---")
if new.trades:
    size_map = {}
    for t in new.trades:
        s = f"{t.size*100:.1f}%"
        if s not in size_map:
            size_map[s] = {"count": 0, "wins": 0, "pnls": []}
        size_map[s]["count"] += 1
        if t.pnl and t.pnl > 0:
            size_map[s]["wins"] += 1
        if t.pnl is not None:
            size_map[s]["pnls"].append(t.pnl)

    import numpy as np
    for s in sorted(size_map.keys()):
        d = size_map[s]
        avg = np.mean(d["pnls"]) * 100 if d["pnls"] else 0
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        print(f"  size={s}: {d['count']} trades, WR={wr:.1f}%, avg_pnl={avg:+.3f}%")
