"""Signal-driven exit test — AAPL only."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from backtest.runner import run_backtest, clear_backtest_cache
from config.default import CONFIG

clear_backtest_cache()
CONFIG["data"]["tickers"] = ["AAPL"]

t0 = time.time()
res = run_backtest(CONFIG, strategy_name="ai", save_results=False)
elapsed = time.time() - t0

m = res.metrics
trades = res.trades

# Exit reason breakdown
from collections import Counter
reasons = Counter(t.exit_reason for t in trades)
buy_trades = [t for t in trades if t.signal.value == "BUY"]
short_trades = [t for t in trades if t.signal.value == "SHORT"]

print(f"\n=== AAPL Signal-Driven Exit Test ({elapsed:.1f}s) ===")
print(f"Total trades: {len(trades)} (BUY={len(buy_trades)}, SHORT={len(short_trades)})")
print(f"Total return: {m.get('total_return', 0)*100:+.2f}%")
print(f"Sharpe ratio: {m.get('sharpe_ratio', 0):.2f}")
print(f"Sortino ratio: {m.get('sortino_ratio', 0):.2f}")
print(f"Win rate: {m.get('win_rate', 0)*100:.1f}%")
print(f"Avg bars held: {m.get('avg_bars_held', 0):.1f}")
print(f"Max drawdown: {m.get('max_drawdown', 0)*100:.2f}%")
print(f"\nExit reason breakdown:")
total = len(trades)
for reason, count in reasons.most_common():
    pct = count / total * 100 if total > 0 else 0
    print(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)")
