"""Test signal-driven exit on AAPL only, then optionally run full 8-ticker."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from collections import Counter
from config.loader import load_config

config = load_config()

# --- Override tickers to AAPL only for fast test ---
config['data']['tickers'] = ['AAPL']

# Print strategy exit params
strat = config.get('strategy', {})
print('=== Signal-Driven Exit Config ===')
print(f'  top_exit_threshold:    {strat.get("top_exit_threshold", 0.52)}')
print(f'  bottom_exit_threshold: {strat.get("bottom_exit_threshold", 0.52)}')
print(f'  hard_sl:               {strat.get("hard_sl", 0.015):.1%}')
print(f'  max_bars_held:         {strat.get("max_bars_held", 48)}')
print()

# Clear cache to force fresh run
from backtest.runner import clear_backtest_cache
cleared = clear_backtest_cache()
if cleared:
    print(f'Cleared {cleared} cached results')

from backtest.runner import run_backtest

print('=== Running AAPL-only backtest ===')
result = run_backtest(config, strategy_name='ai', save_results=False)

trades = result.trades
print(f'\nTotal trades: {len(trades)}')

if not trades:
    print('No trades generated. Check predictions.')
    sys.exit(1)

# Exit reason breakdown
reasons = Counter(t.exit_reason for t in trades)
print('\n=== Exit Reason Breakdown ===')
for reason, count in reasons.most_common():
    pct = count / len(trades) * 100
    print(f'  {reason:20s}: {count:4d} ({pct:5.1f}%)')

# Win rate
wins = sum(1 for t in trades if t.pnl and t.pnl > 0)
win_rate = wins / len(trades) * 100
print(f'\nWin rate: {wins}/{len(trades)} = {win_rate:.1f}%')

# Signal exit fraction
signal_exits = reasons.get('signal_exit', 0)
signal_pct = signal_exits / len(trades) * 100
print(f'Signal exits: {signal_exits}/{len(trades)} = {signal_pct:.1f}%')

# Metrics
m = result.metrics or {}
print(f'\nSharpe:       {m.get("sharpe_ratio", 0):.2f}')
print(f'Total return: {m.get("total_return", 0):.2%}')
print(f'Max drawdown: {m.get("max_drawdown", 0):.2%}')
print(f'Avg bars held: {m.get("avg_bars_held", 0):.1f}')

# Decision: run full backtest?
if win_rate > 45 and signal_pct > 50:
    print('\n--- AAPL looks good. Running full 8-ticker backtest... ---\n')

    config['data']['tickers'] = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG',
        'NVDA', 'TSLA', 'SPY', 'QQQ',
    ]
    cleared = clear_backtest_cache()
    result2 = run_backtest(config, strategy_name='ai', save_results=True)

    trades2 = result2.trades
    print(f'\n=== Full Backtest: {len(trades2)} trades ===')
    reasons2 = Counter(t.exit_reason for t in trades2)
    print('\nExit Reason Breakdown:')
    for reason, count in reasons2.most_common():
        pct = count / len(trades2) * 100
        print(f'  {reason:20s}: {count:4d} ({pct:5.1f}%)')

    wins2 = sum(1 for t in trades2 if t.pnl and t.pnl > 0)
    print(f'\nWin rate: {wins2}/{len(trades2)} = {wins2/len(trades2)*100:.1f}%')

    m2 = result2.metrics or {}
    print(f'Sharpe:       {m2.get("sharpe_ratio", 0):.2f}')
    print(f'Total return: {m2.get("total_return", 0):.2%}')
    print(f'Max drawdown: {m2.get("max_drawdown", 0):.2%}')
    print(f'Profit factor: {m2.get("profit_factor", 0):.2f}')
    print(f'Avg bars held: {m2.get("avg_bars_held", 0):.1f}')

    # Per-ticker breakdown
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for t in trades2:
        by_ticker[t.ticker].append(t)
    print('\nPer-ticker:')
    for ticker in sorted(by_ticker):
        tt = by_ticker[ticker]
        w = sum(1 for t in tt if t.pnl and t.pnl > 0)
        avg_pnl = sum(t.pnl for t in tt if t.pnl) / len(tt) * 100
        print(f'  {ticker:6s}: {len(tt):3d} trades, WR={w/len(tt)*100:5.1f}%, avg_pnl={avg_pnl:+.2f}%')
else:
    print(f'\n--- AAPL not passing gates (WR={win_rate:.1f}%, signal_exit={signal_pct:.1f}%). ---')
    print('Skipping full backtest. Inspect exit reasons above.')
