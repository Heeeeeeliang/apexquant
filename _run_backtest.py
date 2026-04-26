"""Temporary script to run AI backtest and report results."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from config.loader import load_config
config = load_config()

# Confirm cost assumptions
bt = config['backtest']
print('=== Cost Assumptions ===')
print(f'  Initial capital: ${bt["initial_capital"]:,.0f}')
print(f'  Commission: {bt["commission"]*100:.1f}% of trade value')
print(f'  Slippage: {bt["slippage"]*100:.2f}%')
print(f'  Position size: {bt["position_size"]*100:.0f}% of portfolio')
print(f'  Max positions: {bt["max_positions"]}')
print()

from backtest.runner import run_backtest
result = run_backtest(config, strategy_name='ai', save_results=True)

print('\n=== AI Strategy Results ===')
print(f'Total Trades: {len(result.trades)}')

# Buy/Short split
if result.trades:
    buy_trades = [t for t in result.trades if getattr(t.signal, 'value', str(t.signal)).upper() == 'BUY']
    short_trades = [t for t in result.trades if getattr(t.signal, 'value', str(t.signal)).upper() == 'SHORT']
    print(f'  BUY trades:   {len(buy_trades)}')
    print(f'  SHORT trades: {len(short_trades)}')

    if buy_trades:
        buy_pnl = [t.pnl for t in buy_trades]
        print(f'  BUY  avg pnl: {sum(buy_pnl)/len(buy_pnl)*100:+.2f}%  wins: {sum(1 for p in buy_pnl if p>0)}/{len(buy_pnl)}')
    if short_trades:
        short_pnl = [t.pnl for t in short_trades]
        print(f'  SHORT avg pnl: {sum(short_pnl)/len(short_pnl)*100:+.2f}%  wins: {sum(1 for p in short_pnl if p>0)}/{len(short_pnl)}')

print()
if result.metrics:
    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

if result.trades:
    print(f'\nTrade details (first 30):')
    for t in result.trades[:30]:
        direction = getattr(t.signal, 'value', str(t.signal))
        print(f'  {t.ticker} {direction:5s} entry={t.entry_price:.2f} '
              f'exit={t.exit_price:.2f} pnl={t.pnl*100:+.2f}% '
              f'bars={t.bars_held} reason={t.exit_reason}')
else:
    print('\nNo trades generated.')
