"""Diagnose equity curve and Sharpe components."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from config.loader import load_config
config = load_config()

from backtest.runner import run_backtest
result = run_backtest(config, strategy_name='ai', save_results=False)

eq = result.equity_curve
print(f"Equity curve: {len(eq)} points, from {eq.index[0]} to {eq.index[-1]}")
print(f"First value: ${eq.iloc[0]:,.2f}, Last: ${eq.iloc[-1]:,.2f}")

# Daily resample
daily = eq.resample("1D").last().dropna()
daily_ret = daily.pct_change().dropna()

print(f"\nDaily returns: {len(daily_ret)} days")
print(f"  Mean:   {daily_ret.mean()*100:.6f}%")
print(f"  Std:    {daily_ret.std()*100:.6f}%")
print(f"  Min:    {daily_ret.min()*100:.4f}%")
print(f"  Max:    {daily_ret.max()*100:.4f}%")
print(f"  Zeros:  {(daily_ret == 0).sum()} / {len(daily_ret)}")
print(f"  Near-zero (<0.001%): {(daily_ret.abs() < 0.00001).sum()} / {len(daily_ret)}")

# Sharpe with rf=0.04
rf_daily = 0.04 / 252
excess = daily_ret - rf_daily
sharpe_rf4 = excess.mean() / excess.std() * np.sqrt(252)
print(f"\nSharpe (rf=4%): {sharpe_rf4:.2f}")

# Sharpe with rf=0
sharpe_rf0 = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
print(f"Sharpe (rf=0%): {sharpe_rf0:.2f}")

# Unique equity values
unique_vals = eq.nunique()
print(f"\nUnique equity values: {unique_vals} / {len(eq)} total points")
