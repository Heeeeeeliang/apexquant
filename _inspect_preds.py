"""Inspect what predictions are available on bars during backtest."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from config.loader import load_config
config = load_config()

from backtest.runner import _load_bars, _load_predictions
bars = _load_bars(config)
preds = _load_predictions(config, bars)

for ticker in sorted(preds.keys())[:2]:
    df = preds[ticker]
    print(f"\n{ticker}: {len(df)} rows, columns={list(df.columns)}")
    print(df.describe().to_string())
    break
