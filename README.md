# ApexQuant

A research platform for building and backtesting quantitative trading strategies with pluggable AI models.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## What is ApexQuant

Most quant-research pipelines are tangled one-offs: a notebook that trains a model, another notebook that backtests it, a third that produces charts, and every change to the data layer cascades through all three. ApexQuant separates those concerns. Train models wherever you like (Colab, a GPU box, a laptop) and save each one as a `weights` file plus a small `meta.json`. The platform auto-discovers the folder, picks the right adapter, runs inference, backtests a strategy against the signals, and produces reports.

The architecture is folder-based auto-discovery, not a rigid registry. Drop `models/my_model/{weights.joblib, meta.json}` and it appears in the UI as Ready. Strategies consume signals by name (`tp_top`, `vol_prob`) rather than importing predictor modules, so swapping a model doesn't require touching strategy code.

ApexQuant was built while validating a three-layer cascaded decomposition framework that reaches +67.2% total return / Sharpe 2.14 on the bundled research dataset (NASDAQ ITCH 2020–2022, 8 tickers). The framework is one concrete use case; the platform itself is general — any signal architecture that fits the `Predictor` interface plugs in.

## Screenshot

![UI](docs/screenshot_dashboard.png)

*Most useful screens to capture: Dashboard (model readiness + preflight), Backtest Analysis (verdict cards + equity curve), Strategy Editor (in-UI rule authoring).*

## Quickstart — 60 seconds to a working backtest

```bash
git clone https://github.com/Heeeeeeliang/apexquant.git
cd apexquant
pip install -e .
python run_all.py --frontend
```

The Streamlit UI opens at `http://localhost:8501`. The demo runs on the bundled 6-month AAPL + SPY sample (`examples/sample_data/`) with the `run9_trailstop` preset and produces roughly 84 trades over the sample window. Full-scale data (the 2020–2022 NASDAQ set behind the headline numbers) lives in the [companion research repo](https://github.com/Heeeeeeliang/Applying-Deep-Time-Series-Learning-to-Stock-Forecasting-and-Quant-Trading).

## How ApexQuant works

**Step 1 — Connect your data.** Drop OHLCV CSVs into `data/` or point the loader at a Google Drive folder. Standard columns (`Open, High, Low, Close, Volume`, plus a `DatetimeIndex`) are auto-detected. Custom sources plug in via `data/loader.py`.

**Step 2 — Add your models.** Train in any environment. Save the weights plus a `meta.json` describing the adapter type and output signal name:

```json
{
  "adapter": "lightgbm",
  "output": "probability",
  "output_label": "tp_bottom",
  "task": "bottom",
  "direction": "long"
}
```

Sync the folder into `models/` and the registry picks it up on next launch — the model appears in the Dashboard as Ready.

**Step 3 — Write a strategy, run a backtest.** In the Strategy Editor, consume model signals by name (not by position) — e.g. `signals["tp_bottom"].prob > 0.5`. Click Run. The Backtest Analysis page returns Sharpe, win rate, max drawdown, an equity curve, and a per-layer attribution breakdown.

## Feature overview

- Folder-based model registry with auto-discovery
- Adapters for LightGBM and PyTorch CNN/LSTM; extensible via the `Predictor` base class
- Streamlit UI with Dashboard health checks, config presets, Backtest Analysis with verdict cards, and diagnostic scans
- Backtrader-based execution engine with custom fill control
- Strategy Editor for custom logic, plus a `strategies/user/` folder for saved strategies
- Preset library with reproducible runs (`run1_baseline`, `run8_tranche_exit`, `run9_trailstop`)
- Over 150 tests (unit + end-to-end) and a repo-level security audit

## Roadmap

**Near-term**
- Live trading integration (Interactive Brokers, Alpaca, selected CN brokers)
- Additional asset classes (crypto, futures, FX)

**Longer-term**
- New model adapters: BERT for news sentiment, Temporal Fusion Transformer, LLM-based signal generation
- Improved backtesting engine: better slippage modelling, market-impact simulation
- Risk management module: position sizing, drawdown control, correlation-aware portfolio construction

## How to extend

**Adding a model.** Subclass `Predictor` in [`predictors/base.py`](predictors/base.py), implement `predict(bar, context) -> PredictionResult`, register via `REGISTRY.register(instance)` at import time, and drop a matching `meta.json`. Existing adapters in `predictors/adapters/` (`vol_adapter.py`, `cnn_adapter.py`, `meta_adapter.py`) show the pattern for wrapping LightGBM and PyTorch checkpoints.

**Adding a data source.** Extend [`data/loader.py`](data/loader.py). The contract: given a ticker and a frequency, return a `pandas.DataFrame` with a `DatetimeIndex` and OHLCV columns. The bundled CSV backend is the reference implementation.

**Writing a custom strategy.** Subclass `BaseStrategy` in [`strategies/base.py`](strategies/base.py) and consume `AggregatedSignal` — strategies never import from `predictors/`. Drop a `.py` into `strategies/user/` or author in-UI via the Strategy Editor at `frontend/pages/3_②_Strategy_Editor.py`.

## Companion repository

The research findings, ablation studies, and training notebooks that produced the bundled demo weights live in a companion repository:

https://github.com/Heeeeeeliang/Applying-Deep-Time-Series-Learning-to-Stock-Forecasting-and-Quant-Trading

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{apexquant2026,
  author  = {Li, Heliang},
  title   = {ApexQuant: A Research Platform for Quantitative Trading with Pluggable AI Models},
  year    = {2026},
  url     = {https://github.com/Heeeeeeliang/apexquant}
}
```

A machine-readable form is in `CITATION.cff`.

## Acknowledgements

Built on Streamlit, Backtrader, LightGBM, PyTorch, and pandas-ta.
