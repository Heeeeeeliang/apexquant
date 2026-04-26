# ApexQuant

A quant research platform built on a three-layer cascaded decomposition framework — designed in response to the ~50% directional-accuracy ceiling of direct price prediction.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green) ![Tests passing](https://img.shields.io/badge/tests-157%20passed-brightgreen) ![Last commit](https://img.shields.io/github/last-commit/Heeeeeeliang/apexquant)

Across 9 mainstream methods (ARIMA, VAR, GARCH variants, LSTM, GRU, Transformer, VMD-LSTM, XGBoost, CatBoost), direct next-bar price-direction prediction plateaus at ~50% directional accuracy under strict no-leakage conditions. ApexQuant responds to this ceiling by decomposing the problem into three cascaded sub-tasks — volatility, turning-point detection, and direction inference — each solved by a specialised model.

## Headline results

| Metric | Run 11 (Trail Stop, final) | EMA + RSI baseline |
|---|---:|---:|
| Total Return | **+67.2 %** | −43.6 % |
| Sharpe Ratio | **2.14** | −1.52 |
| Max Drawdown | **−9.9 %** | −48.0 % |
| Total Trades | 1,908 | 2,341 |
| Profit Factor | 2.17 | 0.96 |
| Win Rate | 50.1 % | 30.7 % |

Backtest scope: 8 NASDAQ tickers (AAPL, MSFT, GOOGL, NVDA, TSLA, SPY, QQQ, GOOG), 15-min + 1-hour bars, 2020-01-07 → 2022-09-30, 70 / 10 / 20 chronological split, 10 bps commission and 5 bps slippage per side. Source of truth: [`results/ablation_table.csv`](https://github.com/Heeeeeeliang/Applying-Deep-Time-Series-Learning-to-Stock-Forecasting-and-Quant-Trading/blob/main/results/ablation_table.csv) in the research repo.

## Architecture

![Three-layer cascade](https://raw.githubusercontent.com/Heeeeeeliang/Applying-Deep-Time-Series-Learning-to-Stock-Forecasting-and-Quant-Trading/main/docs/figures/fig_system_architecture.png)

**Layer 1 — Volatility gate.** A LightGBM regressor reads block-aggregated 1-hour features and predicts the next period's realised volatility (directional accuracy 83.6 %). A threshold-based gate blocks roughly half of all bars, removing regimes where the ATR-scaled stops are too tight relative to noise.

**Layer 2 — Turning-point detection.** A multi-scale CNN with 15-min and 1-hour input branches identifies likely local maxima and minima (AUC ~0.80). A per-side (top / bottom) meta-label LightGBM then filters false positives using the CNN probability plus vol- and trend-conditioning features.

**Layer 3 — Direction inference.** The cascaded top-signal and bottom-signal probabilities, together with the Layer-1 gate, determine trade entry, exit, and position size. Strategies consume the aggregator's output via an `AggregatedSignal` interface; they never import from `predictors/`.

## Five-minute quickstart

```bash
# Clone
git clone https://github.com/Heeeeeeliang/apexquant.git
cd apexquant

# Install (Python 3.10+)
pip install -e .

# Run the bundled end-to-end demo backtest
pytest tests/test_demo_e2e.py -v

# Or launch the Streamlit UI
python run_all.py --frontend
```

The demo runs on bundled 6-month AAPL + SPY sample data (`examples/sample_data/`) with pre-trained checkpoints (`examples/sample_checkpoints/`). Expected output: **84 trades, +2.2 % total return, Sharpe 1.31** over the sample window — a smaller-scale reproduction of the cascade behaviour. The full-scale results in the headline table require the full dataset from the research repo.

## Try it in the UI

After `python run_all.py --frontend`, the Streamlit app opens at `http://localhost:8501`.

1. **Config page** — select preset `run9_trailstop` (the Trail-Stop configuration behind the headline numbers).
2. **Preflight Checks** — run the six readiness checks (Models, Data, Config, Dependencies, FeatureAlignment, VolProbDistribution). All should pass green or yellow.
3. **Backtest page** — click Run AI Backtest. On bundled sample data this completes in roughly 10 seconds.
4. **Verdict + Attribution tabs** — inspect the traffic-light verdict card and per-layer contribution breakdown.
5. **Diagnostics page** — review the post-backtest scans: equity curve, trade clustering, PnL autocorrelation, feature drift.

## Why these results matter

- **Magnitude.** Going from −43.6 % / Sharpe −1.52 on the EMA + RSI baseline to +67.2 % / Sharpe 2.14 is not a small delta — both return and risk-adjusted return cross zero. Max drawdown contracts from −48.0 % to −9.9 %.
- **Attribution.** The ablation study shows the Vol Gate alone contributes roughly +63 percentage points of total return (Run 4 → Run 5 in the ablation table). Without it, the cascade gives back most of its edge. The remaining architectural components add incrementally without large individual jumps.
- **Honest caveats:**
  - Backtest uses same-bar fills; live execution would differ due to next-bar-fill slippage.
  - Commissions and slippage are modelled simply (10 bps commission and 5 bps slippage per side, flat).
  - Feature set validated on the 2020–2022 regime; out-of-distribution behaviour is not characterised.
  - These are research findings, not production returns.

## Extending ApexQuant

### Adding a predictor

Subclass `BasePredictor` in `predictors/base.py` and register the class at import time:

```python
from predictors.base import BasePredictor
from predictors.registry import REGISTRY

@REGISTRY.register
class MyPredictor(BasePredictor):
    name = "my_model"
    # implement train() / predict()
```

Drop a matching `meta.json` plus `weights.{joblib,pt}` into `models/my_model/`. The registry auto-discovers it on next launch via `REGISTRY.reload()`. See `predictors/base.py` for the full interface.

### Adding a data source

Extend `data/loader.py`. The loader contract: given a ticker and a frequency, return a `pandas.DataFrame` with a `DatetimeIndex` and columns `Open, High, Low, Close, Volume` (plus any feature columns). Any source that produces that shape plugs in; see the CSV backend in the same file for a reference implementation.

### Writing a custom strategy

Subclass `BaseStrategy` in `strategies/base.py`. Strategies consume `AggregatedSignal` only, which keeps signal generation and execution logic cleanly separated. You can either drop a `.py` file into `strategies/user/` or use the in-UI Strategy Editor (`frontend/pages/3_②_Strategy_Editor.py`) for a no-code path.

## Research and methodology

Full methodology, ablation studies, the nine-method ceiling analysis, configs for all 11 ablation runs, and the thesis (available after April 27, 2026) live in the companion research repository:

🔗 https://github.com/Heeeeeeliang/Applying-Deep-Time-Series-Learning-to-Stock-Forecasting-and-Quant-Trading

## Reproducibility and quality signals

- **Test suite.** `pytest tests/` passes **157 tests** (5 skipped, 2 xfailed for documented reasons).
- **End-to-end demo test.** `pytest tests/test_demo_e2e.py` exercises the same code path as the Streamlit UI's Run AI Backtest button, using the bundled sample data plus checkpoints. It is the canonical "does this repo work on a fresh clone" signal.
- **Security audit.** `SECURITY_AUDIT_v2.md` — zero critical, high, or medium findings after the pre-publish cleanup (notebook pickle-RCE vectors, hardcoded personal paths, and Colab Drive-path leaks were all remediated).
- **Presets.** `config/presets.py` ships three ready-to-run configurations (`run1_baseline`, `run8_tranche_exit`, `run9_trailstop`). The full 11-run ablation suite, corresponding to the rows of the headline table's source ablation CSV, lives in the research repo under `configs/ablation/`.

## Project structure

<details>
<summary>Full project structure</summary>

```
apexquant/
├── config/              # Layered configuration system
│   ├── default.py       #   Master CONFIG dict + helpers
│   ├── presets.py       #   run1 / run8 / run9 named presets
│   ├── schema.py        #   Pydantic validators + guards
│   └── local_override.json  # (gitignored) per-machine overrides
├── data/                # Data layer
│   ├── bar.py           #   Universal Bar dataclass (OHLCV + indicators + AI)
│   ├── loader.py        #   CSV / yfinance / Databento loaders
│   ├── cleaner.py       #   Missing-value imputation + outlier clipping
│   ├── feature_engine.py#   pandas-ta feature computation
│   ├── vol_features.py  #   Block-aggregated 1-hour features for Layer 1
│   ├── meta_features.py #   CNN + meta-label features for Layer 2/3
│   └── context.py       #   External data stubs (VIX, sentiment)
├── predictors/          # Three-layer predictor framework
│   ├── base.py          #   Predictor / BasePredictor ABCs
│   ├── result.py        #   PredictionResult + AggregatedSignal
│   ├── registry.py      #   Auto-registration via @REGISTRY.register
│   ├── adapters/        #   VolAdapter / CnnAdapter / MetaAdapter
│   ├── calibrator.py    #   Platt / isotonic probability calibration
│   ├── factor_layer.py  #   Normalisation for aggregator input
│   ├── aggregator.py    #   LearnedAggregator (LightGBM / logistic)
│   ├── p01_volatility.py    # Layer 1: LightGBM vol predictor
│   ├── p02_turning_point.py # Layer 2: Multi-scale CNN
│   └── p03_meta_label.py    # Layer 3: Meta-label LightGBM
├── strategies/          # Decoupled strategy layer
│   ├── base.py          #   Signal, Trade, BaseStrategy ABC
│   ├── builtin/         #   AIStrategy, TechnicalStrategy
│   ├── momentum.py      #   Direction-threshold momentum
│   ├── mean_reversion.py#   Fade-extreme mean reversion
│   └── user/            #   User-saved strategies (from editor)
├── backtest/            # Event-driven backtesting
│   ├── engine.py        #   BacktestEngine + BacktestResult
│   ├── runner.py        #   run_backtest() / run_comparison()
│   ├── inference.py     #   generate_predictions() via REGISTRY
│   ├── metrics.py       #   Sharpe, Sortino, Calmar, drawdown, PF, etc.
│   ├── reporter.py      #   JSON / CSV / chart report generation
│   └── bt_*.py          #   Backtrader adapter (feeds, strategy, runner)
├── analytics/           # Verdict, attribution, health preflight
├── diagnostics/         # Post-backtest scans (equity, clustering, drift)
├── pipeline/            # Declarative pipeline executor
├── compute/             # Pluggable compute backends (local, Colab, GCP, AWS)
├── frontend/            # Streamlit multi-page UI
│   ├── app.py           #   Main entry (dark theme, sidebar, status)
│   ├── components/      #   Reusable UI widgets
│   └── pages/           #   Dashboard, Setup, Strategy Editor, Pipeline,
│                        #     Backtest Analysis, Diagnostics
├── llm/                 # LLM-backed strategy generation + sentiment (optional)
├── services/            # External integrations (Google Drive sync, etc.)
├── examples/
│   ├── sample_data/     # Bundled 6-month AAPL + SPY CSVs for the demo
│   └── sample_checkpoints/
│                        #   Minimal checkpoint bundle (vol + 2 CNNs + 2 meta-
│                        #   label classifiers) consumed by test_demo_e2e.py
├── tests/               # pytest suite (unit + test_demo_e2e.py E2E)
├── models/              # (gitignored) trained model artefacts
├── results/             # (gitignored) backtest runs + reports
├── run_all.py           # CLI entry point — full pipeline / backtest / UI launch
├── pyproject.toml       # PEP 621 package metadata
├── environment.yml      # Conda environment definition
├── requirements.txt     # pip dependency list
├── Dockerfile           # Container deployment
└── SECURITY_AUDIT_v2.md # Pre-publish audit report
```

</details>

## Requirements and deployment

- Python 3.10+.
- Dependencies: `pip install -e .` (from `pyproject.toml`, which reads `requirements.txt`), or `conda env create -f environment.yml` for conda users.
- GPU optional — auto-detected for CNN inference.
- Deployment: Streamlit local (`python run_all.py --frontend`), Docker (`docker build -t apexquant . && docker run -p 8501:8501 apexquant`), Google Cloud Run, or ngrok tunnel for a temporary public URL.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use ApexQuant in academic or commercial work:

```bibtex
@software{apexquant2026,
  author  = {Li, Heliang},
  title   = {ApexQuant: A Three-Layer Cascaded Decomposition Framework for Quantitative Trading},
  year    = {2026},
  url     = {https://github.com/Heeeeeeliang/apexquant}
}
```

Full thesis citation will be added after academic submission (post April 27, 2026).

## Acknowledgements

Built on Streamlit, Backtrader, LightGBM, PyTorch, and pandas-ta. BSc thesis research project at the University of Leeds.
