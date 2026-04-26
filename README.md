# ApexQuant

Quantitative trading research platform with pluggable AI signal architecture.
Built for COMP3931 Individual Project — University of Leeds 2024/25.

## Architecture

Three-layer AI signal cascade:

| Layer | Task | Model | Key Metric |
|-------|------|-------|------------|
| 1 | Volatility Prediction | LightGBM | DA = 83.6% |
| 2 | Turning-Point Detection | Multi-Scale CNN | AUC = 0.826 |
| 3 | Trade Filtering | Meta-Label LightGBM | WR = 64.5% |

Signals are combined by a **LearnedAggregator** (weights optimised on the
validation set only) then consumed by decoupled strategies via the
`AggregatedSignal` interface.

```
Volatility LGB ──┐
CNN Turning Pt ───┼──▶ LearnedAggregator ──▶ Strategy ──▶ Backtest Engine
Meta-Label LGB ──┘
```

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py --frontend
```

The Streamlit UI opens at `http://localhost:8501`.

## CLI Usage

```bash
# Full pipeline: train all layers + backtest
python run_all.py --all

# Train specific layers
python run_all.py --steps 01 02 03

# Backtest only (requires pre-trained models)
python run_all.py --backtest-only --strategy ai

# AI vs Technical comparison
python run_all.py --backtest-only --compare

# Custom strategy
python run_all.py --backtest-only --strategy strategies/user/my_strategy.py

# Date range
python run_all.py --backtest-only --compare --start 2022-01-01 --end 2022-12-31

# Launch frontend
python run_all.py --frontend
```

## Deployment

### Docker

```bash
docker build -t apexquant .
docker run -p 8501:8501 apexquant
```

### Google Cloud Run

```bash
gcloud run deploy apexquant \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi
```

### Local demo with public URL

```bash
streamlit run frontend/app.py &
ngrok http 8501
```

## Project Structure

```
apexquant/
├── config/              # Layered configuration system
│   ├── default.py       #   Master CONFIG dict + helpers
│   └── local_override.json  # (gitignored) per-machine overrides
├── data/                # Data layer
│   ├── bar.py           #   Universal Bar dataclass (OHLCV + 20 indicators + AI)
│   ├── loader.py        #   CSV / yfinance / Databento loaders
│   ├── cleaner.py       #   Missing-value imputation + outlier clipping
│   └── context.py       #   External data stubs (VIX, sentiment)
├── predictors/          # Three-layer predictor framework
│   ├── base.py          #   Predictor / BasePredictor ABCs
│   ├── result.py        #   PredictionResult + AggregatedSignal
│   ├── registry.py      #   Auto-registration via @REGISTRY.register
│   ├── calibrator.py    #   Platt / isotonic probability calibration
│   ├── factor_layer.py  #   Normalisation for aggregator input
│   ├── aggregator.py    #   LearnedAggregator (LightGBM / logistic)
│   ├── p01_volatility.py    # Layer 1: LightGBM vol classifier
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
│   ├── metrics.py       #   Sharpe, Sortino, Calmar, drawdown, etc.
│   ├── reporter.py      #   JSON / CSV / chart report generation
│   └── runner.py        #   run_backtest() / run_comparison()
├── compute/             # Pluggable compute backends
│   ├── base.py          #   ComputeBackend ABC
│   ├── local_backend.py #   Subprocess + GPU detection
│   ├── colab_backend.py #   Google Drive async file sync
│   ├── gcloud_backend.py#   Vertex AI stub
│   └── aws_backend.py   #   SageMaker stub
├── frontend/            # Streamlit multi-page UI
│   ├── app.py           #   Main entry (dark theme, sidebar, status)
│   └── pages/
│       ├── 1_dashboard.py   # Metric cards, model status, quick actions
│       ├── 2_config.py      # 5-tab configuration editor
│       ├── 3_strategy.py    # Code editor + live backtest
│       ├── 4_backtest.py    # Run browser + analysis tabs
│       └── 5_diagnostics.py # Model introspection (DA, AUC, WR, aggregator)
├── models/              # Trained model artifacts
├── results/             # Output: runs, predictions, charts
├── run_all.py           # CLI entry point
├── Dockerfile           # Container deployment
└── requirements.txt     # Python dependencies
```

## Key Design Decisions

- **Chronological splits only** — no random shuffling. Train → Val → Test
  are strictly time-ordered. Validation set used for threshold / weight
  tuning; test set touched exactly once.
- **Bar as universal carrier** — `data.bar.Bar` holds OHLCV, 20 technical
  indicators, and AI predictions in a single dataclass.
- **Strategy decoupling** — strategies consume `AggregatedSignal` only;
  they never import from `predictors/`.
- **Next-bar-open execution** — signals are queued and executed at the
  next bar's open price with slippage, preventing look-ahead bias.

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependency list
- GPU optional (auto-detected for CNN training)

## COMP3931 — University of Leeds 2024/25
