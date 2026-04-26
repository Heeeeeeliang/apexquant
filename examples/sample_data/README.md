# Bundled sample data — demo only

## Contents

| File | Ticker | Frequency | Rows | Date range (UTC) | Size |
|---|---|---|---:|---|---:|
| `AAPL_15min.csv` | AAPL | 15-minute bars | 7,743 | 2022-01-03 09:00 → 2022-06-30 23:45 | 428 KB |
| `SPY_15min.csv`  | SPY  | 15-minute bars | 7,724 | 2022-01-03 09:00 → 2022-06-30 23:45 | 427 KB |

Columns: `timestamp,open,high,low,close,volume`. The loader auto-normalises
lowercase OHLCV headers to the Title-case variants used downstream.

## Source

Sliced from the bar-aggregated OHLCV CSVs in the companion research data
repository (`apexquant_data_release`), which in turn derive from Databento
historical NASDAQ ITCH (2020-01-07 → 2022-09-30). The slice window
(2022-01-01 → 2022-07-01) was chosen to cover the first-half-of-2022
downtrend (multiple distinct volatility regimes, exercises the Vol Gate
under realistic conditions) while keeping the bundle under the 30 MB
budget for a public code repository.

## Intended use

**Demo only.** These two files exist so the bundled
`tests/test_demo_e2e.py` and the Streamlit UI quickstart can run
end-to-end on a fresh clone without any external downloads. They are
**not** a training dataset and **not** the data used to produce the
thesis's headline numbers — the full dataset (8 tickers × 4 resolutions,
2.75 years) lives in the research repo.

## Getting the full dataset

For the complete Databento-derived bars + engineered features + splits:
see [`apexquant_data_release`](https://github.com/<USER>/apexquant-data-release)
(placeholder URL). That repo includes the dataset card with licensing
and redistribution terms.
