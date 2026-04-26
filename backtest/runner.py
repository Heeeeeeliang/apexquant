"""
Backtest runner for ApexQuant.

Main entry point for CLI and frontend to execute backtests.
Handles data loading, strategy instantiation, engine execution,
metrics computation, and report generation.

Usage::

    from backtest.runner import run_backtest, run_comparison
    from config.default import CONFIG

    result = run_backtest(CONFIG, strategy_name="ai")
    ai_result, tech_result = run_comparison(CONFIG)
"""

__all__ = ["run_backtest", "run_comparison", "clear_backtest_cache"]

import hashlib
import json as _json
import pickle
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from backtest.engine import BacktestResult
from backtest.metrics import compute_metrics
from backtest.reporter import BacktestReporter

# ---------------------------------------------------------------------------
# Backtest result cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("backtest/cache")


def _predictions_fingerprint(config: dict[str, Any]) -> str:
    """Return a fingerprint of prediction CSV files on disk.

    Incorporates the modification time of every ``*.csv`` in the
    predictions directory so that the cache is invalidated whenever
    predictions are regenerated.
    """
    pred_dir = Path(
        config.get("data", {}).get("predictions_dir", "results/predictions")
    )
    if not pred_dir.exists():
        return "0"
    mtimes = [
        f.stat().st_mtime
        for f in sorted(pred_dir.glob("*.csv"))
    ]
    return str(hash(tuple(mtimes)))


def _cache_key(
    config: dict[str, Any],
    strategy_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Compute a deterministic hash of config + strategy + date range + predictions."""
    blob = (
        _json.dumps(config, sort_keys=True, default=str)
        + "|" + strategy_name
        + "|" + str(start_date)
        + "|" + str(end_date)
        + "|" + _predictions_fingerprint(config)
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_cached_result(key: str) -> BacktestResult | None:
    cache_file = _CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
            # Validate the cached equity curve — stale caches from before
            # DatetimeIndex fixes may have a broken index type.
            eq = result.equity_curve
            if len(eq) > 0 and not isinstance(eq.index, pd.DatetimeIndex):
                logger.warning(
                    "Cached result has non-DatetimeIndex equity ({}), discarding",
                    type(eq.index).__name__,
                )
                cache_file.unlink(missing_ok=True)
                return None
            logger.info("Loaded cached backtest result: {}", cache_file.name)
            return result
        except Exception as exc:
            logger.warning("Failed to load cache {}: {}", cache_file, exc)
    return None


def _save_cached_result(key: str, result: BacktestResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{key}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        logger.info("Cached backtest result: {}", cache_file.name)
    except Exception as exc:
        logger.warning("Failed to save cache {}: {}", cache_file, exc)


def clear_backtest_cache() -> int:
    """Delete all cached backtest results. Returns count of files removed."""
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for f in _CACHE_DIR.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    return count

# Use Backtrader engine by default, fall back to legacy if unavailable
try:
    from backtest.bt_runner import run_engine as _bt_run_engine
    _USE_BACKTRADER = True
    logger.info("Using Backtrader engine")
except ImportError:
    _USE_BACKTRADER = False
    logger.info("Backtrader not available, using legacy engine")


def run_backtest(
    config: dict[str, Any],
    strategy_name: str = "ai",
    strategy_class: type | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    save_results: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> BacktestResult:
    """Run a backtest with the specified strategy.

    This is the main entry point called by the frontend and CLI.

    Steps:

    1. Load data for all tickers via :func:`data.loader.load_data`.
    2. Load pre-computed predictions from ``results/`` directory.
    3. Instantiate the correct strategy.
    4. Run :class:`BacktestEngine`.
    5. Compute metrics.
    6. Save results if requested.

    Args:
        config: Full CONFIG dict.
        strategy_name: Strategy to use:

            - ``"ai"`` — :class:`AIStrategy`
            - ``"technical"`` — :class:`TechnicalStrategy`
            - file path — load user strategy module dynamically

        strategy_class: Optional strategy class (overrides strategy_name).
            Must be a subclass of BaseStrategy.
        start_date: Optional start date filter (ISO format).
        end_date: Optional end date filter (ISO format).
        save_results: Whether to save reports to disk.

    Returns:
        Completed :class:`BacktestResult` with metrics populated.
    """
    logger.info(
        "Starting backtest: strategy={}, start={}, end={}",
        strategy_name,
        start_date or "earliest",
        end_date or "latest",
    )

    # --- 0. Check result cache ---
    ck = _cache_key(config, strategy_name, start_date, end_date)
    cached = _load_cached_result(ck)
    if cached is not None:
        return cached

    # --- 1. Load data ---
    # Technical strategy runs on 1-hour bars (matching the QuantConnect
    # reference).  Override freq before loading so the correct CSVs are
    # read and Backtrader receives hourly bars.
    if strategy_name == "technical":
        tech_freq = config.get("technical", {}).get(
            "freq", config.get("data", {}).get("freq_long", "1hour")
        )
        bars_by_ticker = _load_bars(config, freq_override=tech_freq)
    else:
        bars_by_ticker = _load_bars(config)

    if not bars_by_ticker:
        logger.warning("No data loaded, returning empty result")
        empty = BacktestResult(strategy_name=strategy_name)
        compute_metrics(empty)
        return empty

    # --- 2. Load predictions ---
    predictions_by_ticker = _load_predictions(config, bars_by_ticker)

    # --- 3. Instantiate strategy ---
    if strategy_class is not None:
        strategy = _safe_instantiate(strategy_class, config)
        logger.info("Using provided strategy class: {}", strategy_class.__name__)
    else:
        strategy = _instantiate_strategy(strategy_name, config)

    # Technical strategy uses its own check_exit_conditions (EMA cross
    # exit + TP/SL), so disable the tranche exit path in bt_strategy.
    if strategy_name == "technical" and "_ablation" not in config:
        config["_ablation"] = {
            "enable_tranches": False,
            "enable_signal_reversal": False,
            "enable_vol_collapse": False,
            "enable_preemption": False,
            "enable_trail_stop": False,
        }

    # --- 4. Run engine ---
    try:
        if _USE_BACKTRADER:
            logger.info("Running Backtrader engine...")
            result = _bt_run_engine(
                strategy=strategy,
                config=config,
                bars_by_ticker=bars_by_ticker,
                predictions_by_ticker=predictions_by_ticker,
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback,
            )
        else:
            from backtest._legacy import BacktestEngine
            engine = BacktestEngine(strategy, config)
            logger.info("Running legacy backtest engine...")
            result = engine.run(
                bars_by_ticker=bars_by_ticker,
                predictions_by_ticker=predictions_by_ticker,
                start_date=start_date,
                end_date=end_date,
            )
    except Exception as exc:
        logger.error(
            "Engine failed (strategy={}):\n{}",
            strategy_name, traceback.format_exc(),
        )
        raise

    # --- 5. Compute metrics ---
    compute_metrics(result)

    # --- 5b. Collect diagnostics (lightweight, <500ms) ---
    try:
        from diagnostics.engine_hooks import collect_diagnostics
        _diag = collect_diagnostics(result)
        result.config_snapshot["_diagnostics_summary"] = {
            "total_trades": _diag.trade_quality.total_trades,
            "max_win_streak": _diag.trade_quality.max_win_streak,
            "max_loss_streak": _diag.trade_quality.max_loss_streak,
            "best_hour": _diag.trade_quality.best_hour,
            "worst_hour": _diag.trade_quality.worst_hour,
            "collection_time_ms": round(_diag.collection_time_ms, 1),
        }
    except Exception as exc:
        logger.debug("Diagnostics collection skipped: {}", exc)

    # --- 6. Save results ---
    if save_results:
        run_id = _generate_run_id(strategy_name)
        output_dir = config.get("backtest", {}).get(
            "output_dir", "results/runs"
        )
        reporter = BacktestReporter(result, run_id=run_id, output_dir=output_dir)
        reporter.save_all()
        _save_latest(result)

    # --- 7. Cache result for future identical runs ---
    _save_cached_result(ck, result)

    # --- Log summary ---
    _log_summary(result)

    return result


def run_comparison(
    config: dict[str, Any],
    start_date: str | None = None,
    end_date: str | None = None,
    save_results: bool = True,
) -> tuple[BacktestResult, BacktestResult]:
    """Run both AI and Technical strategies and compare.

    Args:
        config: Full CONFIG dict.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        save_results: Whether to save reports.

    Returns:
        Tuple of ``(ai_result, tech_result)``.
    """
    logger.info("Running comparison: AI vs Technical")

    # --- Load data once (shared) ---
    bars_by_ticker = _load_bars(config)

    if not bars_by_ticker:
        logger.warning("No data loaded for comparison")
        empty_ai = BacktestResult(strategy_name="ai_strategy")
        empty_tech = BacktestResult(strategy_name="technical")
        compute_metrics(empty_ai)
        compute_metrics(empty_tech)
        return empty_ai, empty_tech

    predictions_by_ticker = _load_predictions(config, bars_by_ticker)

    # --- Run AI strategy ---
    # If AI strategy fails or produces 0 trades, we still continue
    # with the baseline comparison.
    logger.info("--- Running AI Strategy ---")
    try:
        ai_strategy = _instantiate_strategy("ai", config)
        ai_result = _run_engine(
            ai_strategy, config, bars_by_ticker,
            predictions_by_ticker, start_date, end_date,
        )
    except Exception as exc:
        logger.warning(
            "AI strategy failed — using empty result:\n{}",
            traceback.format_exc(),
        )
        ai_result = BacktestResult(strategy_name="ai_strategy")

    try:
        compute_metrics(ai_result)
    except Exception as exc:
        logger.error(
            "compute_metrics(ai) failed:\n{}", traceback.format_exc(),
        )
        ai_result.metrics = _default_metrics_fallback()

    if not ai_result.trades:
        logger.warning(
            "AI strategy produced 0 trades (predictions may not be loaded). "
            "Baseline comparison will still run."
        )

    # --- Run Technical strategy (on 1-hour bars) ---
    logger.info("--- Running Technical Strategy ---")
    try:
        tech_config = deepcopy(config)
        tech_config["_ablation"] = {
            "enable_tranches": False,
            "enable_signal_reversal": False,
            "enable_vol_collapse": False,
            "enable_preemption": False,
            "enable_trail_stop": False,
        }
        tech_freq = tech_config.get("technical", {}).get(
            "freq", tech_config.get("data", {}).get("freq_long", "1hour")
        )
        tech_bars = _load_bars(tech_config, freq_override=tech_freq)
        tech_strategy = _instantiate_strategy("technical", tech_config)
        tech_result = _run_engine(
            tech_strategy, tech_config, tech_bars,
            predictions_by_ticker, start_date, end_date,
        )
    except Exception as exc:
        logger.warning(
            "Technical strategy failed — using empty result:\n{}",
            traceback.format_exc(),
        )
        tech_result = BacktestResult(strategy_name="technical")

    try:
        compute_metrics(tech_result)
    except Exception as exc:
        logger.error(
            "compute_metrics(tech) failed:\n{}", traceback.format_exc(),
        )
        tech_result.metrics = _default_metrics_fallback()

    # --- Comparison table ---
    try:
        BacktestReporter.generate_comparison_table(ai_result, tech_result)
    except Exception as exc:
        logger.warning("generate_comparison_table failed: {}", exc)

    # --- Save comparison reports ---
    if save_results:
        run_id = _generate_run_id("comparison")
        output_dir = config.get("backtest", {}).get("output_dir", "results/runs")

        try:
            ai_reporter = BacktestReporter(
                ai_result, run_id=f"{run_id}_ai", output_dir=output_dir
            )
            ai_reporter.save_all()
        except Exception as exc:
            logger.warning("Failed to save AI report: {}", exc)

        try:
            tech_reporter = BacktestReporter(
                tech_result, run_id=f"{run_id}_technical", output_dir=output_dir
            )
            tech_reporter.save_all()
        except Exception as exc:
            logger.warning("Failed to save Technical report: {}", exc)

        try:
            ai_reporter = BacktestReporter(
                ai_result, run_id=f"{run_id}_ai", output_dir=output_dir
            )
            ai_reporter.save_comparison_charts(
                ai_result, tech_result,
                str(Path(output_dir) / run_id / "charts"),
            )
        except Exception as exc:
            logger.warning("Failed to save comparison charts: {}", exc)

        try:
            _save_latest(ai_result, baseline=tech_result)
        except Exception as exc:
            logger.warning("Failed to save latest results: {}", exc)

    return ai_result, tech_result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_metrics_fallback() -> dict[str, Any]:
    """Zeroed-out metrics dict for when compute_metrics itself crashes."""
    return {
        "total_return": 0.0, "annualized_return": 0.0,
        "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_drawdown": 0.0, "max_drawdown_duration_days": 0,
        "total_trades": 0, "win_rate": 0.0,
        "avg_trade_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "avg_bars_held": 0.0, "max_bars_held": 0,
        "exit_reasons": {}, "long_trades": 0, "short_trades": 0,
        "long_win_rate": 0.0, "short_win_rate": 0.0, "calmar_ratio": 0.0,
    }


def _run_engine(strategy, config, bars_by_ticker, predictions_by_ticker,
                 start_date=None, end_date=None, progress_callback=None):
    """Dispatch to Backtrader or legacy engine."""
    try:
        if _USE_BACKTRADER:
            return _bt_run_engine(
                strategy=strategy,
                config=config,
                bars_by_ticker=bars_by_ticker,
                predictions_by_ticker=predictions_by_ticker,
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback,
            )
        else:
            from backtest._legacy import BacktestEngine
            engine = BacktestEngine(strategy, config)
            return engine.run(
                bars_by_ticker=bars_by_ticker,
                predictions_by_ticker=predictions_by_ticker,
                start_date=start_date,
                end_date=end_date,
            )
    except Exception:
        logger.error(
            "_run_engine failed (strategy={}):\n{}",
            getattr(strategy, 'name', strategy),
            traceback.format_exc(),
        )
        raise


def _load_bars(
    config: dict[str, Any],
    freq_override: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load bar data for all configured tickers.

    Args:
        config: Full CONFIG dict.
        freq_override: If set, use this frequency instead of
            ``config["data"]["freq_short"]``.

    Returns:
        Dict mapping ticker to DataFrame.
    """
    from data.loader import load_all_tickers

    freq = freq_override or config.get("data", {}).get("freq_short", "15min")
    logger.info("Loading bar data for freq={}", freq)

    bars = load_all_tickers(freq, config)
    logger.info("Loaded {} tickers", len(bars))

    # Compute indicators if missing (CSVs may contain only OHLCV)
    for ticker, df in bars.items():
        if "ema_8" not in df.columns and "Close" in df.columns:
            bars[ticker] = _add_indicators(df)
            logger.debug("Computed indicators for {}", ticker)

    # Compute FeatureEngine features (returns, volatility, pandas-ta)
    # These populate bar.features in the backtest engine so that
    # adapters receive the same feature vectors used during training.
    try:
        from data.feature_engine import compute_features_df

        for ticker, df in bars.items():
            if "returns" not in df.columns and "Close" in df.columns:
                df = compute_features_df(df)
                # Deduplicate columns — FeatureEngine may overlap with _add_indicators
                if df.columns.duplicated().any():
                    dupes = df.columns[df.columns.duplicated()].tolist()
                    logger.warning("Dropping {} duplicate columns for {}: {}", len(dupes), ticker, dupes)
                    df = df.loc[:, ~df.columns.duplicated(keep="first")]
                bars[ticker] = df
                logger.debug("Computed FeatureEngine features for {}", ticker)
    except ImportError:
        logger.warning(
            "pandas_ta not available — FeatureEngine features will be missing. "
            "Install with: pip install pandas_ta"
        )

    return bars


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 20 technical indicators to a DataFrame with OHLCV columns.

    Computes EMA-8/21/50, RSI-14, MACD (line/signal/hist), ATR-14,
    Bollinger Bands (upper/mid/lower), volume ratio, VWAP, OBV,
    ADX-14, Stochastic K/D, Williams %R, CCI-20, and MFI-14.
    Only uses pandas — no external TA library required.
    """
    c = df["Close"]
    h = df["High"]
    l = df["Low"]  # noqa: E741
    v = df["Volume"]

    # EMAs
    df["ema_8"] = c.ewm(span=8, adjust=False).mean()
    df["ema_21"] = c.ewm(span=21, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()

    # RSI-14
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ATR-14
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_mid"] = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # Volume ratio (current / 20-bar SMA)
    vol_sma = v.rolling(20).mean().replace(0, 1e-10)
    df["volume_ratio"] = v / vol_sma

    # VWAP (cumulative for each trading day; resets daily if DatetimeIndex)
    typical = (h + l + c) / 3.0
    cum_tp_vol = (typical * v).cumsum()
    cum_vol = v.cumsum().replace(0, 1e-10)
    df["vwap"] = cum_tp_vol / cum_vol

    # OBV (On-Balance Volume)
    sign = c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["obv"] = (sign * v).cumsum()

    # ADX-14
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    # Zero out when opposite DM is larger
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0
    atr = df["atr_14"]  # already computed above
    plus_di = 100.0 * (plus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-10))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    df["adx_14"] = dx.ewm(span=14, adjust=False).mean()

    # Stochastic Oscillator (14, 3)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100.0 * (c - low14) / (high14 - low14).replace(0, 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R (14)
    df["willr_14"] = -100.0 * (high14 - c) / (high14 - low14).replace(0, 1e-10)

    # CCI-20 (Commodity Channel Index)
    tp = (h + l + c) / 3.0
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, 1e-10))

    # MFI-14 (Money Flow Index)
    mf_raw = typical * v
    pos_mf = mf_raw.where(typical > typical.shift(), 0.0).rolling(14).sum()
    neg_mf = mf_raw.where(typical < typical.shift(), 0.0).rolling(14).sum()
    mf_ratio = pos_mf / neg_mf.replace(0, 1e-10)
    df["mfi_14"] = 100.0 - (100.0 / (1.0 + mf_ratio))

    return df


def _load_predictions(
    config: dict[str, Any],
    bars_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Load pre-computed predictions from the results directory.

    Looks for CSV files matching
    ``{predictions_dir}/{ticker}_predictions.csv``.

    Args:
        config: Full CONFIG dict.
        bars_by_ticker: Loaded bar DataFrames (for index alignment).

    Returns:
        Dict mapping ticker to predictions DataFrame.
    """
    predictions_dir = Path(
        config.get("data", {}).get("predictions_dir", "results/predictions")
    )
    predictions: dict[str, pd.DataFrame] = {}

    if not predictions_dir.exists():
        logger.warning(
            "Predictions directory not found: {}. "
            "Use the 'Generate Predictions' button to create prediction CSVs.",
            predictions_dir,
        )
        return predictions

    for ticker in bars_by_ticker:
        pred_path = predictions_dir / f"{ticker}_predictions.csv"
        if pred_path.exists():
            try:
                pred_df = pd.read_csv(pred_path)
                # Parse timestamp
                for col_name in ["ts_event", "timestamp", "datetime", "date"]:
                    if col_name in pred_df.columns:
                        pred_df[col_name] = pd.to_datetime(pred_df[col_name], utc=True)
                        pred_df = pred_df.set_index(col_name)
                        break
                predictions[ticker] = pred_df
                # Log which prediction columns (output_labels) were found
                non_ts_cols = [c for c in pred_df.columns if c not in {"ts_event", "timestamp", "datetime", "date"}]
                logger.info(
                    "Loaded predictions for {}: {} rows, columns={}",
                    ticker,
                    len(pred_df),
                    non_ts_cols,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load predictions for {}: {}",
                    ticker,
                    exc,
                )
        else:
            logger.warning(
                "No prediction CSV for ticker '{}' (expected {})",
                ticker,
                pred_path,
            )

    logger.info(
        "Loaded predictions for {}/{} tickers from {}",
        len(predictions),
        len(bars_by_ticker),
        predictions_dir,
    )
    return predictions


def _safe_instantiate(cls: type, config: dict[str, Any]) -> Any:
    """Instantiate a strategy class, tolerating __init__ signatures.

    Tries ``cls(config)`` first.  If that fails with a TypeError (e.g. the
    user wrote ``def __init__(self):`` without calling ``super().__init__``),
    falls back to ``cls()`` and patches in all the attributes that
    ``BaseStrategy.__init__`` would have set.
    """
    try:
        return cls(config)
    except TypeError:
        instance = cls()
        # Patch every attribute that BaseStrategy.__init__ normally sets,
        # so that lifecycle methods, helpers, and the engine all work.
        _defaults: dict[str, Any] = {
            "config": config,
            "positions": {},
            "signal_buffer": {},
            "trade_history": [],
            "_initialized": False,
        }
        for attr, default in _defaults.items():
            if not hasattr(instance, attr):
                setattr(instance, attr, default)
        return instance


def _instantiate_strategy(
    strategy_name: str, config: dict[str, Any]
) -> Any:
    """Instantiate a strategy by name or file path.

    Args:
        strategy_name: ``"ai"``, ``"technical"``, or a file path
            to a user strategy module.
        config: Full CONFIG dict.

    Returns:
        A :class:`BaseStrategy` instance.

    Raises:
        ValueError: If the strategy name is unknown and no file found.
    """
    if strategy_name == "ai":
        from strategies.builtin.ai_strategy import AIStrategy
        return AIStrategy.from_config(config)

    if strategy_name == "technical":
        from strategies.builtin.technical import TechnicalStrategy
        return TechnicalStrategy(config)

    # Try loading from file path
    strategy_path = Path(strategy_name)
    if strategy_path.exists() and strategy_path.suffix == ".py":
        return _load_user_strategy(strategy_path, config)

    raise ValueError(
        f"Unknown strategy '{strategy_name}'. "
        f"Valid: 'ai', 'technical', or a path to a .py file"
    )


def _load_user_strategy(path: Path, config: dict[str, Any]) -> Any:
    """Load a user-defined strategy from a Python file.

    The file must define a class that inherits from
    :class:`BaseStrategy` and accepts ``config`` as its only
    constructor argument.

    Args:
        path: Path to the .py file.
        config: Full CONFIG dict.

    Returns:
        Instantiated strategy.

    Raises:
        ValueError: If no BaseStrategy subclass is found in the file.
    """
    import importlib.util

    from strategies.base import BaseStrategy

    spec = importlib.util.spec_from_file_location("user_strategy", str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first BaseStrategy subclass in the module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseStrategy)
            and attr is not BaseStrategy
        ):
            logger.info("Loaded user strategy: {} from {}", attr_name, path)
            return _safe_instantiate(attr, config)

    raise ValueError(
        f"No BaseStrategy subclass found in {path}. "
        f"Define a class that inherits from strategies.base.BaseStrategy."
    )


def _generate_run_id(strategy_name: str) -> str:
    """Generate a unique run ID.

    Args:
        strategy_name: Strategy name for the suffix.

    Returns:
        Run ID like ``"20240115_143022_ai"``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{strategy_name}"


def _save_latest(result: BacktestResult, baseline: BacktestResult | None = None) -> None:
    """Save results as latest_* files for dashboard consumption.

    Args:
        result: Primary backtest result.
        baseline: Optional baseline result.
    """
    import json

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_data = {
        "strategy": result.strategy_name,
        "total_trades": len(result.trades),
        "metrics": {k: v for k, v in result.metrics.items() if not isinstance(v, dict)},
    }
    with open(results_dir / "latest_backtest.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, default=str)

    # Equity CSV
    equity = result.equity_curve
    if len(equity) > 0:
        eq_df = pd.DataFrame({
            "timestamp": equity.index,
            "portfolio_value": equity.values,
        })
        eq_df.to_csv(results_dir / "latest_equity.csv", index=False)

    # Trades CSV
    if result.trades:
        trade_rows = []
        for t in result.trades:
            trade_rows.append({
                "trade_id": t.trade_id,
                "ticker": t.ticker,
                "timestamp": t.timestamp,
                "signal": t.signal.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_timestamp": t.exit_timestamp,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })
        pd.DataFrame(trade_rows).to_csv(
            results_dir / "latest_trades.csv", index=False
        )

    # Baseline
    if baseline is not None:
        baseline_data = {
            "strategy": baseline.strategy_name,
            "total_trades": len(baseline.trades),
            "metrics": {k: v for k, v in baseline.metrics.items() if not isinstance(v, dict)},
        }
        with open(results_dir / "latest_baseline.json", "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2, default=str)

    logger.info("Saved latest results to {}", results_dir)


def _log_summary(result: BacktestResult) -> None:
    """Log a summary of the backtest result.

    Args:
        result: Completed backtest result with metrics.
    """
    m = result.metrics
    if not m:
        return

    logger.info("=" * 50)
    logger.info("Backtest Summary: {}", result.strategy_name)
    logger.info("-" * 50)
    logger.info("  Total Return:    {:.2%}", m.get("total_return", 0))
    logger.info("  Sharpe Ratio:    {:.2f}", m.get("sharpe_ratio", 0))
    logger.info("  Sortino Ratio:   {:.2f}", m.get("sortino_ratio", 0))
    logger.info("  Max Drawdown:    {:.2%}", m.get("max_drawdown", 0))
    logger.info("  Win Rate:        {:.1%}", m.get("win_rate", 0))
    logger.info("  Profit Factor:   {:.2f}", m.get("profit_factor", 0))
    logger.info("  Total Trades:    {}", m.get("total_trades", 0))
    logger.info("  Avg Bars Held:   {:.1f}", m.get("avg_bars_held", 0))
    logger.info("=" * 50)
