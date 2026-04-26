"""
Batch inference — generate prediction CSVs for all tickers.

Bridges the gap between trained model weights (in ``models/``) and the
prediction CSV files that the backtest engine expects at
``results/predictions/{ticker}_predictions.csv``.

LightGBM adapters (VolAdapter, MetaAdapter) use **vectorised batch
inference** — the full feature matrix is passed to ``model.predict()``
/ ``model.predict_proba()`` in a single call.

Each model directory contains a ``feature_names.json`` that documents
the exact training-time feature order.  This module loads those names
and aligns the local feature matrix accordingly — matching columns are
placed in the correct position, missing columns are filled with NaN
(not zeros), and extra local columns are dropped.

Usage::

    from backtest.inference import generate_predictions
    summary = generate_predictions(config)
"""

__all__ = ["generate_predictions", "load_feature_names_from_dir"]

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from predictors.registry import REGISTRY


_OHLCV = {"Open", "High", "Low", "Close", "Volume"}


# ---------------------------------------------------------------------------
# Adapter type helpers
# ---------------------------------------------------------------------------

def _is_lgb_adapter(pred: Any) -> bool:
    """Return True if *pred* is a LightGBM-based adapter (Vol or Meta)."""
    from predictors.adapters.vol_adapter import VolAdapter
    from predictors.adapters.meta_adapter import MetaAdapter
    return isinstance(pred, (VolAdapter, MetaAdapter))


def _is_classifier(pred: Any) -> bool:
    """Return True if the adapter wraps a LightGBM *classifier*."""
    from predictors.adapters.meta_adapter import MetaAdapter
    return isinstance(pred, MetaAdapter)


def _needs_vol_pipeline(training_names: list[str] | None) -> bool:
    """Return True if the training feature list contains block-aggregated
    volatility features (``b0_rv`` etc.), meaning the vol feature pipeline
    from ``data.vol_features`` is needed instead of raw bar features."""
    if not training_names:
        return False
    return "b0_rv" in training_names


def _needs_meta_pipeline(training_names: list[str] | None) -> bool:
    """Return True if the training feature list contains flattened CNN-window
    features (``s0_open`` etc.), meaning the meta feature pipeline
    from ``data.meta_features`` is needed."""
    if not training_names:
        return False
    return "s0_open" in training_names


def load_feature_names_from_dir(model_dir: Path) -> list[str] | None:
    """Load training-time feature names from a model directory.

    Reads ``feature_names.json`` and returns the ``features`` list.
    Returns None if the file is missing or unreadable.

    Args:
        model_dir: Path to the model directory containing feature_names.json.

    Returns:
        List of feature name strings, or None.
    """
    path = Path(model_dir) / "feature_names.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        names = data.get("features", [])
        return names
    except Exception as exc:
        logger.warning("Failed to load feature_names.json from {}: {}", path, exc)
        return None


def _load_feature_names(adapter: Any) -> list[str] | None:
    """Load training-time feature names for an adapter (convenience wrapper)."""
    model_dir = getattr(adapter, "_model_dir", None)
    if model_dir is None:
        return None
    names = load_feature_names_from_dir(Path(model_dir))
    if names is not None:
        logger.info(
            "Loaded {} training feature names for '{}' from {}",
            len(names), adapter.name, model_dir,
        )
    return names


def _align_to_training_order(
    df: pd.DataFrame,
    feature_cols: list[str],
    training_names: list[str],
    adapter_name: str,
) -> np.ndarray:
    """Build a feature matrix aligned to the model's training-time column order.

    - Columns present locally are placed in the correct position.
    - Missing columns are filled with NaN (so LightGBM uses its native
      NaN-handling rather than treating zeros as real values).
    - Extra local columns not in the training list are dropped.

    Returns an (n_rows, n_training_features) float64 array.
    """
    n_rows = len(df)
    n_train = len(training_names)
    X = np.full((n_rows, n_train), np.nan, dtype=np.float64)

    # Build a lookup from local column name to column index in df
    local_cols = {c: i for i, c in enumerate(feature_cols)}
    local_vals = df[feature_cols].to_numpy(dtype=np.float64)

    matched = 0
    for ti, tname in enumerate(training_names):
        if tname in local_cols:
            X[:, ti] = local_vals[:, local_cols[tname]]
            matched += 1

    missing = n_train - matched
    if missing > 0:
        logger.warning(
            "Feature alignment for '{}': {}/{} matched, {} missing (filled with NaN)",
            adapter_name, matched, n_train, missing,
        )
    else:
        logger.info(
            "Feature alignment for '{}': all {}/{} features matched",
            adapter_name, matched, n_train,
        )
    return X


def _align_matrix(
    X: np.ndarray, expected: int, name: str
) -> np.ndarray:
    """Pad or truncate columns of *X* so that ``X.shape[1] == expected``.

    Fallback when no feature_names.json is available.  Uses NaN padding
    instead of zeros.
    """
    n_cols = X.shape[1]
    if n_cols == expected:
        return X
    logger.warning(
        "Batch {}: feature matrix has {} cols, model expects {}; {}",
        name, n_cols, expected,
        "padding with NaN" if n_cols < expected else "truncating",
    )
    if n_cols < expected:
        pad = np.full((X.shape[0], expected - n_cols), np.nan, dtype=X.dtype)
        return np.hstack([X, pad])
    return X[:, :expected]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_predictions(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run all registered LightGBM models on all bars for all tickers.

    Uses **batch inference**: for each LightGBM adapter the full feature
    matrix is passed to ``model.predict()`` / ``model.predict_proba()``
    in a single call.  CNN adapters are skipped entirely.

    Iterates all registered predictors directly (ignores the
    ``config["predictors"]["enabled"]`` list, which may be empty).

    Args:
        config: Full CONFIG dict.
        output_dir: Where to write CSVs.  Defaults to
            ``config["data"]["predictions_dir"]`` or
            ``results/predictions``.
        progress_callback: Optional ``(current, total, ticker)`` callable.

    Returns:
        Summary dict with keys ``tickers_processed``, ``total_rows``,
        ``predictors_used``, and per-ticker row counts.
    """
    if output_dir is None:
        output_dir = Path(
            config.get("data", {}).get("predictions_dir", "results/predictions")
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Reload registry and diagnose what we have
    # ------------------------------------------------------------------
    # Honour an optional models_dir override from config so bundled demos
    # and tests can point at a subset bundle (e.g. examples/sample_checkpoints/)
    # instead of the repo-wide ./models/ tree.
    _models_dir = (
        config.get("data", {}).get("models_dir")
        or config.get("models_dir")
        or config.get("drive", {}).get("models_dir")
        or "models"
    )
    REGISTRY.reload(models_dir=str(_models_dir))
    registered = REGISTRY.list_all()
    logger.info(
        "[diag] REGISTRY contains {} predictor(s): {}", len(registered), registered
    )
    if not registered:
        logger.warning("No predictors registered — nothing to do")
        return {"tickers_processed": 0, "total_rows": 0, "predictors_used": []}

    # ------------------------------------------------------------------
    # Step 2: Iterate ALL registered predictors directly (bypass
    #         config["predictors"]["enabled"] which is [] by default).
    # ------------------------------------------------------------------
    all_predictors = [REGISTRY.get(name) for name in registered]
    for pred in all_predictors:
        logger.info(
            "[diag] Predictor '{}': type={}, is_lgb={}, is_classifier={}, "
            "is_ready={}, output_label='{}'",
            pred.name,
            type(pred).__name__,
            _is_lgb_adapter(pred),
            _is_classifier(pred),
            pred.is_ready(),
            pred.output_label,
        )

    lgb_adapters = [p for p in all_predictors if _is_lgb_adapter(p) and p.is_ready()]
    skipped = [p.name for p in all_predictors if not _is_lgb_adapter(p)]
    if skipped:
        logger.info("Skipping non-LightGBM adapters: {}", skipped)

    logger.info(
        "[diag] LightGBM adapters ready for batch inference: {}",
        [a.name for a in lgb_adapters],
    )

    if not lgb_adapters:
        logger.warning("No LightGBM adapters ready — nothing to do")
        return {"tickers_processed": 0, "total_rows": 0, "predictors_used": registered}

    # ------------------------------------------------------------------
    # Step 3: Load models (call adapter.load() if _model is None)
    # ------------------------------------------------------------------
    for adapter in lgb_adapters:
        if adapter._model is None:
            try:
                adapter.load()
                logger.info(
                    "[diag] Loaded '{}': _model type={}, n_features={}",
                    adapter.name,
                    type(adapter._model).__name__,
                    getattr(adapter._model, "n_features_in_", None)
                    or getattr(adapter._model, "n_features_", None),
                )
            except Exception as exc:
                logger.warning(
                    "[diag] Failed to load '{}' ({}): {}",
                    adapter.name, type(exc).__name__, exc,
                )
        else:
            logger.info(
                "[diag] '{}' already loaded: _model type={}",
                adapter.name, type(adapter._model).__name__,
            )

    lgb_adapters = [a for a in lgb_adapters if a._model is not None]
    if not lgb_adapters:
        logger.warning("All LightGBM adapters failed to load")
        return {"tickers_processed": 0, "total_rows": 0, "predictors_used": registered}

    # ------------------------------------------------------------------
    # Step 3b: Pre-load feature_names.json for each adapter
    # ------------------------------------------------------------------
    adapter_feature_names: dict[str, list[str] | None] = {}
    for adapter in lgb_adapters:
        adapter_feature_names[adapter.name] = _load_feature_names(adapter)

    logger.info(
        "Batch inference with {} adapter(s): {}",
        len(lgb_adapters), [a.name for a in lgb_adapters],
    )

    # ------------------------------------------------------------------
    # Step 4: Load ticker data
    # ------------------------------------------------------------------
    from data.loader import load_all_tickers

    freq = config.get("data", {}).get("freq_short", "15min")
    bars_by_ticker = load_all_tickers(freq, config)

    if not bars_by_ticker:
        logger.warning("No ticker data loaded — cannot generate predictions")
        return {"tickers_processed": 0, "total_rows": 0, "predictors_used": registered}

    logger.info("[diag] Loaded {} ticker(s): {}", len(bars_by_ticker), list(bars_by_ticker))

    # ------------------------------------------------------------------
    # Step 5: Compute FeatureEngine features
    # ------------------------------------------------------------------
    try:
        from data.feature_engine import compute_features_df

        for ticker, df in bars_by_ticker.items():
            if "returns" not in df.columns and "Close" in df.columns:
                bars_by_ticker[ticker] = compute_features_df(df)
                logger.info(
                    "[diag] FeatureEngine for {}: {} cols after compute",
                    ticker, len(bars_by_ticker[ticker].columns),
                )
    except ImportError:
        logger.warning(
            "pandas_ta not available — FeatureEngine features will be missing"
        )

    # Identify feature columns (everything except OHLCV) per ticker
    feature_cols_by_ticker: dict[str, list[str]] = {}
    for ticker, df in bars_by_ticker.items():
        feature_cols_by_ticker[ticker] = [
            c for c in df.columns
            if c not in _OHLCV
            and c.lower() not in {"open", "high", "low", "close", "volume"}
        ]
        logger.info(
            "[diag] {} feature columns for {}: first 10 = {}",
            len(feature_cols_by_ticker[ticker]),
            ticker,
            feature_cols_by_ticker[ticker][:10],
        )

    # ------------------------------------------------------------------
    # Step 5b: Detect vol-pipeline adapters and build vol features
    # ------------------------------------------------------------------
    _vol_adapters = [
        a for a in lgb_adapters
        if not _is_classifier(a)
        and _needs_vol_pipeline(adapter_feature_names.get(a.name))
    ]
    vol_features_by_ticker: dict[str, pd.DataFrame] = {}
    if _vol_adapters:
        from data.vol_features import build_vol_features

        logger.info(
            "Vol-pipeline adapters detected: {} — building block features",
            [a.name for a in _vol_adapters],
        )
        for ticker in bars_by_ticker:
            try:
                vol_features_by_ticker[ticker] = build_vol_features(
                    ticker, config,
                )
                logger.info(
                    "Vol features for {}: {} blocks × {} cols",
                    ticker,
                    len(vol_features_by_ticker[ticker]),
                    vol_features_by_ticker[ticker].shape[1],
                )
            except Exception as exc:
                logger.warning(
                    "Vol feature pipeline failed for {} ({}): {}",
                    ticker, type(exc).__name__, exc,
                )

    # ------------------------------------------------------------------
    # Step 5c: Detect meta-pipeline adapters and build meta features
    #          (separate per CNN task so cnn_prob is correct for each)
    # ------------------------------------------------------------------
    _meta_adapters = [
        a for a in lgb_adapters
        if _is_classifier(a)
        and _needs_meta_pipeline(adapter_feature_names.get(a.name))
    ]
    # meta_features_by_task_ticker[task][ticker] = DataFrame
    meta_features_by_task_ticker: dict[str, dict[str, pd.DataFrame]] = {}
    if _meta_adapters:
        from data.meta_features import build_meta_features as _build_meta

        # Determine which CNN tasks are needed
        meta_tasks: set[str] = set()
        for a in _meta_adapters:
            if "bottom" in a.name:
                meta_tasks.add("bottom")
            elif "top" in a.name:
                meta_tasks.add("top")
            else:
                meta_tasks.add("bottom")  # default fallback

        logger.info(
            "Meta-pipeline adapters detected: {} — tasks: {}",
            [a.name for a in _meta_adapters], sorted(meta_tasks),
        )

        for task in sorted(meta_tasks):
            meta_features_by_task_ticker[task] = {}
            for ticker in bars_by_ticker:
                try:
                    mf = _build_meta(ticker, config, cnn_task=task)
                    meta_features_by_task_ticker[task][ticker] = mf
                    logger.info(
                        "Meta features ({}) for {}: {} rows × {} cols",
                        task, ticker, len(mf), mf.shape[1],
                    )
                except Exception as exc:
                    logger.warning(
                        "Meta feature pipeline ({}) failed for {} ({}): {}",
                        task, ticker, type(exc).__name__, exc,
                    )

    # ------------------------------------------------------------------
    # Step 6: Batch inference per ticker
    # ------------------------------------------------------------------
    total_tickers = len(bars_by_ticker)
    summary: dict[str, Any] = {
        "tickers_processed": 0,
        "total_rows": 0,
        "predictors_used": [a.name for a in lgb_adapters],
        "per_ticker": {},
    }

    for t_idx, (ticker, df) in enumerate(bars_by_ticker.items()):
        feature_cols = feature_cols_by_ticker[ticker]
        n_bars = len(df)
        timestamps = df.index
        logger.info(
            "Generating predictions for {} ({} bars, {} features)",
            ticker, n_bars, len(feature_cols),
        )

        if n_bars == 0:
            logger.warning("No bars for {} — skipping", ticker)
            summary["per_ticker"][ticker] = 0
            continue

        pred_columns: dict[str, np.ndarray] = {}

        # --- Pass 1: VolAdapter (LGBMRegressor) ---
        vol_df = vol_features_by_ticker.get(ticker)

        for adapter in lgb_adapters:
            if _is_classifier(adapter):
                continue  # meta-label adapters run in pass 2

            training_names = adapter_feature_names.get(adapter.name)
            expected = getattr(adapter._model, "n_features_in_", None) or getattr(
                adapter._model, "n_features_", None
            )

            # Use vol-pipeline features when available for this adapter
            use_vol = (
                vol_df is not None
                and len(vol_df) > 0
                and _needs_vol_pipeline(training_names)
            )

            if use_vol:
                X = vol_df.to_numpy(dtype=np.float64)
                vol_timestamps = vol_df.index
                logger.info(
                    "[diag] Pass 1 — '{}': using vol pipeline ({} blocks × {} features)",
                    adapter.name, X.shape[0], X.shape[1],
                )
            elif training_names and feature_cols:
                X = _align_to_training_order(
                    df, feature_cols, training_names, adapter.name
                )
            elif feature_cols:
                X_base = df[feature_cols].to_numpy(dtype=np.float64)
                X = _align_matrix(X_base, expected, adapter.name) if expected else X_base
            else:
                logger.warning("No features for {} / {} — skipping", ticker, adapter.name)
                continue

            try:
                raw = adapter._model.predict(X)  # shape (n_rows,)

                if use_vol:
                    # Vol models output log(RV), not logits.  Convert to a
                    # probability in [0, 1] by computing the percentile rank
                    # of the predicted RV within its own distribution.  High
                    # predicted RV → high vol_prob, low → low vol_prob.
                    predicted_rv = np.exp(raw)
                    median_rv = float(np.median(predicted_rv))
                    std_rv = float(np.std(predicted_rv)) or 1e-10
                    # Logistic mapping centred on the median:
                    # prob = sigmoid((predicted_rv - median) / std)
                    z = (predicted_rv - median_rv) / std_rv
                    probs = 1.0 / (1.0 + np.exp(-z))

                    logger.info(
                        "  {} vol transform: median_rv={:.6f}, std_rv={:.6f}, "
                        "prob range [{:.4f}, {:.4f}]",
                        adapter.name, median_rv, std_rv,
                        float(probs.min()), float(probs.max()),
                    )

                    # Forward-fill block predictions to bar timestamps
                    block_series = pd.Series(
                        probs, index=vol_timestamps, name=adapter.output_label,
                    )
                    bar_probs = block_series.reindex(timestamps, method="ffill")
                    bar_probs = bar_probs.fillna(method="bfill").fillna(0.5)
                    pred_columns[adapter.output_label] = bar_probs.values
                else:
                    # Non-vol regressors: keep sigmoid (legacy behaviour)
                    probs = 1.0 / (1.0 + np.exp(-raw))
                    pred_columns[adapter.output_label] = probs

                logger.info(
                    "  {} → {} predictions, prob range [{:.4f}, {:.4f}]",
                    adapter.name,
                    len(pred_columns[adapter.output_label]),
                    float(pred_columns[adapter.output_label].min()),
                    float(pred_columns[adapter.output_label].max()),
                )
            except Exception as exc:
                logger.error(
                    "[diag] Batch predict FAILED for '{}' ({}): {}",
                    adapter.name, type(exc).__name__, exc,
                )

        # --- Pass 2: MetaAdapter (LGBMClassifier) ---
        for adapter in lgb_adapters:
            if not _is_classifier(adapter):
                continue

            training_names = adapter_feature_names.get(adapter.name)
            expected = getattr(adapter._model, "n_features_in_", None) or getattr(
                adapter._model, "n_features_", None
            )

            # Determine which CNN task this adapter needs
            adapter_task = "bottom" if "bottom" in adapter.name else "top"
            task_meta = meta_features_by_task_ticker.get(adapter_task, {})
            meta_df = task_meta.get(ticker)

            # Use meta-pipeline features when available
            use_meta = (
                meta_df is not None
                and len(meta_df) > 0
                and _needs_meta_pipeline(training_names)
            )

            if use_meta:
                X_meta = meta_df.to_numpy(dtype=np.float64)
                meta_timestamps = meta_df.index
                logger.info(
                    "[diag] Pass 2 — '{}': using meta pipeline [{}] ({} rows × {} features)",
                    adapter.name, adapter_task, X_meta.shape[0], X_meta.shape[1],
                )
            elif training_names and feature_cols:
                X_meta = _align_to_training_order(
                    df, feature_cols, training_names, adapter.name
                )
                meta_timestamps = None
            elif feature_cols:
                X_base = df[feature_cols].to_numpy(dtype=np.float64)
                X_meta = _align_matrix(X_base, expected, adapter.name) if expected else X_base
                meta_timestamps = None
            else:
                logger.warning("No features for {} / {} — skipping", ticker, adapter.name)
                continue

            logger.info(
                "[diag] Pass 2 — '{}' ({}): model expects {} features, "
                "X_meta has {}",
                adapter.name,
                type(adapter._model).__name__,
                expected,
                X_meta.shape[1],
            )

            try:
                probs = adapter._model.predict_proba(X_meta)[:, 1]

                if use_meta:
                    # Forward-fill meta predictions (15-min) to bar timestamps
                    meta_series = pd.Series(
                        probs, index=meta_timestamps, name=adapter.output_label,
                    )
                    # Strip tz from timestamps for alignment if needed
                    ts_notz = timestamps
                    if hasattr(timestamps, "tz") and timestamps.tz is not None:
                        if meta_series.index.tz is None:
                            ts_notz = timestamps.tz_localize(None)
                    elif meta_series.index.tz is not None:
                        meta_series.index = meta_series.index.tz_localize(None)

                    bar_probs = meta_series.reindex(ts_notz, method="ffill")
                    bar_probs = bar_probs.fillna(method="bfill").fillna(0.5)
                    pred_columns[adapter.output_label] = bar_probs.values
                else:
                    pred_columns[adapter.output_label] = probs

                logger.info(
                    "  {} → {} predictions, prob range [{:.4f}, {:.4f}]",
                    adapter.name,
                    len(pred_columns[adapter.output_label]),
                    float(pred_columns[adapter.output_label].min()),
                    float(pred_columns[adapter.output_label].max()),
                )
            except Exception as exc:
                logger.error(
                    "[diag] Batch predict_proba FAILED for '{}' ({}): {}",
                    adapter.name, type(exc).__name__, exc,
                )

        # ----------------------------------------------------------
        # Write CSV
        # ----------------------------------------------------------
        if pred_columns:
            pred_df = pd.DataFrame(pred_columns, index=timestamps)
            pred_df.index.name = "timestamp"
            pred_df.to_csv(output_dir / f"{ticker}_predictions.csv")
            logger.info(
                "Wrote {} prediction columns × {} rows for {} to {}",
                len(pred_columns), n_bars, ticker, output_dir,
            )
            summary["per_ticker"][ticker] = n_bars
            summary["total_rows"] += n_bars
        else:
            logger.warning("No predictions generated for {}", ticker)
            summary["per_ticker"][ticker] = 0

        summary["tickers_processed"] += 1

        if progress_callback is not None:
            progress_callback(t_idx + 1, total_tickers, ticker)

    logger.info(
        "Batch inference complete: {} tickers, {} total prediction rows",
        summary["tickers_processed"],
        summary["total_rows"],
    )
    return summary


