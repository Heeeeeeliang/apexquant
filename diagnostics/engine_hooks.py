"""
Post-run diagnostics collector.

Attaches to a completed BacktestResult and runs all diagnostic checks,
producing a single :class:`DiagnosticsReport` that the UI can render.

Usage::

    from diagnostics.engine_hooks import collect_diagnostics
    report = collect_diagnostics(result, config)
"""

__all__ = ["DiagnosticsReport", "collect_diagnostics"]

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diagnostics.equity_scan import EquityScanReport, scan_equity_curve
from diagnostics.feature_drift import FeatureDriftReport, check_feature_drift
from diagnostics.pnl_autocorrelation import PnlAutocorrReport, scan_pnl_autocorrelation
from diagnostics.trade_clustering import ClusteringScanReport, scan_trade_clustering
from diagnostics.trade_quality import TradeQualityReport, compute_trade_quality


@dataclass
class DiagnosticsReport:
    """Aggregated diagnostics for a single backtest run."""

    trade_quality: TradeQualityReport = field(default_factory=TradeQualityReport)
    feature_drift: list[FeatureDriftReport] = field(default_factory=list)
    equity_scan: EquityScanReport = field(default_factory=EquityScanReport)
    trade_clustering: ClusteringScanReport = field(default_factory=ClusteringScanReport)
    pnl_autocorrelation: PnlAutocorrReport = field(default_factory=PnlAutocorrReport)
    collection_time_ms: float = 0.0

    @property
    def has_drift_errors(self) -> bool:
        return any(d.drift_severity == "error" for d in self.feature_drift)

    @property
    def has_drift_warnings(self) -> bool:
        return any(d.drift_severity == "warning" for d in self.feature_drift)


def collect_diagnostics(
    result: Any,
    config: dict[str, Any] | None = None,
    runtime_columns: list[str] | None = None,
) -> DiagnosticsReport:
    """Run all diagnostics on a completed BacktestResult.

    Args:
        result: A completed BacktestResult (or any object with .trades).
        config: Optional CONFIG dict (unused currently, reserved for
            future diagnostics that need config context).
        runtime_columns: Feature column names used during inference.
            If provided, runs feature drift checks against all models.

    Returns:
        A :class:`DiagnosticsReport`.
    """
    t0 = time.perf_counter()

    trades = getattr(result, "trades", result)
    tq = compute_trade_quality(trades)

    # Feature drift
    drift_reports: list[FeatureDriftReport] = []
    if runtime_columns is not None:
        models_dir = Path("models")
        if models_dir.exists():
            for meta_path in models_dir.rglob("meta.json"):
                model_dir = meta_path.parent
                fn_path = model_dir / "feature_names.json"
                if fn_path.exists():
                    drift_reports.append(
                        check_feature_drift(runtime_columns, model_dir)
                    )

    # Equity curve scan
    equity = getattr(result, "equity_curve", None)
    eq_scan = scan_equity_curve(equity)

    # Trade clustering
    clustering = scan_trade_clustering(trades)

    # PnL autocorrelation
    pnl_ac = scan_pnl_autocorrelation(trades)

    elapsed = (time.perf_counter() - t0) * 1000

    return DiagnosticsReport(
        trade_quality=tq,
        feature_drift=drift_reports,
        equity_scan=eq_scan,
        trade_clustering=clustering,
        pnl_autocorrelation=pnl_ac,
        collection_time_ms=elapsed,
    )
