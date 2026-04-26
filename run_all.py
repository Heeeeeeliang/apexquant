"""
ApexQuant CLI -- unified entry point for pipeline, backtest, and frontend.

Supports running individual pipeline steps, full end-to-end execution,
backtest-only mode with strategy selection, AI-vs-Technical comparison,
and launching the Streamlit frontend.

Usage::

    python run_all.py --all                        # Full pipeline + backtest
    python run_all.py --steps 01 02 03             # Specific pipeline steps
    python run_all.py --backtest-only               # Skip training, backtest only
    python run_all.py --strategy ai                 # Strategy: ai | technical | path.py
    python run_all.py --start 2022-01-01 --end 2022-12-31
    python run_all.py --compare                     # AI vs Technical comparison
    python run_all.py --frontend                    # Launch Streamlit app
"""

__all__: list[str] = []

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Loguru setup
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<7}</level> | "
        "<cyan>{message}</cyan>"
    ),
)

# ---------------------------------------------------------------------------
# Pipeline step registry
# ---------------------------------------------------------------------------

_STEPS: dict[str, dict[str, str]] = {
    "01": {
        "name": "Volatility Prediction",
        "script": "predictors/p01_volatility.py",
        "description": "Layer 1: LightGBM volatility classifier (DA=83.6%)",
    },
    "02": {
        "name": "Turning-Point Detection",
        "script": "predictors/p02_turning_point.py",
        "description": "Layer 2: Multi-scale CNN turning-point detector (AUC=0.826)",
    },
    "03": {
        "name": "Meta-Label Filtering",
        "script": "predictors/p03_meta_label.py",
        "description": "Layer 3: LightGBM meta-label trade filter (WR=64.5%)",
    },
    "04": {
        "name": "Aggregator Fitting",
        "script": "predictors/aggregator.py",
        "description": "Fit LearnedAggregator on validation set",
    },
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Load config from default, merge with local override if present.

    Returns:
        Merged CONFIG dict.
    """
    from config.default import CONFIG, deep_merge

    config = deepcopy(CONFIG)

    # Merge local override
    local_path = Path("config/local_override.json")
    if local_path.exists():
        try:
            with open(local_path, encoding="utf-8") as f:
                local = json.load(f)
            config = deep_merge(config, local)
            logger.info("Merged local override from {}", local_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load local override: {}", exc)

    # Also try config/local.py
    local_py = Path("config/local.py")
    if local_py.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("local_config", str(local_py))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "CONFIG"):
                    config = deep_merge(config, mod.CONFIG)
                    logger.info("Merged local.py config")
        except Exception as exc:
            logger.warning("Failed to load config/local.py: {}", exc)

    return config


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_pipeline_steps(
    steps: list[str], config: dict[str, Any]
) -> bool:
    """Run pipeline steps via the compute backend.

    Args:
        steps: List of step IDs (e.g. ["01", "02"]).
        config: Full CONFIG dict.

    Returns:
        True if all steps completed successfully.
    """
    from compute import get_backend

    backend = get_backend(config)
    logger.info("Compute backend: {}", type(backend).__name__)

    try:
        from tqdm import tqdm
        step_iter = tqdm(steps, desc="Pipeline", unit="step")
    except ImportError:
        step_iter = steps

    all_ok = True
    for step_id in step_iter:
        step = _STEPS.get(step_id)
        if step is None:
            logger.error("Unknown step: {}. Valid: {}", step_id, sorted(_STEPS.keys()))
            all_ok = False
            continue

        logger.info("=" * 60)
        logger.info("Step {}: {}", step_id, step["name"])
        logger.info("  {}", step["description"])
        logger.info("  Script: {}", step["script"])
        logger.info("=" * 60)

        script_path = Path(step["script"])
        if not script_path.exists():
            logger.error("Script not found: {}", script_path)
            all_ok = False
            continue

        try:
            job_id = backend.submit_job(
                str(script_path), config,
                job_name=f"step_{step_id}_{step['name']}",
            )
            logger.info("Job submitted: {}", job_id)

            # Poll for completion
            while True:
                status = backend.get_status(job_id)
                state = status.get("status", "unknown")

                if state in ("completed", "done", "finished"):
                    logger.info("Step {} completed successfully", step_id)
                    break
                elif state in ("failed", "error"):
                    logger.error(
                        "Step {} failed: {}",
                        step_id,
                        status.get("message", "unknown error"),
                    )
                    # Print logs
                    try:
                        logs = backend.get_logs(job_id)
                        if logs:
                            logger.error("--- Job Logs ---")
                            for line in logs:
                                logger.error("  {}", line.rstrip())
                    except Exception:
                        pass
                    all_ok = False
                    break
                else:
                    time.sleep(2)

        except Exception as exc:
            logger.error("Step {} execution failed: {}", step_id, exc)
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------

def _run_backtest(
    config: dict[str, Any],
    strategy: str,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Run a single-strategy backtest and print results.

    Args:
        config: Full CONFIG dict.
        strategy: Strategy name or file path.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    """
    from backtest.runner import run_backtest

    logger.info("=" * 60)
    logger.info("Running backtest: strategy={}", strategy)
    logger.info("=" * 60)

    result = run_backtest(
        config,
        strategy_name=strategy,
        start_date=start_date,
        end_date=end_date,
        save_results=True,
    )

    _print_results(result.metrics, title=f"Backtest Results ({strategy})")


def _run_comparison(
    config: dict[str, Any],
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Run AI vs Technical comparison and print results table.

    Args:
        config: Full CONFIG dict.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
    """
    from backtest.runner import run_comparison

    logger.info("=" * 60)
    logger.info("Running AI vs Technical Comparison")
    logger.info("=" * 60)

    ai_result, tech_result = run_comparison(
        config,
        start_date=start_date,
        end_date=end_date,
        save_results=True,
    )

    _print_comparison(ai_result.metrics, tech_result.metrics)


def _print_results(metrics: dict[str, Any], title: str = "Results") -> None:
    """Print a formatted metrics table.

    Args:
        metrics: Metrics dict from BacktestResult.
        title: Table header.
    """
    logger.info("")
    logger.info("=" * 50)
    logger.info("  {}", title)
    logger.info("=" * 50)

    if not metrics:
        logger.warning("  No metrics available")
        return

    display_order = [
        ("total_return", "Total Return", True),
        ("annualized_return", "Annual Return", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
        ("sortino_ratio", "Sortino Ratio", False),
        ("calmar_ratio", "Calmar Ratio", False),
        ("max_drawdown", "Max Drawdown", True),
        ("win_rate", "Win Rate", True),
        ("profit_factor", "Profit Factor", False),
        ("total_trades", "Total Trades", False),
        ("avg_trade_pnl", "Avg Trade PnL", True),
        ("avg_bars_held", "Avg Bars Held", False),
    ]

    for key, label, is_pct in display_order:
        val = metrics.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            formatted = f"{val:.2%}" if is_pct else f"{val:.4f}"
        else:
            formatted = str(val)
        logger.info("  {:>20s}  {}", label, formatted)

    logger.info("=" * 50)


def _print_comparison(
    ai_metrics: dict[str, Any], tech_metrics: dict[str, Any]
) -> None:
    """Print a side-by-side AI vs Technical comparison table.

    Args:
        ai_metrics: AI strategy metrics.
        tech_metrics: Technical strategy metrics.
    """
    qc_baseline = {"sharpe_ratio": -1.12, "total_trades": 589, "win_rate": 0.50}

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "  {:>22s}  {:>12s}  {:>12s}  {:>12s}",
        "Metric", "AI Strategy", "EMA+RSI", "QC Baseline",
    )
    logger.info("-" * 72)

    rows = [
        ("total_return", "Total Return", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
        ("sortino_ratio", "Sortino Ratio", False),
        ("max_drawdown", "Max Drawdown", True),
        ("win_rate", "Win Rate", True),
        ("profit_factor", "Profit Factor", False),
        ("total_trades", "Total Trades", False),
        ("avg_trade_pnl", "Avg Trade PnL", True),
        ("avg_bars_held", "Avg Bars Held", False),
    ]

    def _fmt(val: Any, is_pct: bool) -> str:
        if val is None:
            return "--"
        if isinstance(val, float):
            return f"{val:.2%}" if is_pct else f"{val:.4f}"
        return str(val)

    for key, label, is_pct in rows:
        ai_val = _fmt(ai_metrics.get(key), is_pct)
        tech_val = _fmt(tech_metrics.get(key), is_pct)
        qc_val = _fmt(qc_baseline.get(key), is_pct)
        logger.info("  {:>22s}  {:>12s}  {:>12s}  {:>12s}", label, ai_val, tech_val, qc_val)

    logger.info("=" * 72)

    # Highlight winner
    ai_sharpe = ai_metrics.get("sharpe_ratio", 0) or 0
    tech_sharpe = tech_metrics.get("sharpe_ratio", 0) or 0
    delta = ai_sharpe - tech_sharpe
    logger.info("")
    logger.info(
        "  AI vs Technical: Sharpe delta = {:+.2f} ({})",
        delta,
        "AI wins" if delta > 0 else "Technical wins" if delta < 0 else "Tie",
    )


# ---------------------------------------------------------------------------
# Frontend launch
# ---------------------------------------------------------------------------

def _launch_frontend() -> None:
    """Launch the Streamlit frontend."""
    logger.info("Launching Streamlit frontend...")
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port=8501",
        "--server.address=localhost",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Frontend stopped by user")
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description="ApexQuant — AI-driven trading research platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_all.py --all                        Full pipeline + backtest
  python run_all.py --steps 01 02 03             Train specific layers
  python run_all.py --backtest-only --compare    AI vs Technical comparison
  python run_all.py --strategy path/to/strat.py  User strategy backtest
  python run_all.py --frontend                   Launch Streamlit UI
""",
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all", action="store_true",
        help="Run full pipeline (steps 01-04) then backtest",
    )
    mode.add_argument(
        "--steps", nargs="+", metavar="STEP",
        choices=list(_STEPS.keys()),
        help="Run specific pipeline steps: 01 02 03 04",
    )
    mode.add_argument(
        "--backtest-only", action="store_true",
        help="Skip training, run backtest only (requires pre-trained models)",
    )
    mode.add_argument(
        "--frontend", action="store_true",
        help="Launch the Streamlit frontend",
    )

    # Backtest options
    parser.add_argument(
        "--strategy", type=str, default="ai",
        help='Strategy: "ai", "technical", or path to a .py file (default: ai)',
    )
    parser.add_argument(
        "--start", type=str, default=None, metavar="DATE",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None, metavar="DATE",
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both AI and Technical strategies and print comparison",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Adjust log level
    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level:<7}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

    logger.info("=" * 60)
    logger.info("  ApexQuant — AI-Driven Trading Research Platform")
    logger.info("  COMP3931 — University of Leeds 2024/25")
    logger.info("=" * 60)

    # Load config
    config = _load_config()
    logger.info("Config loaded (backend={})", config.get("compute", {}).get("backend", "local"))

    # --- Frontend mode ---
    if args.frontend:
        _launch_frontend()
        return

    # --- Full pipeline ---
    if args.all:
        logger.info("Mode: Full Pipeline")
        ok = _run_pipeline_steps(list(_STEPS.keys()), config)
        if not ok:
            logger.error("Pipeline failed — skipping backtest")
            sys.exit(1)

        # Run backtest after pipeline
        if args.compare:
            _run_comparison(config, args.start, args.end)
        else:
            _run_backtest(config, args.strategy, args.start, args.end)
        return

    # --- Specific steps ---
    if args.steps:
        logger.info("Mode: Steps {}", args.steps)
        ok = _run_pipeline_steps(args.steps, config)
        if not ok:
            sys.exit(1)
        return

    # --- Backtest only ---
    if args.backtest_only or args.compare:
        if args.compare:
            _run_comparison(config, args.start, args.end)
        else:
            _run_backtest(config, args.strategy, args.start, args.end)
        return

    # No mode selected — show help
    parser.print_help()


if __name__ == "__main__":
    main()
