"""
Backtest reporting for ApexQuant.

Generates JSON summaries, trade CSVs, equity CSVs, and comparison
charts from :class:`BacktestResult` objects.

Usage::

    from backtest.reporter import BacktestReporter
    from backtest.engine import BacktestResult

    reporter = BacktestReporter(result, run_id="20240115_ai")
    reporter.save_all()

    # AI vs Technical comparison table
    table = BacktestReporter.generate_comparison_table(ai_result, tech_result)
"""

__all__ = ["BacktestReporter"]

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from backtest.engine import BacktestResult


class BacktestReporter:
    """Generates reports from a completed backtest.

    Attributes:
        result: The backtest result to report on.
        run_id: Unique run identifier for file naming.
        output_dir: Root directory for all output files.
    """

    def __init__(
        self,
        result: BacktestResult,
        run_id: str = "",
        output_dir: str = "results/runs",
    ) -> None:
        """Initialise the reporter.

        Args:
            result: A completed :class:`BacktestResult`.
            run_id: Unique run identifier.  If empty, auto-generated
                from strategy name and current time.
            output_dir: Root directory for output files.
        """
        self.result = result
        self.run_id = run_id or self._generate_run_id()
        self.output_dir = Path(output_dir) / self.run_id

        logger.info("BacktestReporter: run_id={}, output={}", self.run_id, self.output_dir)

    # ------------------------------------------------------------------
    # Save all
    # ------------------------------------------------------------------

    def save_all(self) -> Path:
        """Save all report artifacts.

        Creates the output directory and saves JSON metrics,
        trades CSV, equity CSV, and charts.

        Returns:
            Path to the output directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_json(str(self.output_dir / "metrics.json"))
        self.save_trades_csv(str(self.output_dir / "trades.csv"))
        self.save_equity_csv(str(self.output_dir / "equity.csv"))
        self.save_charts(str(self.output_dir / "charts"))

        logger.info("All reports saved to {}", self.output_dir)
        return self.output_dir

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, path: str) -> None:
        """Serialize metrics and metadata to JSON.

        Args:
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "run_id": self.run_id,
            "strategy_name": self.result.strategy_name,
            "start_date": _dt_to_str(self.result.start_date),
            "end_date": _dt_to_str(self.result.end_date),
            "total_trades": len(self.result.trades),
            "metrics": _make_json_safe(self.result.metrics),
            "config_snapshot": _make_json_safe(self.result.config_snapshot),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Metrics JSON saved to {}", file_path)

    # ------------------------------------------------------------------
    # Trades CSV
    # ------------------------------------------------------------------

    def save_trades_csv(self, path: str) -> None:
        """Export all closed trades to CSV.

        Columns: trade_id, ticker, timestamp, signal, entry_price,
        exit_price, size, commission, slippage, pnl, exit_reason,
        bars_held.

        Args:
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for t in self.result.trades:
            portfolio_value = t.notional
            position_value = portfolio_value * t.size if t.size else 0.0
            pnl_dollars = position_value * t.pnl if t.pnl is not None else 0.0
            rows.append({
                "trade_id": t.trade_id,
                "ticker": t.ticker,
                "timestamp": t.timestamp,
                "signal": t.signal.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_timestamp": t.exit_timestamp,
                "size": t.size,
                "commission": t.commission,
                "slippage": t.slippage,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "conviction_tier": t.conviction_tier or None,
                "portfolio_value": portfolio_value,
                "position_value": position_value,
                "pnl_dollars": pnl_dollars,
                "entry_tp_pct": t.entry_tp_pct,
                "entry_sl_pct": t.entry_sl_pct,
                "tranches_exited": t.tranches_exited,
            })

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        logger.info("Trades CSV saved to {} ({} trades)", file_path, len(rows))

    # ------------------------------------------------------------------
    # Equity CSV
    # ------------------------------------------------------------------

    def save_equity_csv(self, path: str) -> None:
        """Export equity curve to CSV with daily returns.

        Columns: timestamp, portfolio_value, daily_return.

        Args:
            path: Output file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        equity = self.result.equity_curve
        if len(equity) == 0:
            pd.DataFrame(
                columns=["timestamp", "portfolio_value", "daily_return"]
            ).to_csv(file_path, index=False)
            logger.info("Empty equity CSV saved to {}", file_path)
            return

        daily_return = equity.pct_change().fillna(0.0)

        df = pd.DataFrame({
            "timestamp": equity.index,
            "portfolio_value": equity.values,
            "daily_return": daily_return.values,
        })
        df.to_csv(file_path, index=False)
        logger.info("Equity CSV saved to {} ({} points)", file_path, len(df))

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def save_charts(self, charts_dir: str) -> None:
        """Generate and save charts as PNG.

        Creates three charts:

        1. ``equity_curve.png`` — equity curve with drawdown shading
        2. ``per_ticker_winrate.png`` — bar chart of win rate per ticker
        3. ``trade_distribution.png`` — histogram of trade PnL

        Args:
            charts_dir: Directory to save chart PNGs.
        """
        charts_path = Path(charts_dir)
        charts_path.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping chart generation")
            return

        self._chart_equity(plt, charts_path / "equity_curve.png")
        self._chart_per_ticker_winrate(plt, charts_path / "per_ticker_winrate.png")
        self._chart_trade_distribution(plt, charts_path / "trade_distribution.png")

        logger.info("Charts saved to {}", charts_path)

    def save_comparison_charts(
        self,
        ai_result: BacktestResult,
        tech_result: BacktestResult,
        charts_dir: str,
    ) -> None:
        """Generate the AI vs Technical comparison equity chart.

        Args:
            ai_result: AI strategy backtest result.
            tech_result: Technical strategy backtest result.
            charts_dir: Directory to save the chart.
        """
        charts_path = Path(charts_dir)
        charts_path.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping comparison chart")
            return

        self._chart_equity_comparison(
            plt, ai_result, tech_result,
            charts_path / "equity_comparison.png",
        )

    # ------------------------------------------------------------------
    # Individual chart methods
    # ------------------------------------------------------------------

    def _chart_equity(self, plt: Any, path: Path) -> None:
        """Equity curve with drawdown shading.

        Args:
            plt: matplotlib.pyplot module.
            path: Output file path.
        """
        equity = self.result.equity_curve
        if len(equity) < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity.index, equity.values, label="Portfolio Value", color="#2196F3", linewidth=1.5)

        # Drawdown shading
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        ax.fill_between(
            equity.index, equity.values, cummax.values,
            where=equity < cummax,
            color="#F44336", alpha=0.15, label="Drawdown",
        )

        ax.set_title(f"ApexQuant - {self.result.strategy_name} Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _chart_equity_comparison(
        plt: Any,
        ai_result: BacktestResult,
        tech_result: BacktestResult,
        path: Path,
    ) -> None:
        """AI vs Technical equity comparison chart.

        Args:
            plt: matplotlib.pyplot module.
            ai_result: AI strategy result.
            tech_result: Technical strategy result.
            path: Output file path.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ai_eq = ai_result.equity_curve
        tech_eq = tech_result.equity_curve

        if len(ai_eq) > 0:
            ax.plot(ai_eq.index, ai_eq.values, label="AI Strategy", color="#2196F3", linewidth=1.5)
            cummax = ai_eq.cummax()
            ax.fill_between(
                ai_eq.index, ai_eq.values, cummax.values,
                where=ai_eq < cummax,
                color="#F44336", alpha=0.1,
            )

        if len(tech_eq) > 0:
            ax.plot(tech_eq.index, tech_eq.values, label="EMA+RSI Baseline", color="#FF9800", linewidth=1.5, linestyle="--")

        ax.set_title("ApexQuant - AI Strategy vs EMA+RSI Baseline")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _chart_per_ticker_winrate(self, plt: Any, path: Path) -> None:
        """Bar chart of win rate per ticker.

        Args:
            plt: matplotlib.pyplot module.
            path: Output file path.
        """
        trades = self.result.trades
        if not trades:
            return

        # Compute win rate per ticker
        ticker_stats: dict[str, dict[str, int]] = {}
        for t in trades:
            if t.pnl is None:
                continue
            if t.ticker not in ticker_stats:
                ticker_stats[t.ticker] = {"wins": 0, "total": 0}
            ticker_stats[t.ticker]["total"] += 1
            if t.pnl > 0:
                ticker_stats[t.ticker]["wins"] += 1

        if not ticker_stats:
            return

        tickers = sorted(ticker_stats.keys())
        win_rates = [
            ticker_stats[t]["wins"] / ticker_stats[t]["total"] * 100
            for t in tickers
        ]
        colors = ["#4CAF50" if wr > 50 else "#F44336" for wr in win_rates]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(tickers, win_rates, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(y=50, color="#9E9E9E", linestyle="--", linewidth=1, label="50% (random)")

        # Value labels on bars
        for bar, wr in zip(bars, win_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{wr:.1f}%", ha="center", va="bottom", fontsize=9,
            )

        ax.set_title(f"Win Rate per Ticker - {self.result.strategy_name}")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Win Rate (%)")
        ax.set_ylim(0, max(win_rates) + 15)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _chart_trade_distribution(self, plt: Any, path: Path) -> None:
        """Histogram of trade PnL.

        Args:
            plt: matplotlib.pyplot module.
            path: Output file path.
        """
        pnls = [t.pnl * 100 for t in self.result.trades if t.pnl is not None]
        if not pnls:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            pnls, bins=min(50, max(10, len(pnls) // 3)),
            color="#2196F3", edgecolor="white", alpha=0.8,
        )

        mean_pnl = float(np.mean(pnls))
        ax.axvline(x=0, color="#9E9E9E", linestyle="--", linewidth=1.5, label="Zero")
        ax.axvline(x=mean_pnl, color="#FF9800", linestyle="-", linewidth=1.5, label=f"Mean ({mean_pnl:.2f}%)")

        ax.set_title(f"Trade PnL Distribution - {self.result.strategy_name}")
        ax.set_xlabel("PnL (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    @staticmethod
    def generate_comparison_table(
        ai_result: BacktestResult,
        tech_result: BacktestResult,
        qc_baseline: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Generate a side-by-side comparison table.

        Args:
            ai_result: AI strategy backtest result (metrics must be populated).
            tech_result: Technical strategy backtest result.
            qc_baseline: Optional QuantConnect baseline metrics dict.
                Defaults to ``{"sharpe_ratio": -1.12, "total_trades": 589}``.

        Returns:
            DataFrame with rows = metrics, columns = strategies.
        """
        if qc_baseline is None:
            qc_baseline = {
                "sharpe_ratio": -1.12,
                "total_trades": 589,
                "total_return": None,
                "sortino_ratio": None,
                "max_drawdown": None,
                "win_rate": None,
                "profit_factor": None,
                "avg_trade_pnl": None,
                "calmar_ratio": None,
            }

        metric_names = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "max_drawdown_duration_days",
            "win_rate",
            "profit_factor",
            "total_trades",
            "avg_trade_pnl",
            "avg_bars_held",
            "calmar_ratio",
            "long_trades",
            "short_trades",
        ]

        rows: list[dict[str, Any]] = []
        for metric in metric_names:
            rows.append({
                "Metric": metric,
                "AI Strategy": ai_result.metrics.get(metric),
                "EMA+RSI": tech_result.metrics.get(metric),
                "QC Baseline": qc_baseline.get(metric),
            })

        df = pd.DataFrame(rows).set_index("Metric")

        # Format for display
        _fmt_map = {
            "total_return": _fmt_pct,
            "annualized_return": _fmt_pct,
            "max_drawdown": _fmt_pct,
            "win_rate": _fmt_pct,
            "sharpe_ratio": _fmt_float,
            "sortino_ratio": _fmt_float,
            "profit_factor": _fmt_float,
            "avg_trade_pnl": _fmt_pct,
            "calmar_ratio": _fmt_float,
            "avg_bars_held": _fmt_float,
        }

        # Log the table
        logger.info("=" * 70)
        logger.info("{:>30s}  {:>12s}  {:>12s}  {:>12s}", "Metric", "AI Strategy", "EMA+RSI", "QC Baseline")
        logger.info("-" * 70)
        for _, row in df.iterrows():
            metric = row.name
            fmt_fn = _fmt_map.get(metric, _fmt_val)
            ai_val = fmt_fn(row["AI Strategy"])
            tech_val = fmt_fn(row["EMA+RSI"])
            qc_val = fmt_fn(row["QC Baseline"])
            logger.info("{:>30s}  {:>12s}  {:>12s}  {:>12s}", metric, ai_val, tech_val, qc_val)
        logger.info("=" * 70)

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_run_id(self) -> str:
        """Generate a unique run ID from strategy name and timestamp.

        Returns:
            Run ID string like ``"20240115_143022_ai_strategy"``.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.result.strategy_name or "unknown"
        return f"{ts}_{name}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dt_to_str(dt: datetime | None) -> str | None:
    """Convert a datetime to ISO string, handling None."""
    if dt is None:
        return None
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fmt_pct(val: Any) -> str:
    """Format a value as a percentage."""
    if val is None:
        return "-"
    return f"{float(val):.2%}"


def _fmt_float(val: Any) -> str:
    """Format a value as a float."""
    if val is None:
        return "-"
    return f"{float(val):.2f}"


def _fmt_val(val: Any) -> str:
    """Format a general value."""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)
