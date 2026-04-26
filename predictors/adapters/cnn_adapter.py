"""
CNN adapter — wraps a pre-trained dual-branch MultiScaleCNN for Layer 2.

The checkpoint stores the model state dict, architecture parameters,
and per-branch Z-score scalers so that inference is self-contained.

Weight layout::

    models/layer2/tp_bottom/multiscale_cnn_v1/
        weights.pt   — checkpoint dict (see below)
        meta.json    — optional metadata

Checkpoint keys::

    model_state_dict  — ``nn.Module.state_dict()``
    n_short_feat      — int (e.g. 66)
    n_long_feat       — int (e.g. 53)
    short_win         — int (e.g. 30)
    long_win          — int (e.g. 48)
    s_mu, s_sigma     — np.ndarray, Z-score scaler for short branch
    l_mu, l_sigma     — np.ndarray, Z-score scaler for long branch
    task              — str (e.g. ``"bottom"``)

.. note::

    All ``torch`` imports are deferred to :meth:`CnnAdapter.load` /
    :meth:`CnnAdapter.predict` so that the adapter can be *imported and
    registered* even when PyTorch is unavailable or broken (e.g. DLL
    load failures on some Windows environments).
"""

__all__ = ["CnnAdapter"]

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from data.bar import Bar
from predictors.base import Context, Predictor
from predictors.result import PredictionResult

# Streamlit-aware model cache: avoids reloading checkpoints on page reruns.
try:
    import streamlit as _st_cache

    @_st_cache.cache_resource(show_spinner=False)
    def _cached_load_torch(path_str: str, _mtime: float):
        import torch
        return torch.load(path_str, map_location="cpu", weights_only=False)
except Exception:
    def _cached_load_torch(path_str: str, _mtime: float):
        import torch
        return torch.load(path_str, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# MultiScaleCNN — exact architecture matching the training checkpoint.
# Defined as a module-level factory so it can be called inside load()
# without importing torch at the top level.
# ---------------------------------------------------------------------------

def _build_multiscale_cnn(n_short_feat: int, n_long_feat: int) -> Any:
    """Build and return a MultiScaleCNN ``nn.Module``.

    Imports ``torch`` on first call.  Raises ``ImportError`` if torch is
    not available.
    """
    import torch
    import torch.nn as nn

    class MultiScaleCNN(nn.Module):
        """Dual-branch CNN: short branch (15min) + long branch (1hour)."""

        def __init__(self, n_short: int, n_long: int) -> None:
            super().__init__()
            self.short_branch = nn.Sequential(
                nn.Conv1d(n_short, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.long_branch = nn.Sequential(
                nn.Conv1d(n_long, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
            )

        def forward(self, x_short: torch.Tensor, x_long: torch.Tensor) -> torch.Tensor:
            s = self.short_branch(x_short.transpose(1, 2))
            l = self.long_branch(x_long.transpose(1, 2))
            return self.head(torch.cat([s, l], dim=1)).squeeze(-1)

    return MultiScaleCNN(n_short_feat, n_long_feat)


# ---------------------------------------------------------------------------
# CnnAdapter
# ---------------------------------------------------------------------------

class CnnAdapter(Predictor):
    """Dual-branch MultiScaleCNN adapter for turning-point detection.

    Attributes:
        name: Registry key (e.g. ``"cnn_turning_bottom"``).
        output_label: Label written to ``bar.predictions``.
        _model_dir: Path to the weight directory.
    """

    update_freq: str = "bar"
    default_validity: int = 3600

    def __init__(
        self,
        name: str,
        output_label: str,
        model_dir: str | Path,
    ) -> None:
        self.name = name
        self.output_label = output_label
        self._model_dir = Path(model_dir)
        self._model: Any = None
        self._s_mu: np.ndarray | None = None
        self._s_sigma: np.ndarray | None = None
        self._l_mu: np.ndarray | None = None
        self._l_sigma: np.ndarray | None = None
        self._short_win: int = 30
        self._long_win: int = 48
        self._task: str = ""
        self._meta: dict[str, Any] = {}
        super().__init__()

    def load(self) -> None:
        """Load checkpoint, rebuild architecture, and restore scalers."""
        weights_path = self._model_dir / "weights.pt"
        meta_path = self._model_dir / "meta.json"

        ckpt = _cached_load_torch(
            str(weights_path), weights_path.stat().st_mtime
        )

        n_short = int(ckpt["n_short_feat"])
        n_long = int(ckpt["n_long_feat"])
        self._short_win = int(ckpt["short_win"])
        self._long_win = int(ckpt["long_win"])
        self._task = str(ckpt.get("task", ""))

        model = _build_multiscale_cnn(n_short, n_long)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self._model = model

        self._s_mu = np.array(ckpt["s_mu"])
        self._s_sigma = np.array(ckpt["s_sigma"])
        self._l_mu = np.array(ckpt["l_mu"])
        self._l_sigma = np.array(ckpt["l_sigma"])

        logger.info(
            "Loaded CNN [{}]: n_short={}, n_long={}, short_win={}, long_win={}",
            self._task, n_short, n_long, self._short_win, self._long_win,
        )

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)

    def is_ready(self) -> bool:
        """Check that weight file exists on disk."""
        return (self._model_dir / "weights.pt").exists()

    def get_version(self) -> str:
        return self._meta.get("version", f"{self.name}_v1")

    def predict(self, bar: Bar, context: Context) -> PredictionResult:
        """Run CNN inference on a single bar."""
        import torch

        if self._model is None:
            self.load()

        short_window = self._build_window(context.history, self._short_win)
        long_window = self._build_long_window(context.history, self._long_win)

        short_norm = (short_window - self._s_mu) / (self._s_sigma + 1e-8)
        long_norm = (long_window - self._l_mu) / (self._l_sigma + 1e-8)

        x_short = torch.tensor(short_norm, dtype=torch.float32).unsqueeze(0)
        x_long = torch.tensor(long_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logit = self._model(x_short, x_long).item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())

        return self._make_result(
            prob=prob,
            raw_score=logit,
            confidence=abs(prob - 0.5) * 2.0,
        )

    def _build_window(
        self, history: list[Bar], win: int
    ) -> np.ndarray:
        """Build a (win, n_feat) array from the most recent bars."""
        n_expected = int(self._s_mu.shape[0]) if self._s_mu is not None else None
        recent = history[-win:] if len(history) >= win else history
        rows: list[list[float]] = []
        for b in recent:
            if b.features:
                row = list(b.features.values())
            else:
                row = self._bar_to_features(b)
            rows.append(self._align_row(row, n_expected, "short"))

        arr = np.array(rows, dtype=np.float64)
        if arr.shape[0] < win:
            pad = np.zeros((win - arr.shape[0], arr.shape[1]), dtype=np.float64)
            arr = np.vstack([pad, arr])
        return arr

    def _build_long_window(
        self, history: list[Bar], win: int
    ) -> np.ndarray:
        """Build the long-frequency window via stride-sampling."""
        n_expected = int(self._l_mu.shape[0]) if self._l_mu is not None else None
        stride = 4
        sampled = history[::stride]
        sampled = sampled[-win:] if len(sampled) >= win else sampled

        rows: list[list[float]] = []
        for b in sampled:
            if b.features:
                row = list(b.features.values())
            else:
                row = self._bar_to_features(b)
            rows.append(self._align_row(row, n_expected, "long"))

        if not rows:
            n_feat = n_expected or 53
            return np.zeros((win, n_feat), dtype=np.float64)

        arr = np.array(rows, dtype=np.float64)
        if arr.shape[0] < win:
            pad = np.zeros((win - arr.shape[0], arr.shape[1]), dtype=np.float64)
            arr = np.vstack([pad, arr])
        return arr

    def _align_row(
        self, row: list[float], expected: int | None, branch: str
    ) -> list[float]:
        """Pad or truncate a single row to match scaler dimensions."""
        if expected is None or len(row) == expected:
            return row
        if not hasattr(self, "_align_warned"):
            self._align_warned: set[tuple[str, int, int]] = set()
        key = (branch, len(row), expected)
        if key not in self._align_warned:
            self._align_warned.add(key)
            logger.warning(
                "CnnAdapter [{}] {}: row has {} features, scaler expects {}; {}",
                self.name, branch, len(row), expected,
                "padding" if len(row) < expected else "truncating",
            )
        if len(row) < expected:
            return row + [0.0] * (expected - len(row))
        return row[:expected]

    @staticmethod
    def _bar_to_features(bar: Bar) -> list[float]:
        """Fallback feature extraction from bar attributes."""
        vals: list[float] = [
            bar.open, bar.high, bar.low, bar.close, bar.volume,
        ]
        for attr in [
            "ema_8", "ema_21", "ema_50", "rsi_14", "macd",
            "macd_signal", "macd_hist", "atr_14", "bb_upper",
            "bb_lower", "bb_mid", "volume_ratio", "vwap", "obv",
            "adx_14", "stoch_k", "stoch_d", "willr_14", "cci_20", "mfi_14",
        ]:
            v = getattr(bar, attr, None)
            vals.append(v if v is not None else 0.0)
        return vals
