"""
Layer 2 — Turning-Point Detection (Multi-Scale CNN).

Detects local price turning points (peaks and troughs) using a
1-D CNN with multiple kernel sizes to capture patterns at
different time scales.

Thesis result: AUC = 0.826.

Usage::

    from predictors.registry import REGISTRY
    tp = REGISTRY.get("turning_point")
    metrics = tp.train(train_bars, val_bars)
    bars = tp.predict(test_bars)
    print(bars[0].predictions["turning_point"])  # e.g. 0.72
"""

__all__ = ["TurningPointPredictor"]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import get
from data.bar import Bar
from predictors.base import BasePredictor
from predictors.registry import REGISTRY


class MultiScaleCNN(nn.Module):
    """1-D CNN with parallel convolution branches of different kernel sizes.

    Each branch captures patterns at a different time scale.  Outputs
    are concatenated and passed through a classification head.

    Args:
        n_features: Number of input feature channels.
        kernel_sizes: List of 1-D kernel widths.
        num_filters: Number of filters per branch.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        kernel_sizes: list[int],
        num_filters: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(n_features, num_filters, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            self.branches.append(branch)

        total_filters = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_features, seq_len).

        Returns:
            Logits of shape (batch, 1).
        """
        branch_outputs = [branch(x).squeeze(-1) for branch in self.branches]
        combined = torch.cat(branch_outputs, dim=1)
        return self.classifier(combined)


@REGISTRY.register
class TurningPointPredictor(BasePredictor):
    """Multi-Scale CNN for detecting price turning points.

    Attributes:
        name: ``"turning_point"``.
        score_threshold: Probability threshold tuned on validation set.
        device: Torch device string.
    """

    name: str = "turning_point"

    def __init__(self) -> None:
        super().__init__()
        self.score_threshold: float = self.config.get("score_threshold", 0.5)
        self.device: str = get("system.device", "cpu")

    def train(self, train_bars: list[Bar], val_bars: list[Bar]) -> dict[str, float]:
        """Train the Multi-Scale CNN on turning-point labels.

        Args:
            train_bars: Training bars with features.
            val_bars: Validation bars for threshold tuning.

        Returns:
            Dict with ``val_auc``, ``best_epoch``, ``score_threshold``.
        """
        lookback = get("features.lookback_window", 20)
        X_train, y_train = self._prepare_sequences(train_bars, lookback)
        X_val, y_val = self._prepare_sequences(val_bars, lookback)

        logger.info(
            "Training turning-point detector: {} train, {} val sequences",
            len(X_train),
            len(X_val),
        )

        n_features = X_train.shape[1]
        kernel_sizes = self.config.get("kernel_sizes", [3, 5, 11])
        num_filters = self.config.get("num_filters", 64)
        dropout = self.config.get("dropout", 0.3)

        self.model = MultiScaleCNN(n_features, kernel_sizes, num_filters, dropout)
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.get("learning_rate", 1e-3)
        )
        criterion = nn.BCEWithLogitsLoss()
        epochs = self.config.get("epochs", 50)
        patience = self.config.get("patience", 10)
        batch_size = self.config.get("batch_size", 64)

        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size, shuffle=False)

        best_auc = 0.0
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)

            val_auc = self._evaluate(val_loader)
            logger.debug("Epoch {}: loss={:.4f}, val_auc={:.4f}", epoch, train_loss / len(X_train), val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch {}", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Tune threshold on validation set
        self._tune_threshold(val_loader, y_val)

        metrics = {
            "val_auc": best_auc,
            "best_epoch": best_epoch,
            "score_threshold": self.score_threshold,
        }
        logger.info("Turning-point training complete: {}", metrics)
        return metrics

    def predict(self, bars: list[Bar]) -> list[Bar]:
        """Predict turning-point probability for each bar.

        Args:
            bars: Bars with features populated.

        Returns:
            Bars with ``predictions["turning_point"]`` set.
        """
        lookback = get("features.lookback_window", 20)
        X, _ = self._prepare_sequences(bars, lookback)

        self.model.eval()
        self.model = self.model.to(self.device)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        # First `lookback` bars get no prediction (insufficient history)
        for i, bar in enumerate(bars):
            if i < lookback:
                bar.predictions["turning_point"] = 0.0
            else:
                prob = float(probs[i - lookback])
                bar.predictions["turning_point"] = prob
                logger.debug("Bar {} tp_prob={:.3f}", bar.timestamp.date(), prob)

        logger.info("Turning-point prediction complete for {} bars", len(bars))
        return bars

    def _prepare_sequences(
        self, bars: list[Bar], lookback: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sliding-window sequences from bar features.

        Args:
            bars: Input bars.
            lookback: Sequence length.

        Returns:
            Tuple of (X, y) where X has shape (N, features, lookback).
        """
        feature_names = sorted(bars[0].features.keys()) if bars[0].features else []
        if not feature_names:
            raise ValueError("Bars must have features computed before training")

        mat = np.array([[b.features.get(f, 0.0) for f in feature_names] for b in bars])
        labels = self._make_turning_point_labels(bars)

        sequences: list[np.ndarray] = []
        targets: list[float] = []

        for i in range(lookback, len(bars)):
            seq = mat[i - lookback : i].T  # shape: (features, lookback)
            sequences.append(seq)
            targets.append(labels[i])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        return X, y

    def _make_turning_point_labels(self, bars: list[Bar]) -> np.ndarray:
        """Create binary turning-point labels.

        A turning point is a local peak or trough within a +-2 bar window.

        Args:
            bars: List of bars.

        Returns:
            Binary numpy array of length len(bars).
        """
        closes = np.array([b.close for b in bars])
        labels = np.zeros(len(closes), dtype=np.float32)
        window = 2

        for i in range(window, len(closes) - window):
            local = closes[i - window : i + window + 1]
            if closes[i] == local.max() or closes[i] == local.min():
                labels[i] = 1.0

        return labels

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create a PyTorch DataLoader.

        Args:
            X: Feature array.
            y: Label array.
            batch_size: Batch size.
            shuffle: Whether to shuffle.

        Returns:
            DataLoader instance.
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate(self, loader: DataLoader) -> float:
        """Compute AUC on a data loader.

        Args:
            loader: PyTorch DataLoader.

        Returns:
            ROC AUC score.
        """
        self.model.eval()
        all_probs: list[float] = []
        all_labels: list[float] = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(yb.numpy().tolist())

        if len(set(all_labels)) < 2:
            return 0.0
        return float(roc_auc_score(all_labels, all_probs))

    def _tune_threshold(self, loader: DataLoader, y_val: np.ndarray) -> None:
        """Tune the score threshold on validation data to maximise F1.

        Args:
            loader: Validation DataLoader.
            y_val: Validation labels.
        """
        self.model.eval()
        all_probs: list[float] = []

        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.model(xb).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())

        probs_arr = np.array(all_probs)
        best_f1 = 0.0
        best_t = 0.5

        for t in np.arange(0.3, 0.8, 0.01):
            preds = (probs_arr >= t).astype(int)
            tp = ((preds == 1) & (y_val == 1)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            fn = ((preds == 0) & (y_val == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self.score_threshold = float(best_t)
        logger.info("Tuned turning-point threshold: {:.2f} (F1={:.3f})", best_t, best_f1)
