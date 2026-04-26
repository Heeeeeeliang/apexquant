"""
Signals proxy — path-based access to model predictions with per-layer aggregation.

Wraps ``bar.predictions`` (keyed by internal output_label) and exposes them
via folder-path keys and per-layer aggregated attributes.

Usage in Strategy Editor::

    signals = bar.signals

    # Access specific model prediction by folder path
    vol = signals["layer1/volatility/lightgbm_v3"]   # float 0-1

    # Access aggregated layer signals (mean of all models in that layer)
    layer1 = signals.layer1   # float 0-1
    layer2 = signals.layer2   # float 0-1
    layer3 = signals.layer3   # float 0-1

    # See all available predictions
    print(signals.all)        # dict of {path: float}

    # Safe access with default
    val = signals.get("layer1/volatility/lightgbm_v3", 0.5)
"""

__all__ = ["SignalsProxy"]

from typing import Any

from loguru import logger


class SignalsProxy:
    """Proxy that maps folder-path keys to model prediction probabilities.

    Constructed once per bar by the backtest engine.  Provides
    ``__getitem__`` for path-based lookup and ``layer1``/``layer2``/
    ``layer3`` properties for per-layer aggregation.

    Attributes:
        _by_path: Internal mapping from folder path to probability.
    """

    def __init__(
        self,
        predictions: dict[str, Any],
        label_to_path: dict[str, str],
    ) -> None:
        """Build the path-to-probability mapping.

        Args:
            predictions: ``bar.predictions`` dict keyed by output_label.
                Values may be floats or PredictionResult objects.
            label_to_path: Mapping from output_label to folder path
                relative to ``models/``, e.g.
                ``{"vol_prob": "layer1/volatility/lightgbm_v3"}``.
        """
        self._by_path: dict[str, float] = {}

        for label, value in predictions.items():
            prob = self._extract_prob(value)
            path = label_to_path.get(label, label)
            self._by_path[path] = prob

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, path: str) -> float:
        """Lookup a prediction by folder path.

        Args:
            path: Folder path relative to ``models/``, e.g.
                ``"layer1/volatility/lightgbm_v3"``.

        Returns:
            Probability float in ``[0, 1]``.

        Raises:
            KeyError: If no prediction exists for *path*.
        """
        if path not in self._by_path:
            raise KeyError(
                f"No prediction for '{path}'. "
                f"Available: {list(self._by_path.keys())}"
            )
        return self._by_path[path]

    def __contains__(self, path: str) -> bool:
        return path in self._by_path

    def get(self, path: str, default: float = 0.5) -> float:
        """Safe lookup with a default value.

        Args:
            path: Folder path relative to ``models/``.
            default: Returned when *path* is not present.

        Returns:
            Probability float, or *default*.
        """
        return self._by_path.get(path, default)

    # ------------------------------------------------------------------
    # Per-layer aggregation
    # ------------------------------------------------------------------

    def _layer_mean(self, layer: str) -> float:
        """Average all predictions whose path starts with *layer*.

        Args:
            layer: Layer prefix, e.g. ``"layer1"``.

        Returns:
            Mean probability, or ``0.5`` if no models matched.
        """
        vals = [
            v for k, v in self._by_path.items()
            if k.startswith(f"{layer}/")
        ]
        return float(sum(vals) / len(vals)) if vals else 0.5

    @property
    def layer1(self) -> float:
        """Mean probability across all Layer 1 (volatility) models."""
        return self._layer_mean("layer1")

    @property
    def layer2(self) -> float:
        """Mean probability across all Layer 2 (turning-point) models."""
        return self._layer_mean("layer2")

    @property
    def layer3(self) -> float:
        """Mean probability across all Layer 3 (meta-label) models."""
        return self._layer_mean("layer3")

    @property
    def all(self) -> dict[str, float]:
        """All predictions as ``{path: probability}``."""
        return dict(self._by_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prob(value: Any) -> float:
        """Extract a float probability from a prediction value.

        Handles raw floats, ints, and PredictionResult objects.

        Args:
            value: Prediction value from ``bar.predictions``.

        Returns:
            Float probability.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "prob"):
            return float(value.prob)
        return 0.5

    def __repr__(self) -> str:
        lines = [f"  {k}: {v:.4f}" for k, v in sorted(self._by_path.items())]
        return "SignalsProxy(\n" + "\n".join(lines) + "\n)"

    def __len__(self) -> int:
        return len(self._by_path)
