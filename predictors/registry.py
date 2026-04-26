"""
Predictor registry — singleton that maps names to live predictor instances.

New predictors call ``REGISTRY.register(instance)`` at import time; no
other code needs to change.  The registry also provides ``run_all`` to
execute every enabled predictor in parallel via :class:`ThreadPoolExecutor`.

Usage::

    from predictors.registry import REGISTRY
    from predictors.base import Predictor, Context
    from data.bar import Bar

    REGISTRY.register(my_predictor)
    pred = REGISTRY.get("vol_lgb")

    # Run all enabled predictors concurrently
    results = REGISTRY.run_all(bar, context, config)
    print(results)  # {"vol_prob": PredictionResult(...), ...}
"""

from __future__ import annotations

__all__ = ["REGISTRY", "PredictorRegistry"]

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from data.bar import Bar
    from predictors.base import Context, Predictor
    from predictors.result import PredictionResult


class PredictorRegistry:
    """Singleton registry of all available predictors.

    Stores *instances* of :class:`~predictors.base.Predictor` keyed by
    their ``name`` attribute.  Provides filtering by the
    ``config["predictors"]["enabled"]`` list and concurrent execution.

    Attributes:
        _predictors: Internal mapping from name to predictor instance.
    """

    def __init__(self) -> None:
        self._predictors: dict[str, Predictor] = {}
        self._permanently_failed: dict[str, str] = {}  # name → error message

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, predictor: Predictor | type) -> Predictor | type:
        """Register a predictor instance (or class used as decorator).

        Supports two calling conventions:

        1.  **Instance registration** (preferred for the new interface)::

                REGISTRY.register(my_predictor_instance)

        2.  **Class decorator** (legacy ``BasePredictor`` pattern)::

                @REGISTRY.register
                class VolPredictor(BasePredictor):
                    name = "volatility"
                    ...

        Args:
            predictor: A :class:`Predictor` instance or a class with a
                ``name`` class attribute.

        Returns:
            The same predictor / class, enabling decorator usage.
        """
        if isinstance(predictor, type):
            # Legacy decorator usage — store the *class*
            name: str = getattr(predictor, "name", "")
            if name in self._predictors:
                logger.warning("Overwriting registered predictor: {}", name)
            self._predictors[name] = predictor  # type: ignore[assignment]
            logger.debug("Registered predictor class: {}", name)
            return predictor

        # Instance registration (new Predictor interface)
        name = predictor.name
        if name in self._predictors:
            logger.warning("Overwriting registered predictor: {}", name)
        self._predictors[name] = predictor
        logger.info("Registered predictor instance: {}", name)
        return predictor

    def unregister(self, name: str) -> None:
        """Remove a predictor from the registry.

        Args:
            name: Predictor name to remove.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._predictors:
            available = list(self._predictors.keys())
            raise KeyError(
                f"Cannot unregister '{name}': not found. Available: {available}"
            )
        del self._predictors[name]
        logger.info("Unregistered predictor: {}", name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Predictor:
        """Retrieve a predictor by name.

        If the stored value is a **class** (legacy decorator pattern),
        it is instantiated on first access and the instance replaces
        the class in the registry.

        Args:
            name: Registered predictor name.

        Returns:
            The predictor instance.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._predictors:
            available = list(self._predictors.keys())
            raise KeyError(
                f"Predictor '{name}' not found. Available: {available}"
            )
        entry = self._predictors[name]
        # Lazy-instantiate legacy class registrations
        if isinstance(entry, type):
            entry = entry()
            self._predictors[name] = entry
        return entry

    def list_all(self) -> list[str]:
        """Return all registered predictor names.

        Returns:
            Sorted list of predictor name strings.
        """
        return sorted(self._predictors.keys())

    def list_enabled(self, config: dict[str, Any]) -> list[Predictor]:
        """Return predictor instances whose name appears in the config's
        ``predictors.enabled`` list.

        Args:
            config: Full CONFIG dict (must contain
                ``config["predictors"]["enabled"]``).

        Returns:
            List of enabled :class:`Predictor` instances, in the order
            they appear in the config list.
        """
        enabled_names: list[str] = (
            config.get("predictors", {}).get("enabled", [])
        )
        enabled: list[Predictor] = []
        for name in enabled_names:
            if name in self._predictors:
                enabled.append(self.get(name))
            else:
                logger.warning(
                    "Enabled predictor '{}' not found in registry, skipping", name
                )
        logger.info(
            "Enabled predictors: {} of {} registered",
            len(enabled),
            len(self._predictors),
        )
        return enabled

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def reload(self, models_dir: str = "models") -> int:
        """Clear the registry and re-scan ``models/`` from disk.

        This ensures deleted model folders are removed from the registry
        and new ones are added.

        Args:
            models_dir: Root directory to scan for ``meta.json`` files.

        Returns:
            Number of newly registered adapters (compared to before).
        """
        from predictors import _discover_and_register

        before = set(self.list_all())
        self._predictors.clear()
        self._permanently_failed.clear()
        _discover_and_register(models_dir)
        after = set(self.list_all())
        removed = before - after
        added = after - before
        if removed:
            logger.info("Reload removed {} stale adapter(s): {}", len(removed), removed)
        if added:
            logger.info("Reload found {} new adapter(s): {}", len(added), added)
        return len(added)

    # ------------------------------------------------------------------
    # Label-to-path mapping (for SignalsProxy)
    # ------------------------------------------------------------------

    def get_label_to_path_map(self) -> dict[str, str]:
        """Build a mapping from output_label to folder path relative to ``models/``.

        Inspects each registered predictor's ``_model_dir`` attribute
        (set by adapters) and extracts the portion after ``models/``.

        Returns:
            Dict mapping output_label to folder path, e.g.
            ``{"vol_prob": "layer1/volatility/lightgbm_v3"}``.
        """
        mapping: dict[str, str] = {}
        for name, pred in self._predictors.items():
            label = getattr(pred, "output_label", None)
            model_dir = getattr(pred, "_model_dir", None)
            if label is None or model_dir is None:
                continue
            # Extract path relative to models/
            from pathlib import Path
            parts = Path(model_dir).parts
            try:
                idx = list(parts).index("models")
                rel = "/".join(parts[idx + 1 :])
            except ValueError:
                rel = str(model_dir)
            mapping[label] = rel
        return mapping

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    # Errors that indicate the adapter can never succeed in this
    # process (e.g. missing native library) and should not be retried.
    _PERMANENT_ERROR_TYPES: tuple[type[BaseException], ...] = (
        OSError,
        ImportError,
        ModuleNotFoundError,
    )

    def run_all(
        self,
        bar: Bar,
        context: Context,
        config: dict[str, Any],
    ) -> dict[str, PredictionResult]:
        """Run all enabled predictors in parallel and collect results.

        Each predictor is dispatched to a :class:`ThreadPoolExecutor`.
        Exceptions in individual predictors are caught and logged so
        that one failing predictor does not block the others.

        Predictors that fail with an :class:`OSError`,
        :class:`ImportError`, or :class:`ModuleNotFoundError` are
        marked as *permanently failed* and skipped on all subsequent
        calls (avoids a multi-second retry penalty per bar).

        Args:
            bar: The current OHLCV bar to predict on.
            context: Ambient runtime context.
            config: Full CONFIG dict (used to resolve the enabled list).

        Returns:
            Dict mapping ``output_label`` to :class:`PredictionResult`
            for every predictor that completed successfully.
        """
        enabled = self.list_enabled(config)
        if not enabled:
            logger.warning("No enabled predictors to run")
            return {}

        # Filter out permanently-failed adapters
        active = [p for p in enabled if p.name not in self._permanently_failed]
        if len(active) < len(enabled):
            skipped = len(enabled) - len(active)
            logger.debug("Skipping {} permanently-failed predictor(s)", skipped)

        if not active:
            return {}

        results: dict[str, PredictionResult] = {}

        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            future_to_name = {
                executor.submit(self._run_single, pred, bar, context): pred.name
                for pred in active
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[result.label] = result
                    logger.debug(
                        "Predictor '{}' returned: label={}, prob={:.4f}",
                        name,
                        result.label,
                        result.prob,
                    )
                except self._PERMANENT_ERROR_TYPES as exc:
                    self._permanently_failed[name] = str(exc)
                    logger.warning(
                        "Adapter '{}' permanently disabled due to load "
                        "failure ({}): {}",
                        name,
                        type(exc).__name__,
                        exc,
                    )
                except Exception as exc:
                    logger.warning(
                        "Predictor '{}' failed ({}): {}",
                        name,
                        type(exc).__name__,
                        exc,
                    )

        logger.info(
            "run_all complete: {}/{} predictors returned results",
            len(results),
            len(active),
        )
        return results

    @staticmethod
    def _run_single(
        predictor: Predictor,
        bar: Bar,
        context: Context,
    ) -> PredictionResult:
        """Execute a single predictor's ``predict`` method.

        Args:
            predictor: The predictor instance to run.
            bar: Current bar.
            context: Runtime context.

        Returns:
            The predictor's :class:`PredictionResult`.
        """
        return predictor.predict(bar, context)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._predictors

    def __len__(self) -> int:
        return len(self._predictors)

    def __repr__(self) -> str:
        return f"PredictorRegistry({self.list_all()})"


REGISTRY = PredictorRegistry()
