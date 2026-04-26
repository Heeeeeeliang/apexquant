"""
Configuration layer loader.

Discovers optional override modules (``config.local``, ``config.cloud``)
and deep-merges them into the live CONFIG dict.

Merge order: ``default.py`` → ``local.py`` → ``cloud.py``.
Later layers override earlier ones key-by-key.

Usage::

    # Typically called once at import time by config/__init__.py
    from config.loader import load_config
    load_config()
"""

__all__ = ["load_config"]

import importlib

from loguru import logger

from config.default import CONFIG, deep_merge


def load_config() -> dict:
    """Discover and merge optional config layers into CONFIG.

    Looks for ``config.local`` and ``config.cloud`` modules (plain
    Python files exporting a ``CONFIG`` dict).  Each layer is
    deep-merged on top of the accumulated result.

    Returns:
        The module-level ``CONFIG`` dict after all merges.
    """
    for layer_name in ("local", "cloud"):
        try:
            mod = importlib.import_module(f"config.{layer_name}")
            layer = getattr(mod, "CONFIG", {})
            merged = deep_merge(CONFIG, layer)
            CONFIG.clear()
            CONFIG.update(merged)
            logger.info("Merged config layer: {}", layer_name)
        except ModuleNotFoundError:
            logger.debug("Config layer '{}' not found, skipping", layer_name)

    return CONFIG
