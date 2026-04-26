"""
ApexQuant Configuration Module
==============================

Provides a single ``CONFIG`` dict with all system settings and
four helper functions:

- ``get(key_path, default)`` — dot-notation read
- ``save_config(path)``      — persist to JSON
- ``load_config(path)``      — restore from JSON (deep-merged)
- ``deep_merge(base, over)`` — recursive dict merge

Override chain:  ``default.py`` → ``local.py`` → ``cloud.py``
(each optional layer is deep-merged on top of the previous one).

Usage::

    from config import get, CONFIG, save_config, load_config

    use_ai = get("strategy.use_ai")              # True
    capital = get("backtest.initial_capital")     # 100000
    save_config("results/runs/snapshot.json")
    load_config("results/runs/snapshot.json")
"""

__all__ = ["CONFIG", "get", "save_config", "load_config", "deep_merge"]

from config.default import CONFIG, deep_merge, get, load_config, save_config
from config.loader import load_config as _apply_layers

# Apply optional local.py / cloud.py override layers at import time
_apply_layers()
