"""
Preset save/load for user-defined configuration presets.

Presets are stored as JSON files in ``config/user_presets/``.
Built-in presets (from :mod:`config.presets`) cannot be overwritten.

Usage::

    from config.preset_io import save_user_preset, load_user_preset, list_user_presets

    save_user_preset("my_experiment", config_dict)
    cfg = load_user_preset("my_experiment")
"""

__all__ = [
    "save_user_preset",
    "load_user_preset",
    "delete_user_preset",
    "list_user_presets",
]

import json
from pathlib import Path
from typing import Any

from config.presets import PRESETS

_USER_PRESETS_DIR = Path("config/user_presets")


def save_user_preset(name: str, config: dict[str, Any]) -> Path:
    """Save a config dict as a user preset.

    Args:
        name: Preset name (used as filename stem). Must not collide
            with built-in preset IDs.
        config: Full config dict to save.

    Returns:
        Path to the saved JSON file.

    Raises:
        ValueError: If name collides with a built-in preset.
    """
    if name in PRESETS:
        raise ValueError(
            f"Cannot overwrite built-in preset '{name}'. "
            "Choose a different name."
        )

    _USER_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    path = _USER_PRESETS_DIR / f"{name}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    return path


def load_user_preset(name: str) -> dict[str, Any]:
    """Load a user preset from disk.

    Args:
        name: Preset name (filename stem).

    Returns:
        The config dict.

    Raises:
        FileNotFoundError: If the preset doesn't exist.
    """
    path = _USER_PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"User preset '{name}' not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_user_preset(name: str) -> bool:
    """Delete a user preset file.

    Args:
        name: Preset name.

    Returns:
        True if deleted, False if not found.
    """
    path = _USER_PRESETS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_user_presets() -> list[dict[str, str]]:
    """List all user-defined presets.

    Returns:
        List of {id, name, path} dicts.
    """
    if not _USER_PRESETS_DIR.exists():
        return []

    result = []
    for p in sorted(_USER_PRESETS_DIR.glob("*.json")):
        result.append({
            "id": p.stem,
            "name": p.stem,
            "path": str(p),
        })
    return result
