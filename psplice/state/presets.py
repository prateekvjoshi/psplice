"""
Preset storage.

A preset is a named snapshot of an intervention configuration: steering
vectors, head masks, layer-skip settings, LoRA adapter path, and decode
settings.  Presets are stored as JSON files in the user data directory and
can be loaded into any running daemon session.

Tensor data (steering vectors) is NOT embedded in the preset.  Instead, the
absolute path to the original .pt file is saved.  If that file is deleted or
moved, loading the preset will fail with a clear error message.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .session import get_data_dir


def get_presets_dir() -> Path:
    d = get_data_dir() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preset_path(name: str) -> Path:
    return get_presets_dir() / f"{name}.json"


def save_preset(name: str, data: dict[str, Any]) -> None:
    """Write a preset to disk."""
    preset_path(name).write_text(json.dumps(data, indent=2))


def load_preset(name: str) -> Optional[dict[str, Any]]:
    """Load a preset from disk. Returns None if not found."""
    p = preset_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def list_presets() -> list[str]:
    """Return names of all saved presets (without the .json extension)."""
    return sorted(p.stem for p in get_presets_dir().glob("*.json"))


def delete_preset(name: str) -> bool:
    """Delete a preset. Returns True if it existed."""
    p = preset_path(name)
    if p.exists():
        p.unlink()
        return True
    return False
