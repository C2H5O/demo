from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config when PyYAML is available, otherwise JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML configs.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    data = data or {}
    base_value = data.pop("base_config", None)
    if base_value is None:
        return data
    base_path = Path(str(base_value)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    if base_path.resolve() == path.resolve():
        raise ValueError("A config cannot inherit from itself")
    return _deep_merge(load_config(base_path), data)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

