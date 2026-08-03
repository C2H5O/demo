"""Portable calibration-file reader.

The source project only discovered calibration paths; it did not define a
single SCARED calibration schema. This reader therefore preserves raw mappings
and arrays without inventing coordinate conversions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_calibration(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Calibration file not found: {}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Calibration JSON must contain an object")
        return value
    if suffix in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Calibration YAML must contain a mapping")
        return value
    if suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    raise ValueError(
        "Unsupported calibration format {}. No source-project parser exists for it.".format(
            suffix
        )
    )
