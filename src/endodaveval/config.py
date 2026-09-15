"""Configuration parsing and the fixed cross-method paper contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union


PathLike = Union[str, Path]
EVALUATION_RESOLUTION_HW = (256, 320)
INTERNAL_MODEL_RESOLUTION_HW = (224, 280)
SCARED_MIN_DEPTH = 0.001
SCARED_MAX_DEPTH = 100.0


class ConfigError(ValueError):
    pass


def project_path(value: PathLike, config: Mapping[str, Any]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = Path(str(config["_project_root"])) / path
    return path.resolve()


def validate_contract(config: Mapping[str, Any]) -> Tuple[int, int]:
    dataset = config["dataset"]
    evaluation = config["evaluation"]
    if set(int(value) for value in dataset.get("dataset_ids", [])) != {8, 9}:
        raise ConfigError("Paper evaluation requires exactly SCARED datasets 8 and 9")
    if str(dataset.get("ground_truth_directory", "")) != "data/depth":
        raise ConfigError("Paper evaluation requires SCARED GT at data/depth")
    if float(dataset.get("ground_truth_scale", 0.0)) != 0.001:
        raise ConfigError("SCARED ground_truth_scale must be 0.001 mm-to-m")
    shape = (int(evaluation.get("height", 0)), int(evaluation.get("width", 0)))
    if shape != EVALUATION_RESOLUTION_HW:
        raise ConfigError("evaluation resolution must be 256x320 HxW")
    if float(evaluation.get("min_depth", 0.0)) != SCARED_MIN_DEPTH:
        raise ConfigError("min_depth must be 0.001 m")
    if float(evaluation.get("max_depth", 0.0)) != SCARED_MAX_DEPTH:
        raise ConfigError("max_depth must be 100.0 m")
    if not dataset.get("frame_sources"):
        raise ConfigError("dataset.frame_sources must not be empty")
    return shape


def load_config(path: PathLike) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError("Configuration file does not exist: {}".format(config_path)) from error
    except json.JSONDecodeError as error:
        raise ConfigError("Invalid JSON in {}: {}".format(config_path, error)) from error
    if not isinstance(value, dict):
        raise ConfigError("Configuration root must be an object")
    for section in ("dataset", "endodav", "evaluation"):
        if not isinstance(value.get(section), dict):
            raise ConfigError("Missing configuration object: {}".format(section))
    if "output_root" not in value:
        raise ConfigError("Missing configuration value: output_root")
    value["_config_path"] = str(config_path)
    value["_project_root"] = str(config_path.parent.parent.resolve())
    validate_contract(value)
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

\n