"""Runtime configuration with environment-variable path overrides."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from evaluation.hamlyn.constants import HAMLYN_SEQUENCE_IDS
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/hamlyn_eval.yaml"


def _path(value: str, project_root: Path) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(value)))
    return (candidate if candidate.is_absolute() else project_root / candidate).resolve()


def _python(value: str, project_root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    candidate = Path(expanded)
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = _path(expanded, project_root)
    else:
        located = shutil.which(expanded)
        if located is None:
            raise FileNotFoundError("Python executable not found: {}".format(value))
        resolved = Path(located).resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Python executable is not a file: {}".format(resolved))
    return resolved


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    hamlyn_root: Path
    ours_checkpoint: Path
    da3_checkpoint_dir: Path
    endodav_repository: Path
    endodav_checkpoint: Path
    endodav_pretrained_path: Path
    endo3r_repository: Path
    endo3r_checkpoint: Path
    endo3r_dust3r_checkpoint: Path
    output_root: Path
    ours_python: Path
    endodav_python: Path
    endo3r_python: Path
    device: str


PATH_ENV = {
    "hamlyn_root": "HAMLYN_ROOT",
    "ours_checkpoint": "OURS_CHECKPOINT",
    "da3_checkpoint_dir": "DA3_CHECKPOINT_DIR",
    "endodav_repository": "ENDODAV_REPOSITORY",
    "endodav_checkpoint": "ENDODAV_CHECKPOINT",
    "endodav_pretrained_path": "ENDODAV_PRETRAINED_PATH",
    "endo3r_repository": "ENDO3R_REPOSITORY",
    "endo3r_checkpoint": "ENDO3R_CHECKPOINT",
    "endo3r_dust3r_checkpoint": "ENDO3R_DUST3R_CHECKPOINT",
    "output_root": "HAMLYN_OUTPUT_ROOT",
}


def load_runtime_config(
    config_path: Path = DEFAULT_CONFIG,
    environment: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    environment = os.environ if environment is None else environment
    config = load_config(config_path)
    project_root = PROJECT_ROOT
    configured_ids = tuple(int(value) for value in config["dataset"]["sequence_ids"])
    if configured_ids != HAMLYN_SEQUENCE_IDS:
        raise ValueError("Config sequence_ids do not match the formal 22-sequence protocol")
    values = {"hamlyn_root": str(config["dataset"]["root"]), **config["paths"]}
    resolved = {
        key: _path(environment.get(variable, str(values[key])), project_root)
        for key, variable in PATH_ENV.items()
    }
    ours_python_value = environment.get("OURS_PYTHON", "python")
    endodav_python_value = environment.get("ENDODAV_PYTHON", ours_python_value)
    endo3r_python_value = environment.get("ENDO3R_PYTHON", "python")
    return RuntimeConfig(
        project_root=project_root,
        **resolved,
        ours_python=_python(ours_python_value, project_root),
        endodav_python=_python(endodav_python_value, project_root),
        endo3r_python=_python(endo3r_python_value, project_root),
        device=environment.get("HAMLYN_DEVICE", "cuda:0"),
    )
