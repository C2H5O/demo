"""Fail-fast dataset, checkpoint, external checkout, Python, and CUDA checks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from evaluation.hamlyn.cache import atomic_write_json
from evaluation.hamlyn.config import RuntimeConfig
from evaluation.hamlyn.data import SequenceRecord, print_preflight


class PreflightError(RuntimeError):
    pass


def _require_file(path: Path, message: str) -> None:
    if not path.is_file():
        raise PreflightError("{}: {}".format(message, path))


def _require_directory(path: Path, message: str) -> None:
    if not path.is_dir():
        raise PreflightError("{}: {}".format(message, path))


def _probe_python(
    executable: Path,
    working_directory: Path,
    imports: Sequence[str],
    label: str,
) -> Dict[str, Any]:
    code = (
        "import importlib,json,sys,torch;"
        + "[importlib.import_module(name) for name in " + repr(list(imports)) + "];"
        + "print('__HAMLYN_PROBE__'+json.dumps({"
        + "'python':sys.version.split()[0],'torch':torch.__version__,"
        + "'cuda_available':torch.cuda.is_available(),"
        + "'cuda_device_count':torch.cuda.device_count()}))"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(executable), "-s", "-c", code],
        cwd=str(working_directory),
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    marker = "__HAMLYN_PROBE__"
    line = next(
        (value for value in completed.stdout.splitlines() if value.startswith(marker)),
        None,
    )
    if completed.returncode != 0 or line is None:
        detail = (completed.stderr or completed.stdout).strip()
        raise PreflightError(
            "{} Python probe failed at {} (exit {}): {}".format(
                label, executable, completed.returncode, detail[-3000:]
            )
        )
    payload = json.loads(line[len(marker) :])
    if not payload["cuda_available"]:
        raise PreflightError(
            "{} Python cannot access CUDA: {}".format(label, executable)
        )
    return {"executable": str(executable), **payload}


def run_preflight(
    runtime: RuntimeConfig, records: Sequence[SequenceRecord]
) -> Mapping[str, Any]:
    _require_directory(runtime.hamlyn_root, "Hamlyn root is missing")
    _require_file(runtime.ours_checkpoint, "Ours merged checkpoint is missing")
    _require_file(
        runtime.da3_checkpoint_dir / "model.safetensors",
        "Official DA3-Small safetensors are missing",
    )
    _require_file(
        runtime.da3_checkpoint_dir / "config.json",
        "Official DA3-Small config.json is missing",
    )

    _require_directory(
        runtime.endodav_repository,
        "Official EndoDAV checkout is missing. Run: git clone "
        "https://github.com/Zanue/EndoDAV.git external/EndoDAV",
    )
    for relative in (
        "models/endodav/endodav.py",
        "models/endodav/__init__.py",
        "utils/layers.py",
        "evaluate_depth_video_hamlyn.py",
    ):
        _require_file(
            runtime.endodav_repository / relative,
            "Official EndoDAV checkout is incomplete",
        )
    _require_file(runtime.endodav_checkpoint, "EndoDAV depth_model checkpoint is missing")
    _require_file(
        runtime.endodav_pretrained_path / "video_depth_anything_vits.pth",
        "EndoDAV Video-Depth-Anything ViT-S checkpoint is missing",
    )

    _require_directory(
        runtime.endo3r_repository,
        "Official Endo3R checkout is missing. Run: git clone "
        "https://github.com/wrld/Endo3R.git external/Endo3R",
    )
    _require_file(runtime.endo3r_repository / "demo.py", "Official Endo3R demo.py is missing")
    expected_raft = (
        runtime.endo3r_repository / "checkpoints" / "raft-things.pth"
    ).resolve()
    if runtime.endo3r_raft_checkpoint != expected_raft:
        raise PreflightError(
            "Official Endo3R loads RAFT from ./checkpoints/raft-things.pth; "
            "configured path must resolve to {}: {}".format(
                expected_raft, runtime.endo3r_raft_checkpoint
            )
        )
    _require_file(runtime.endo3r_checkpoint, "Endo3R checkpoint is missing")
    _require_file(
        runtime.endo3r_raft_checkpoint,
        "Endo3R RAFT checkpoint is missing",
    )

    dataset = print_preflight(records)
    python = {
        "ours": _probe_python(
            runtime.ours_python,
            runtime.project_root,
            ("numpy", "cv2", "PIL", "yaml", "safetensors", "depth_anything_3"),
            "Ours/DA3",
        ),
        "endodav": _probe_python(
            runtime.endodav_python,
            runtime.endodav_repository,
            ("numpy", "cv2", "PIL", "yaml", "models.endodav"),
            "EndoDAV",
        ),
        "endo3r": _probe_python(
            runtime.endo3r_python,
            runtime.endo3r_repository,
            ("numpy", "cv2", "PIL", "yaml", "demo"),
            "Endo3R",
        ),
    }
    result = {
        "status": "ok",
        "dataset_root": str(runtime.hamlyn_root),
        "sequence_count": len(records),
        "sequences": dataset,
        "python": python,
        "checkpoints": {
            "ours": str(runtime.ours_checkpoint),
            "da3": str(runtime.da3_checkpoint_dir),
            "endodav": str(runtime.endodav_checkpoint),
            "endodav_pretrained": str(
                runtime.endodav_pretrained_path / "video_depth_anything_vits.pth"
            ),
            "endo3r": str(runtime.endo3r_checkpoint),
            "endo3r_raft": str(runtime.endo3r_raft_checkpoint),
        },
    }
    atomic_write_json(runtime.output_root / "preflight.json", result)
    print("[preflight] dataset, checkpoints, external repositories, Python, and CUDA are valid", flush=True)
    return result
