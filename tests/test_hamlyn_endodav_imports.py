from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from evaluation.hamlyn.endodav import _repository_import_path


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_repository_import_path_isolates_official_top_level_packages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "EndoDAV"
    _write(repository / "models" / "backbones.py", "ORIGIN = 'official'\n")
    _write(
        repository / "models" / "endodav" / "__init__.py",
        "import models.backbones\n"
        "from utils.util import MARKER\n"
        "LOADED = (models.backbones.ORIGIN, MARKER)\n"
        "def endodav():\n"
        "    return (models.backbones.ORIGIN, MARKER)\n",
    )
    _write(repository / "utils" / "util.py", "MARKER = 'official-utils'\n")

    local_models = types.ModuleType("models")
    local_models.__path__ = []
    local_backbones = types.ModuleType("models.backbones")
    local_backbones.ORIGIN = "local"
    local_utils = types.ModuleType("utils")
    local_utils.__path__ = []
    monkeypatch.setitem(sys.modules, "models", local_models)
    monkeypatch.setitem(sys.modules, "models.backbones", local_backbones)
    monkeypatch.setitem(sys.modules, "utils", local_utils)

    with _repository_import_path(repository):
        module = importlib.import_module("models.endodav")
        assert module.LOADED == ("official", "official-utils")

    assert sys.modules["models"] is local_models
    assert sys.modules["models.backbones"] is local_backbones
    assert sys.modules["utils"] is local_utils
    assert "models.endodav" not in sys.modules
    assert "utils.util" not in sys.modules
    assert module.endodav() == ("official", "official-utils")
