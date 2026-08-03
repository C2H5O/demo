"""JSON-lines metric logging without a framework dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, values: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, ensure_ascii=False) + "\n")
