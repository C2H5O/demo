"""Collect the four common-evaluator results into JSON and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.hamlyn.cache import atomic_write_json
from evaluation.hamlyn.constants import (
    EVALUATION_RESOLUTION_HW,
    HAMLYN_SEQUENCE_IDS,
    METHOD_ORDER,
)


def collect_summary(output_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    reference_frames = None
    for method in METHOD_ORDER:
        path = output_root / method / "evaluation.json"
        if not path.is_file():
            raise FileNotFoundError("Method evaluation is missing: {}".format(path))
        value = json.loads(path.read_text(encoding="utf-8"))
        if tuple(value.get("sequence_ids", ())) != HAMLYN_SEQUENCE_IDS:
            raise RuntimeError("{} does not contain the exact 22-sequence split".format(path))
        if tuple(value.get("evaluation_resolution_hw", ())) != EVALUATION_RESOLUTION_HW:
            raise RuntimeError("{} does not use the common 224x280 grid".format(path))
        frames = {
            int(item["sequence_id"]): tuple(item["frame_ids"])
            for item in value.get("per_sequence", [])
        }
        if set(frames) != set(HAMLYN_SEQUENCE_IDS):
            raise RuntimeError("{} has incomplete per-sequence frame metadata".format(path))
        if reference_frames is None:
            reference_frames = frames
        elif frames != reference_frames:
            raise RuntimeError("Method frame IDs differ; refusing to collect mixed results")
        inference = value["inference_resolution_hw"]
        evaluation = value["evaluation_resolution_hw"]
        metrics = value["metrics"]
        rows.append(
            {
                "Method": value["method"],
                "Inference H": int(inference[0]),
                "Inference W": int(inference[1]),
                "Eval H": int(evaluation[0]),
                "Eval W": int(evaluation[1]),
                "AbsRel": float(metrics["abs_relative_difference"]),
                "RMSE": float(metrics["rmse_linear"]),
                "delta1": float(metrics["delta1_acc"]),
            }
        )
    result = {
        "dataset": "Hamlyn",
        "sequence_ids": list(HAMLYN_SEQUENCE_IDS),
        "sequence_count": len(HAMLYN_SEQUENCE_IDS),
        "rows": rows,
    }
    atomic_write_json(output_root / "summary.json", result)
    csv_path = output_root / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} and {}".format(output_root / "summary.json", csv_path), flush=True)
    return result
