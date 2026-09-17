"""Baseline-H evaluation excludes TAE and reports QG efficiency fields."""

from pathlib import Path

import pytest

import evaluation.evaluate_crossclip_projection as evaluation


class _Spool:
    native_prediction_resolutions_hw = {(256, 320)}

    def __init__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def test_tae_disabled_skips_temporal_code_and_omits_tae_output(
    tmp_path: Path, monkeypatch
) -> None:
    selected = tmp_path / "last.pt"
    merged = tmp_path / "ours.pt"
    output = tmp_path / "qg.json"
    config = {
        "device": "cpu",
        "dataset": {"image_height": 256, "image_width": 320},
        "student": {"checkpoint": "base.safetensors"},
        "inference": {"acceleration": "query_group_kv"},
        "kv_sampling": {
            "enabled": True,
            "method": "query_group",
            "query_group_size": 8,
            "kv_frames": 20,
        },
        "vda_evaluation": {
            "checkpoint": str(selected),
            "output": str(output),
            "split": "test",
            "evaluation_height": 128,
            "evaluation_width": 160,
            "tae": {"enabled": False},
        },
    }
    sequence = {
        "sequence_id": "sequence",
        "dataset_id": 8,
        "frame_paths": ["frame.png"],
    }
    monkeypatch.setattr(evaluation, "load_config", lambda path: config)
    monkeypatch.setattr(
        evaluation,
        "_dataset_and_ground_truth",
        lambda *args: (object(), {"sequence": sequence}, {"sequence": object()}, []),
    )
    monkeypatch.setattr(
        evaluation,
        "ensure_merged_student_checkpoint",
        lambda checkpoint, loaded: merged,
    )
    monkeypatch.setattr(evaluation, "_evaluation_model", lambda *args: object())
    monkeypatch.setattr(evaluation, "sequence_frames", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(evaluation.vda_core, "_SequencePredictionSpool", _Spool)
    monkeypatch.setattr(
        evaluation,
        "infer_student_video",
        lambda *args, **kwargs: {
            "output_frame_count": 1,
            "window_count": 1,
            "model_input_frame_count": 32,
            "model_forward_seconds": 2.0,
            "sequence_pipeline_seconds": 2.5,
            "peak_cuda_memory_allocated_bytes": 123,
            "peak_cuda_memory_reserved_bytes": 456,
            "timing_scope": "model including QG",
        },
    )
    monkeypatch.setattr(
        evaluation.vda_core,
        "_evaluate_sequence",
        lambda *args, **kwargs: {
            "sequence_id": "sequence",
            "missing_prediction_count": 0,
            "metrics": {
                "abs_relative_difference": 0.1,
                "rmse_linear": 0.2,
                "delta1_acc": 0.9,
            },
        },
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_tae",
        lambda *args, **kwargs: pytest.fail("TAE must not run for QG-H"),
    )

    result = evaluation.evaluate_vda(tmp_path / "H.yaml")
    assert result["protocol"] == "video-depth-anything-depth-scared-v2"
    assert result["metrics"] == {
        "abs_relative_difference": 0.1,
        "rmse_linear": 0.2,
        "delta1_acc": 0.9,
    }
    assert "tae_sequence_count" not in result
    assert "complete_tae_coverage" not in result
    assert "temporal" not in result["sequences"][0]
    assert result["peak_cuda_memory_allocated"] == 123
    assert result["peak_cuda_memory_reserved"] == 456
    assert result["peak_cuda_memory_allocated_bytes"] == 123
    assert result["peak_cuda_memory_reserved_bytes"] == 456
    assert result["total_model_inference_seconds"] == 2.0
    assert result["mean_frame_inference_seconds"] == 2.0
    assert result["mean_frame_inference_ms"] == 2000.0
    assert result["inference_fps"] == 0.5
    assert result["model_input_resolution_hw"] == [256, 320]
    assert result["evaluation_resolution_hw"] == [128, 160]
