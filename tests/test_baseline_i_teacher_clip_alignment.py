from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from cache.teacher_clip_alignment import (
    ALIGNMENT_METHOD,
    TeacherClipAlignmentIndex,
    audit_teacher_clip_alignment,
    build_teacher_clip_alignment,
    compute_overlap_alignment,
    write_teacher_clip_alignment,
)
from datasets.direct_teacher_distillation_dataset import apply_teacher_geometry_scale
from losses.direct_teacher_distillation_loss import DirectTeacherDistillationLoss
from trainers.direct_teacher_distillation_trainer import _check_resume_contract
from utils.checkpoint import DIRECT_TEACHER_DISTILLATION_PROTOCOL
from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _depth_for_ids(frame_ids, gauge, height=3, width=4):
    base = np.stack(
        [np.full((height, width), 2.0 + 0.01 * frame_id) for frame_id in frame_ids]
    ).astype(np.float32)
    return base / np.float32(gauge)


def _write_minimal_raw_cache(root, sequence_id, clip_start, frame_ids, gauge):
    path = root / sequence_id / "start_{:06d}_len_016_stride_01.npz".format(clip_start)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = _depth_for_ids(frame_ids, gauge)
    np.savez(
        path,
        sequence_id=np.asarray(sequence_id),
        clip_start=np.asarray(clip_start),
        absolute_frame_ids=np.asarray(frame_ids, dtype=np.int64),
        depth=depth,
        valid_mask=np.ones_like(depth, dtype=np.bool_),
        cache_stage=np.asarray("raw"),
        alignment_scale=np.asarray(1.0, dtype=np.float32),
    )
    return path


def test_known_overlap_scale_is_recovered_and_removes_error():
    frame_ids = list(range(8, 16))
    previous = _depth_for_ids(frame_ids, gauge=1.0)
    current = previous / np.float32(1.05)
    result = compute_overlap_alignment(
        previous,
        np.ones_like(previous, dtype=np.bool_),
        frame_ids,
        current,
        np.ones_like(current, dtype=np.bool_),
        frame_ids,
        expected_overlap_frames=8,
        minimum_valid_pixels_per_frame=1,
    )
    assert result["relative_scale_to_previous"] == pytest.approx(1.05, rel=1e-6)
    assert result["aligned_overlap_absrel"] < 1e-6


def test_first_clip_is_unit_sequence_anchor(tmp_path):
    _write_minimal_raw_cache(tmp_path, "sequence_a", 0, range(0, 16), 1.0)
    _write_minimal_raw_cache(tmp_path, "sequence_a", 8, range(8, 24), 1.05)
    metadata = build_teacher_clip_alignment(
        tmp_path, minimum_valid_pixels_per_frame=1
    )
    anchor = metadata["sequences"]["sequence_a"]["clips"]["0"]
    assert anchor["alignment_scale"] == 1.0
    assert anchor["relative_scale_to_previous"] is None
    assert metadata["sequences"]["sequence_a"]["anchor_clip_start"] == 0


def test_three_clip_chain_accumulates_in_current_to_previous_direction(tmp_path):
    r1, r2 = 1.05, 0.90
    _write_minimal_raw_cache(tmp_path, "sequence_a", 0, range(0, 16), 1.0)
    _write_minimal_raw_cache(tmp_path, "sequence_a", 8, range(8, 24), r1)
    _write_minimal_raw_cache(tmp_path, "sequence_a", 16, range(16, 32), r1 * r2)
    metadata = build_teacher_clip_alignment(
        tmp_path, minimum_valid_pixels_per_frame=1
    )
    clips = metadata["sequences"]["sequence_a"]["clips"]
    assert clips["8"]["relative_scale_to_previous"] == pytest.approx(r1, rel=1e-6)
    assert clips["16"]["relative_scale_to_previous"] == pytest.approx(r2, rel=1e-6)
    assert clips["16"]["alignment_scale"] == pytest.approx(r1 * r2, rel=1e-6)
    assert clips["16"]["previous_clip_start"] == 8
    assert clips["16"]["overlap_frame_ids"] == list(range(16, 24))


def test_loader_geometry_scaling_changes_only_metric_geometry():
    depth = torch.arange(1, 17, dtype=torch.float32).view(16, 1, 1)
    extrinsics = torch.eye(4).view(1, 4, 4).repeat(16, 1, 1)[..., :3, :]
    extrinsics[..., :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    intrinsics = torch.eye(3).view(1, 3, 3).repeat(16, 1, 1)
    confidence = torch.linspace(0.1, 1.0, 16).view(16, 1, 1)
    valid = torch.ones_like(depth, dtype=torch.bool)
    rotation_before = extrinsics[..., :3, :3].clone()
    teacher = {
        "depth": depth,
        "xyz_local": torch.ones(16, 1, 1, 3),
        "xyz_global": torch.full((16, 1, 1, 3), 2.0),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "confidence": confidence,
        "valid_mask": valid,
    }
    aligned = apply_teacher_geometry_scale(teacher, 1.25)
    torch.testing.assert_close(aligned["depth"], depth * 1.25)
    torch.testing.assert_close(aligned["xyz_local"], teacher["xyz_local"] * 1.25)
    torch.testing.assert_close(aligned["xyz_global"], teacher["xyz_global"] * 1.25)
    torch.testing.assert_close(aligned["extrinsics"][..., :3, 3], extrinsics[..., :3, 3] * 1.25)
    torch.testing.assert_close(aligned["extrinsics"][..., :3, :3], rotation_before)
    assert aligned["intrinsics"] is intrinsics
    assert aligned["confidence"] is confidence
    assert aligned["valid_mask"] is valid


def test_baseline_i_has_no_student_affine_disparity_path():
    baseline_e = load_config(ROOT / "configs/baselines/E.yaml")
    baseline_i = load_config(ROOT / "configs/baselines/I.yaml")
    assert baseline_i["loss"] == baseline_e["loss"]
    assert baseline_i["teacher_clip_alignment"]["enabled"] is True
    assert baseline_i["teacher_clip_alignment"]["method"] == ALIGNMENT_METHOD
    loss_source = (ROOT / "losses/direct_teacher_distillation_loss.py").read_text(
        encoding="utf-8"
    )
    assert "clip_shared_affine_disparity" not in loss_source
    assert "compute_clip_shared_affine_disparity_distillation_loss" not in loss_source
    DirectTeacherDistillationLoss(baseline_i["loss"])


def test_baseline_i_inherits_all_baseline_e_training_behavior():
    baseline_e = load_config(ROOT / "configs/baselines/E.yaml")
    baseline_i = load_config(ROOT / "configs/baselines/I.yaml")
    expected = deepcopy(baseline_e)
    expected["experiment"] = baseline_i["experiment"]
    expected["teacher_clip_alignment"] = baseline_i["teacher_clip_alignment"]
    expected["training"].update(
        output_dir=baseline_i["training"]["output_dir"],
        resume=baseline_i["training"]["resume"],
    )
    expected["vda_evaluation"].update(
        checkpoint=baseline_i["vda_evaluation"]["checkpoint"],
        output=baseline_i["vda_evaluation"]["output"],
    )
    expected["visualization"].update(
        output_dir=baseline_i["visualization"]["output_dir"]
    )
    assert baseline_i == expected


def test_alignment_metadata_is_auditable_and_required_fields_are_preserved(tmp_path):
    cache_root = tmp_path / "raw" / "train"
    _write_minimal_raw_cache(cache_root, "sequence_a", 0, range(0, 16), 1.0)
    _write_minimal_raw_cache(cache_root, "sequence_a", 8, range(8, 24), 1.05)
    metadata = build_teacher_clip_alignment(
        cache_root, minimum_valid_pixels_per_frame=1
    )
    metadata_path = write_teacher_clip_alignment(metadata, tmp_path / "alignment.json")
    rebuilt = audit_teacher_clip_alignment(cache_root, metadata_path)
    index = TeacherClipAlignmentIndex.from_json(metadata_path)
    record = index.lookup("sequence_a", 8, range(8, 24))
    required = {
        "clip_start",
        "absolute_frame_ids",
        "relative_scale_to_previous",
        "alignment_scale",
        "previous_clip_start",
        "overlap_frame_ids",
        "valid_overlap_pixel_count",
        "raw_overlap_absrel",
        "aligned_overlap_absrel",
        "per_frame_scales",
    }
    assert required <= set(record)
    assert rebuilt["macro_summary"]["pair_count"] == 1
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["sequences"]["sequence_a"]["clips"]["8"]["cache_relative_path"] = "wrong.npz"
    write_teacher_clip_alignment(tampered, metadata_path)
    with pytest.raises(RuntimeError, match="cache_relative_path"):
        audit_teacher_clip_alignment(cache_root, metadata_path)


def test_baseline_i_cannot_resume_baseline_e_target_gauge():
    baseline_e = load_config(ROOT / "configs/baselines/E.yaml")
    baseline_i = load_config(ROOT / "configs/baselines/I.yaml")
    checkpoint = {
        "objective_protocol": DIRECT_TEACHER_DISTILLATION_PROTOCOL,
        "config": baseline_e,
    }
    with pytest.raises(ValueError, match="Teacher clip alignment settings differ"):
        _check_resume_contract(checkpoint, baseline_i, object())
