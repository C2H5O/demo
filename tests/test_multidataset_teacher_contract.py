from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from cache.generate_crossclip_teacher_cache import canonicalize_teacher_outputs
from datasets.crossclip_teacher_dataset import (
    crossclip_teacher_cache_path,
    make_crossclip_rgb_dataset,
)
from datasets.multidataset import (
    CanonicalTemporalRGBDataset,
    discover_canonical_sequences,
    discover_processed_scared_sequences,
)
from datasets.precomputed_highlight import highlight_manifest_payload, parse_highlight_options
from datasets.transforms import (
    load_precomputed_highlight_mask_tensor,
    load_precomputed_student_rgb_tensor,
    load_rgb_tensor,
    load_teacher_rgb_tensor,
    tensor_from_numpy_buffer,
)
from precompute_highlights import _initialize_worker, _process_frame


def test_precomputed_student_decode_skips_resize_and_teacher_is_strict(tmp_path, monkeypatch) -> None:
    student = tmp_path / "student.png"
    teacher = tmp_path / "teacher.png"
    legacy = tmp_path / "legacy.png"
    Image.new("RGB", (560, 448), color=(4, 5, 6)).save(student)
    Image.new("RGB", (1280, 1024), color=(7, 8, 9)).save(teacher)
    Image.new("RGB", (80, 64), color=(1, 2, 3)).save(legacy)

    # Legacy SCARED keeps its deterministic runtime resize behaviour.
    assert load_rgb_tensor(legacy, 448, 560, normalize_mode="minus_one_one").shape == (3, 448, 560)

    monkeypatch.setattr("datasets.transforms._resize_image", lambda *args: (_ for _ in ()).throw(AssertionError("resize")))
    student_tensor = load_precomputed_student_rgb_tensor(student)
    assert student_tensor.shape == (3, 448, 560)
    assert torch.allclose(
        student_tensor[:, 0, 0],
        (torch.tensor([4.0, 5.0, 6.0]) / 255.0) * 2.0 - 1.0,
    )
    assert load_teacher_rgb_tensor(teacher).shape == (3, 1024, 1280)
    with pytest.raises(RuntimeError, match="requires"):
        load_teacher_rgb_tensor(student)


def test_numpy_buffer_conversion_preserves_values_without_from_numpy() -> None:
    floats = __import__("numpy").arange(12, dtype="float32").reshape(3, 4)
    boolean = floats > 5
    assert torch.equal(tensor_from_numpy_buffer(floats), torch.arange(12).float().reshape(3, 4))
    assert torch.equal(tensor_from_numpy_buffer(boolean), torch.tensor(boolean.tolist()))


def test_native_teacher_canonicalization_scales_k_and_preserves_z_depth() -> None:
    shape = (1, 1, 1024, 1280)
    depth = torch.full(shape, 2.0)
    local = torch.zeros(1, 1, 1024, 1280, 3)
    local[..., 2] = 2.0
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    intrinsics[..., 0, 0] = 1000.0
    intrinsics[..., 1, 1] = 800.0
    intrinsics[..., 0, 2] = 600.0
    intrinsics[..., 1, 2] = 500.0
    result = canonicalize_teacher_outputs({
        "depth": depth,
        "xyz_local": local,
        "xyz_global": local.clone(),
        "conf_local": torch.ones(shape),
        "valid_mask": torch.ones(shape, dtype=torch.bool),
        "intrinsics": intrinsics,
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4)[..., :3, :],
    })
    assert result["depth"].shape == (1, 1, 448, 560)
    assert result["xyz_local"].shape == (1, 1, 448, 560, 3)
    assert result["valid_mask"].dtype is torch.bool
    assert torch.equal(result["depth"], result["xyz_local"][..., 2])
    assert result["intrinsics"][0, 0, 0, 0].item() == pytest.approx(437.5)
    assert result["intrinsics"][0, 0, 1, 1].item() == pytest.approx(350.0)
    assert result["intrinsics"][0, 0, 0, 2].item() == pytest.approx(262.5)
    assert result["extrinsics"].shape == (1, 1, 3, 4)


def _write_processed_sequence(root, dataset_name: str, *, evaluation_only: bool) -> None:
    sequence = root / dataset_name / "case_a"
    (sequence / "student_rgb").mkdir(parents=True)
    (sequence / "teacher_rgb").mkdir()
    frames = []
    for index in range(16):
        name = "{:06d}.png".format(index)
        (sequence / "student_rgb" / name).touch()
        (sequence / "teacher_rgb" / name).touch()
        frames.append({"processed_index": index, "source_frame_id": 100 + index, "student_rgb_file": "student_rgb/" + name, "teacher_rgb_file": "teacher_rgb/" + name})
    (sequence / "metadata.json").write_text(json.dumps({"sequence_id": "case_a", "evaluation_only": evaluation_only, "frames": frames}), encoding="utf-8")
    (sequence / "_preprocess_complete.json").write_text("{}", encoding="utf-8")


def test_hamlyn_is_evaluation_only_and_cache_identity_is_dataset_safe(tmp_path) -> None:
    _write_processed_sequence(tmp_path, "C3VD", evaluation_only=False)
    _write_processed_sequence(tmp_path, "Hamlyn", evaluation_only=True)
    train = discover_canonical_sequences(tmp_path, "train")
    test = discover_canonical_sequences(tmp_path, "test")
    assert [item["dataset_name"] for item in train] == ["C3VD"]
    assert [item["dataset_name"] for item in test] == ["Hamlyn"]
    common = {"dataset_id": 1, "keyframe_id": "k", "sequence_id": "same", "clip_start": 0}
    assert crossclip_teacher_cache_path(tmp_path, {**common, "dataset_name": "C3VD"}) != crossclip_teacher_cache_path(tmp_path, {**common, "dataset_name": "StereoMIS"})


def test_processed_scared_uses_student_rgb_and_retains_teacher_rgb(tmp_path) -> None:
    sequence = tmp_path / "dataset_01" / "keyframe_1"
    (sequence / "student_rgb").mkdir(parents=True)
    (sequence / "teacher_rgb").mkdir()
    frames = []
    for index in range(32):
        name = "{:06d}.png".format(index)
        (sequence / "student_rgb" / name).touch()
        (sequence / "teacher_rgb" / name).touch()
        frames.append({
            "processed_index": index,
            "source_frame_id": 100 + index,
            "student_rgb_file": "student_rgb/" + name,
            "teacher_rgb_file": "teacher_rgb/" + name,
        })
    (sequence / "metadata.json").write_text(
        json.dumps({"sequence_id": "scared_case", "frames": frames}),
        encoding="utf-8",
    )
    (sequence / "_preprocess_complete.json").write_text("{}", encoding="utf-8")

    discovered = discover_processed_scared_sequences(tmp_path, "train")
    assert len(discovered) == 1
    assert Path(discovered[0]["frame_paths"][0]).parent.name == "student_rgb"
    assert Path(discovered[0]["teacher_frame_paths"][0]).parent.name == "teacher_rgb"
    assert discovered[0]["sequence_id"] == "dataset_1/keyframe_1"
    assert discovered[0]["source_sequence_id"] == "scared_case"
    dataset = make_crossclip_rgb_dataset({
        "root": str(tmp_path),
        "frame_source": "auto",
        "clip_length": 16,
        "sample_stride": 1,
        "window_stride": 8,
        "drop_incomplete_clip": True,
        "image_height": 448,
        "image_width": 560,
        "resize_mode": "resize",
        "normalize_mode": "zero_one",
    }, "train")
    assert isinstance(dataset, CanonicalTemporalRGBDataset)
    assert [record.clip_start for record in dataset.clips] == [0, 8, 16]
    assert dataset.sequences[0]["absolute_frame_ids"] == list(range(100, 132))
    assert crossclip_teacher_cache_path(
        tmp_path / "train", {
            "dataset_name": "SCARED",
            "dataset_id": 1,
            "keyframe_id": "keyframe_1",
            "sequence_id": "dataset_1/keyframe_1",
            "clip_start": 0,
        },
    ) == (
        tmp_path / "train" / "dataset_01" / "keyframe_1"
        / "dataset_1_keyframe_1" / "start_000000_len_016_stride_01.npz"
    )


def test_canonical_dataset_reads_precomputed_highlight_without_online_processor(tmp_path) -> None:
    sequence_root = tmp_path / "dataset_01" / "keyframe_1"
    student_root = sequence_root / "student_rgb"
    teacher_root = sequence_root / "teacher_rgb"
    mask_root = sequence_root / "student_highlight_mask"
    inpainted_root = sequence_root / "student_inpainted_rgb"
    for directory in (student_root, teacher_root, mask_root, inpainted_root):
        directory.mkdir(parents=True, exist_ok=True)
    frame_paths, teacher_paths = [], []
    for index in range(16):
        name = "{:06d}.png".format(index)
        Image.new("RGB", (560, 448), color=(1, 2, 3)).save(student_root / name)
        Image.new("RGB", (1280, 1024), color=(1, 2, 3)).save(teacher_root / name)
        Image.new("L", (560, 448), color=255 if index == 0 else 0).save(mask_root / name)
        Image.new("RGB", (560, 448), color=(10, 20, 30)).save(inpainted_root / name)
        frame_paths.append(str(student_root / name))
        teacher_paths.append(str(teacher_root / name))
    highlight = {
        "enabled": True,
        "storage": "precomputed",
        "mask_directory_name": mask_root.name,
        "inpainted_directory_name": inpainted_root.name,
    }
    _, detection, mask_name, inpainted_name = parse_highlight_options(highlight)
    (sequence_root / "_highlight_precompute_complete.json").write_text(
        json.dumps(
            highlight_manifest_payload(detection, 16, mask_name, inpainted_name)
        ),
        encoding="utf-8",
    )
    sequence = {
        "dataset_name": "SCARED", "dataset_id": 1, "keyframe_id": "keyframe_1",
        "sequence_id": "dataset_1/keyframe_1", "sequence_length": 16,
        "frame_paths": frame_paths, "teacher_frame_paths": teacher_paths,
        "absolute_frame_ids": list(range(16)), "frame_directory": str(student_root),
        "keyframe_directory": str(sequence_root), "depth_directory": None,
    }
    dataset = CanonicalTemporalRGBDataset(
        [sequence], clip_length=16, sample_stride=1, window_stride=8,
        normalize_mode="zero_one", highlight=highlight,
    )
    sample = dataset[0]
    assert sample["highlight_masks"].shape == (16, 1, 448, 560)
    assert sample["highlight_masks"][0].all()
    assert not sample["highlight_masks"][1:].any()
    torch.testing.assert_close(
        sample["inpainted_images"][0, :, 0, 0],
        torch.tensor([10.0, 20.0, 30.0]) / 255.0,
    )


def test_offline_highlight_worker_materializes_loadable_pngs(tmp_path) -> None:
    pytest.importorskip("cv2")
    source = tmp_path / "student_rgb" / "000000.png"
    mask = tmp_path / "student_highlight_mask" / source.name
    inpainted = tmp_path / "student_inpainted_rgb" / source.name
    source.parent.mkdir()
    Image.new("RGB", (560, 448), color=(10, 20, 30)).save(source)
    _, detection, _, _ = parse_highlight_options({"enabled": True})
    _initialize_worker(
        {name: getattr(detection, name) for name in detection.__dataclass_fields__}
    )
    assert _process_frame((str(source), str(mask), str(inpainted))) == str(source)
    loaded_mask = load_precomputed_highlight_mask_tensor(mask)
    loaded_rgb = load_precomputed_student_rgb_tensor(inpainted, "zero_one")
    assert not loaded_mask.any()
    torch.testing.assert_close(
        loaded_rgb[:, 0, 0], torch.tensor([10.0, 20.0, 30.0]) / 255.0
    )
