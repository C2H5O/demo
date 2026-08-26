import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.preprocessing.common import ProcessedFrame, SequenceWriter, contiguous_runs
from datasets.preprocessing.geometry import contain_depth_valid_aware
from scripts.preprocess_stereomis import split_stereo_frame
from scripts.validate_preprocessed_datasets import clip_count, validate_sequence


def _write_sequence(root: Path, count: int, *, evaluation_only: bool = False) -> Path:
    output = root / "Demo" / "sequence"
    output.parent.mkdir(parents=True, exist_ok=True)
    with SequenceWriter(output, dataset_name="Demo", sequence_id="sequence", overwrite=False, dry_run=False, evaluation_only=evaluation_only) as writer:
        for index in range(count):
            rgb = Image.new("RGB", (100, 80), color=(index % 255, 2, 3))
            writer.write_rgb(ProcessedFrame(Path(f"source/frame{index:04d}.png"), index), index, rgb)
            if evaluation_only:
                depth = np.zeros((80, 100), dtype=np.float32)
                depth[10:20, 10:20] = 23.0
                writer.write_depth_mm(index, depth)
        writer.complete({"output_depth_unit": "mm", "invalid_depth_value": 0} if evaluation_only else {})
    return output


def test_rgb_shapes_and_mapping_are_canonical(tmp_path: Path) -> None:
    sequence = _write_sequence(tmp_path, 16)
    errors, warnings = validate_sequence(sequence)
    assert errors == []
    assert warnings == []
    metadata = json.loads((sequence / "metadata.json").read_text())
    assert metadata["frames"][0]["source_frame_id"] == 0
    assert metadata["frames"][0]["teacher_rgb_file"] == "teacher_rgb/000000.png"
    with Image.open(sequence / "teacher_rgb" / "000000.png") as image:
        assert image.mode == "RGB" and image.size == (1280, 1024)
    with Image.open(sequence / "student_rgb" / "000000.png") as image:
        assert image.mode == "RGB" and image.size == (560, 448)


def test_natural_order_and_source_gaps_split_runs(tmp_path: Path) -> None:
    files = [tmp_path / name for name in ("frame10.png", "frame2.png", "frame3.png", "frame7.png")]
    runs = contiguous_runs(files)
    assert [[path.name for path in run] for run in runs] == [["frame2.png", "frame3.png"], ["frame7.png"], ["frame10.png"]]


@pytest.mark.parametrize("frames,expected", [(15, 0), (16, 1), (17, 2), (100, 85)])
def test_stride_one_clip_counts(frames: int, expected: int) -> None:
    assert clip_count(frames) == expected


def test_explicit_stereo_splits_are_not_guessed() -> None:
    image = Image.new("RGB", (8, 2))
    left, right = split_stereo_frame(image, "side-by-side")
    assert left.size == right.size == (4, 2)
    vertical_left, vertical_right = split_stereo_frame(Image.new("RGB", (2, 8)), "top-bottom")
    assert vertical_left.size == vertical_right.size == (2, 4)
    with pytest.raises(ValueError):
        split_stereo_frame(image, "unknown")


def test_hamlyn_depth_matches_student_grid_and_preserves_invalids(tmp_path: Path) -> None:
    sequence = _write_sequence(tmp_path, 2, evaluation_only=True)
    errors, _ = validate_sequence(sequence)
    assert errors == []
    depth = np.load(sequence / "data" / "depth" / "000000.npy")
    assert depth.shape == (448, 560)
    assert depth[0, 0] == 0
    assert depth.max() == 23.0
    broken = sequence / "data" / "depth" / "000001.npy"
    broken.unlink()
    errors, _ = validate_sequence(sequence)
    assert any("RGB/GT IDs differ" in error for error in errors)


def test_depth_contain_is_not_bilinear_invalid_mixing() -> None:
    source = np.array([[0.0, 10.0], [0.0, 0.0]], dtype=np.float32)
    output = contain_depth_valid_aware(source, (560, 448))
    assert set(np.unique(output)).issubset({0.0, 10.0})


def test_complete_marker_is_only_written_on_complete_and_output_is_resumable(tmp_path: Path) -> None:
    output = tmp_path / "Demo" / "incomplete"
    output.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        with SequenceWriter(output, dataset_name="Demo", sequence_id="incomplete", overwrite=False, dry_run=False) as writer:
            writer.write_rgb(ProcessedFrame(Path("source.png"), 0), 0, Image.new("RGB", (4, 4)))
            raise RuntimeError("simulated interruption")
    assert not (output / "_preprocess_complete.json").exists()
    assert list(output.parent.glob("incomplete.partial-*.failed"))
    complete = _write_sequence(tmp_path, 1)
    with pytest.raises(FileExistsError):
        with SequenceWriter(complete, dataset_name="Demo", sequence_id="sequence", overwrite=False, dry_run=False):
            pass
