import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.preprocessing.common import ProcessedFrame, SequenceWriter, contiguous_runs
from datasets.preprocessing.geometry import contain_depth_valid_aware
from scripts.preprocess_c3vd import candidate_rgb_dirs
from scripts.preprocess_endovis18 import left_frame_directories, sequence_id_for_left_frames
from scripts.preprocess_scared import output_location
from scripts.preprocess_stereomis import sequence_videos, split_stereo_frame
from scripts.validate_preprocessed_datasets import clip_count, validate_sequence
from datasets.scared_discovery import SequenceRecord


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


def test_c3vd_discovers_only_colour_frames_in_a_sequence_directory(tmp_path: Path) -> None:
    sequence = tmp_path / "cecum_t1_a"
    sequence.mkdir()
    for name in ("000001_color.png", "000002_color.png", "000001_occlusion.png"):
        Image.new("RGB", (3, 2)).save(sequence / name)
    discovered = candidate_rgb_dirs(tmp_path)
    assert discovered == [(sequence, [sequence / "000001_color.png", sequence / "000002_color.png"])]


def test_endovis_release_one_and_test_keep_distinct_sequence_identity(tmp_path: Path) -> None:
    release_one = tmp_path / "train" / "miccai_challenge_2018_release_1" / "miccai_challenge_2018_release_1" / "seq_1" / "left_frames"
    test = tmp_path / "test" / "seq_1" / "left_frames"
    for directory in (release_one, test):
        directory.mkdir(parents=True)
        Image.new("RGB", (3, 2)).save(directory / "frame000.png")
    discovered = left_frame_directories(tmp_path)
    identifiers = {sequence_id_for_left_frames(directory, tmp_path) for directory in discovered}
    assert identifiers == {
        "train_miccai_challenge_2018_release_1_miccai_challenge_2018_release_1_seq_1",
        "test_seq_1",
    }


def test_stereomis_p1_video_and_p2_video_are_discovered_per_sequence(tmp_path: Path) -> None:
    p1 = tmp_path / "P1"
    p2 = tmp_path / "P2_0"
    p1.mkdir()
    p2.mkdir()
    (p1 / "video.mp4").touch()
    (p2 / "IFBS_ENDOSCOPE-part0000.mp4").touch()
    (p2 / "groundtruth.txt").touch()
    assert sequence_videos(tmp_path) == [
        (p1, p1 / "video.mp4"),
        (p2, p2 / "IFBS_ENDOSCOPE-part0000.mp4"),
    ]


def test_scared_output_identity_is_dataset_keyframe_and_gap_safe(tmp_path: Path) -> None:
    source = tmp_path / "dataset_1" / "key_frame_3" / "data" / "left"
    record = SequenceRecord(
        dataset_id=1,
        keyframe_id="key_frame_3",
        sequence_id="dataset_1/key_frame_3",
        keyframe_directory=source.parents[2],
        frame_directory=source,
        frame_paths=(),
        calibration_path=None,
        depth_directory=None,
        disparity_directory=None,
        frame_data_directory=None,
        reprojection_directory=None,
        scene_points_directory=None,
        point_cloud_path=None,
        video_path=None,
    )
    sequence_id, destination = output_location(tmp_path / "processed", record, 1, 2)
    assert sequence_id == "dataset_01/key_frame_3_run01"
    assert destination == tmp_path / "processed" / "SCARED" / "dataset_01" / "key_frame_3_run01"


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
