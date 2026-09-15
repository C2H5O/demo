from pathlib import Path

import numpy as np
from PIL import Image

from endodaveval.data import discover_sequences, matched_frame_ids


def _sequence(root: Path, dataset_id: int) -> None:
    data = root / f"dataset_{dataset_id}" / "keyframe_0" / "data"
    (data / "left").mkdir(parents=True)
    (data / "depth").mkdir()
    for identifier in (2, 10):
        Image.new("RGB", (6, 4)).save(data / "left" / f"frame_{identifier:06d}.png")
        np.save(data / "depth" / f"depth_{identifier:06d}.npy", np.ones((4, 6)))


def test_discovers_datasets_8_9_and_matches_numeric_frame_ids(tmp_path: Path) -> None:
    _sequence(tmp_path, 8)
    _sequence(tmp_path, 9)
    records, skipped = discover_sequences(tmp_path, [8, 9], ["left"], "data/depth")
    assert not skipped
    assert [record.dataset_id for record in records] == [8, 9]
    assert all(matched_frame_ids(record) == (2, 10) for record in records)
    assert all(record.ground_truth_directory.name == "depth" for record in records)

\n