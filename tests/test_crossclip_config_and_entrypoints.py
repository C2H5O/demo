from pathlib import Path

from evaluation.evaluate_crossclip_projection import select_protocol
from losses.crossclip_projection_loss import CrossClipProjectionLossConfig
from models.student.da3_small_student import DA3SmallConfig
from utils.config import load_config
from visualization.crossclip_projection import _adaptive_range


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "vggtoda3.yaml"


def test_vggtoda3_config_encodes_fixed_contract() -> None:
    config = load_config(CONFIG_PATH)
    dataset = config["dataset"]
    assert (dataset["clip_length"], dataset["sample_stride"], dataset["window_stride"]) == (16, 1, 8)
    assert dataset["root"] == "/public/home/2024141520249/Documents/datasets/vggtodistilldata/processed/scared"
    assert dataset["normalize_mode"] == "zero_one"
    teacher = config["teacher"]
    assert teacher["raw_cache_root"] == "/public/home/2024141520249/Documents/Projects/vggtofast3r/data/teacher_cache_crossclip_base_raw_448x560"
    assert teacher["use_aligned_cache"] is False
    student = DA3SmallConfig(**config["student"])
    student.validate()
    assert student.use_ray is False and student.use_ray_pose is False
    assert student.use_camera_head is True
    loss = CrossClipProjectionLossConfig(**config["loss"])
    loss.validate()
    assert (loss.lambda_projection, loss.lambda_highlight, loss.lambda_smooth) == (1.0, 0.01, 0.1)


def test_vda_is_default_and_endo3r_is_retained() -> None:
    config = load_config(CONFIG_PATH)
    assert select_protocol(config) == "vda"
    assert select_protocol(config, "endo3r") == "endo3r"


def test_adaptive_visualization_range_uses_only_valid_depth() -> None:
    import numpy as np
    depth = np.asarray([[-100.0, 1.0, 2.0, 3.0, 100.0]], dtype=np.float32)
    valid = np.asarray([[False, True, True, True, False]])
    assert _adaptive_range(depth, valid, (0.0, 100.0)) == (1.0, 3.0)


def test_coordinate_document_covers_stride8_and_world_to_camera() -> None:
    text = (ROOT / "docs" / "coordinate_conventions.md").read_text(encoding="utf-8")
    assert "X_camera = R @ X_world + t" in text
    assert "C_(s-8)[8:16]" in text
    assert "C_(s+8)[0:8]" in text


def test_da3_setup_is_compatible_with_git_without_dash_c() -> None:
    script = (ROOT / "scripts" / "setup_da3.sh").read_text(encoding="utf-8")
    assert 'cd "${DA3_ROOT}"' in script
    assert "git -C" not in script
