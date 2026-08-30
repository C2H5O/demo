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
    assert dataset["highlight"]["enabled"] is True
    assert dataset["highlight"]["storage"] == "precomputed"
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
    assert config["training"]["timing"] == {
        "enabled": True,
        "log_every_micro_batches": 1,
    }


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
    assert 'pip install --no-deps -e "${DA3_ROOT}"' in script
    assert "open3d" in script and "pip install open3d" not in script
    assert '--force-reinstall --no-cache-dir "numpy==1.26.4"' in script
    assert 'conda install --yes --force-reinstall --prefix "${CONDA_PREFIX}"' in script
    assert '"numpy=1.26.4" "numpy-base=1.26.4"' in script
    assert '"opencv-python-headless==4.10.0.84"' in script
    assert "pip install --no-deps --force-reinstall" in script
    assert "opencv-contrib-python-headless" in script
    assert "verify_numpy_abi.py" in script
    for component in ("DinoV2", "DualDPT", "CameraEnc", "CameraDec"):
        assert "import {}".format(component) in script


def test_environment_mismatch_reports_import_provenance() -> None:
    script = (ROOT / "scripts" / "verify_environment.py").read_text(encoding="utf-8")
    for field in (
        "python_executable",
        "numpy_import_file",
        "numpy_distribution_version",
        "numpy_distribution_location",
        "PYTHONPATH",
    ):
        assert field in script
    assert "verify_numpy_abi()" in script


def test_numpy_abi_probe_covers_torch_and_opencv() -> None:
    source = (ROOT / "scripts" / "verify_numpy_abi.py").read_text(encoding="utf-8")
    assert "torch.from_numpy(image)" in source
    assert "cv2.connectedComponentsWithStats(image, 8)" in source
    assert "numpy_file" in source and "cv2_file" in source


def test_da3_checkpoint_uses_sharing_aware_strict_safetensors_load() -> None:
    source = (ROOT / "models" / "student" / "da3_small_student.py").read_text(
        encoding="utf-8"
    )
    assert "from safetensors.torch import load_model" in source
    assert 'checkpoint_container.add_module("model", network)' in source
    assert "load_model(" in source and "strict=True" in source
    assert "network.load_state_dict(network_state" not in source
