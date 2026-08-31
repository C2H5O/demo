from pathlib import Path

import evaluation.evaluate_crossclip_projection as crossclip_evaluation
from evaluation.evaluate_crossclip_projection import select_protocol
from losses.direct_teacher_distillation_loss import (
    DirectTeacherDistillationLossConfig,
)
from models.student.da3_small_student import DA3SmallConfig
from utils.config import load_config
from utils.checkpoint import (
    DIRECT_TEACHER_DISTILLATION_PROTOCOL,
    require_training_objective,
)
from visualization.crossclip_projection import _adaptive_range


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "vggtoda3.yaml"


def test_vggtoda3_config_encodes_fixed_contract() -> None:
    config = load_config(CONFIG_PATH)
    dataset = config["dataset"]
    assert (dataset["clip_length"], dataset["sample_stride"], dataset["window_stride"]) == (16, 1, 8)
    assert dataset["root"] == "/public/home/2024141520249/Documents/datasets/vggtodistilldata/processed/SCARED"
    assert dataset["normalize_mode"] == "zero_one"
    assert dataset["highlight"]["enabled"] is True
    assert dataset["highlight"]["storage"] == "precomputed"
    teacher = config["teacher"]
    assert teacher["raw_cache_root"] == "/public/home/2024141520249/Documents/Projects/vggtofast3r/data/teacher_cache_crossclip_base_raw_448x560"
    assert "aligned_cache_root" not in teacher
    assert "use_aligned_cache" not in teacher
    assert "scale_alignment" not in teacher
    student = DA3SmallConfig(**config["student"])
    student.validate()
    assert student.use_ray is False and student.use_ray_pose is False
    assert student.use_camera_head is True
    loss = DirectTeacherDistillationLossConfig.from_mapping(config["loss"])
    assert (loss.lambda_depth, loss.lambda_camera) == (1.0, 0.1)
    assert (loss.lambda_highlight, loss.lambda_smooth) == (0.01, 0.1)
    assert config["experiment"]["objective_protocol"] == "direct_teacher_distillation_v1"
    assert student.freeze_camera_encoder is True
    assert config["training"]["timing"] == {
        "enabled": True,
        "log_every_micro_batches": 1,
    }
    assert config["dataloader"]["batch_size"] == 16
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["learning_rate"] == 1.0e-5
    assert config["training"]["lora_learning_rate"] == 1.0e-5
    assert config["training"]["min_learning_rate"] == 1.0e-6


def test_vda_is_default_and_endo3r_is_retained() -> None:
    config = load_config(CONFIG_PATH)
    assert select_protocol(config) == "vda"
    assert select_protocol(config, "endo3r") == "endo3r"
    raw_scared = "/public/home/2024141520249/Documents/datasets/vggtodistilldata/scared"
    assert config["vda_evaluation"]["rgb_root"] == raw_scared
    assert config["vda_evaluation"]["gt_root"] == raw_scared
    assert config["endo3r_evaluation"]["rgb_root"] == raw_scared
    assert config["endo3r_evaluation"]["gt_root"] == raw_scared


def test_evaluation_rgb_root_overrides_processed_training_root(monkeypatch) -> None:
    captured = {}

    class EmptyDataset:
        sequences = []

    def make_dataset(dataset_config, split):
        captured.update(dataset_config)
        captured["split"] = split
        return EmptyDataset()

    monkeypatch.setattr(
        crossclip_evaluation, "make_scared_rgb_dataset", make_dataset
    )
    crossclip_evaluation._dataset_and_ground_truth(
        {
            "dataset": {
                "root": "/processed",
                "legacy_scared_root": "/legacy",
                "canonical_root": "/canonical",
            }
        },
        {"rgb_root": "/raw/scared", "frame_source": "auto"},
        "test",
    )

    assert captured["root"] == "/raw/scared"
    assert captured["legacy_scared_root"] == "/raw/scared"
    assert captured["canonical_root"] is None
    assert captured["drop_incomplete_clip"] is False
    assert captured["split"] == "test"


def test_adaptive_visualization_range_uses_only_valid_depth() -> None:
    import numpy as np
    depth = np.asarray([[-100.0, 1.0, 2.0, 3.0, 100.0]], dtype=np.float32)
    valid = np.asarray([[False, True, True, True, False]])
    assert _adaptive_range(depth, valid, (0.0, 100.0)) == (1.0, 3.0)


def test_coordinate_document_covers_same_clip_w2c_relative_pose() -> None:
    text = (ROOT / "docs" / "coordinate_conventions.md").read_text(encoding="utf-8")
    assert "X_camera = R @ X_world + t" in text
    assert "C_n^S" in text and "C_n^T" in text
    assert "E_i @ inverse(E_0)" in text


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


def test_old_projection_checkpoint_cannot_resume_new_objective() -> None:
    import pytest

    with pytest.raises(ValueError, match="Start a new training run"):
        require_training_objective(
            {"config": {"loss": {"mode": "crossclip_projection_highlight_smooth"}}},
            DIRECT_TEACHER_DISTILLATION_PROTOCOL,
        )
    require_training_objective(
        {"objective_protocol": DIRECT_TEACHER_DISTILLATION_PROTOCOL},
        DIRECT_TEACHER_DISTILLATION_PROTOCOL,
    )


def test_active_training_path_has_no_projection_or_neighbor_calls() -> None:
    active_files = (
        ROOT / "datasets" / "direct_teacher_distillation_dataset.py",
        ROOT / "losses" / "direct_teacher_distillation_loss.py",
        ROOT / "trainers" / "direct_teacher_distillation_trainer.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in active_files)
    for forbidden in (
        "project_student_points_to_teacher",
        "grid_sample",
        "global_to_camera_points",
        "teacher_left",
        "teacher_right",
        "build_neighbor_clip_indices",
        "alignment_scale",
        "scale_alignment",
    ):
        assert forbidden not in source


def test_camera_encoder_is_checkpoint_only_and_decoder_executes() -> None:
    source = (ROOT / "models" / "student" / "da3_small_student.py").read_text(
        encoding="utf-8"
    )
    assert "cam_token=None" in source
    assert "self.camera_decoder(feats[-1][1])" in source
    assert "self.camera_encoder(" not in source
