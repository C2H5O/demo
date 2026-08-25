from pathlib import Path

from evaluation.evaluate_crossclip_projection import select_protocol
from losses.crossclip_projection_loss import CrossClipProjectionLossConfig
from models.student.dune_fast3r_head import DuneFast3RHeadConfig
from utils.config import load_config
from visualization.crossclip_projection import _adaptive_range


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "crossclip_teacher_projection.yaml"


def test_crossclip_config_encodes_fixed_method_contract() -> None:
    config = load_config(CONFIG_PATH)
    dataset = config["dataset"]
    assert dataset["clip_length"] == 16
    assert dataset["sample_stride"] == 1
    assert dataset["window_stride"] == 1
    assert dataset["random_clip_sampling"] is True
    assert dataset["teacher_neighbor_offset"] == 1
    assert "ground_truth" not in dataset

    teacher = config["teacher"]
    assert teacher["variant"] == "base"
    assert teacher["cache_protocol"] == "crossclip_local_v1"
    assert teacher["cache_dtype"] == "float32"
    assert teacher["raw_cache_root"] != teacher["aligned_cache_root"]
    assert teacher["inference_batch_size"] == 4
    assert teacher["amp"] is True
    assert teacher["amp_dtype"] == "bfloat16"
    assert teacher["cache_compressed"] is True
    assert teacher["cache_write_workers"] == 2
    assert config["teacher_dataloader"] == {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    student = DuneFast3RHeadConfig(**config["student"])
    student.validate()
    assert tuple(student.encoder_layers) == (2, 5, 8, 11)
    assert student.freeze_encoder is False
    assert student.use_fast3r_decoder is False
    assert student.head_initial_z_bias == 1.0
    assert student.head_initial_weight_std == 1.0e-4
    assert config["training"]["max_consecutive_zero_projection_batches"] == 5
    assert config["training"]["learning_rate"] == 1.0e-4
    assert config["training"]["encoder_learning_rate"] == 1.0e-5
    assert config["training"]["gradient_accumulation_steps"] == 8

    loss = CrossClipProjectionLossConfig(**config["loss"])
    loss.validate()
    assert loss.lambda_projection == 1.0
    assert loss.lambda_highlight == 0.01
    assert loss.lambda_smooth == 0.1
    assert loss.projection_ignore_highlight is False
    assert set(config["loss"]) == {
        "mode",
        "lambda_projection",
        "lambda_highlight",
        "lambda_smooth",
        "projection_eps",
        "projection_ignore_highlight",
        "use_confidence_weight",
    }


def test_vda_is_default_and_endo3r_is_retained() -> None:
    config = load_config(CONFIG_PATH)
    assert select_protocol(config) == "vda"
    assert select_protocol(config, "endo3r") == "endo3r"
    assert config["vda_evaluation"]["protocol"] == "vda"
    assert config["endo3r_evaluation"]["protocol"] == "endo3r"


def test_adaptive_visualization_range_uses_only_valid_depth() -> None:
    import numpy as np

    depth = np.asarray([[-100.0, 1.0, 2.0, 3.0, 100.0]], dtype=np.float32)
    valid = np.asarray([[False, True, True, True, False]])
    low, high = _adaptive_range(depth, valid, (0.0, 100.0))
    assert low == 1.0
    assert high == 3.0


def test_coordinate_document_covers_pose_mapping_and_no_student_alignment() -> None:
    text = (ROOT / "docs" / "coordinate_conventions.md").read_text(encoding="utf-8")
    assert "X_camera = R @ X_world + t" in text
    assert "C_(t-1)[1:16]" in text
    assert "C_(t+1)[0:15]" in text
    assert "No per-batch or per-sample alignment" in text
