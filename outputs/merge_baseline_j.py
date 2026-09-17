"""Merge a Baseline-J DA3-Small training checkpoint into an inference model.

Run from the project root:

    python outputs/merge_baseline_j.py

The source DA3-Small safetensors and Baseline-J checkpoint are read-only.  The
only persistent output is ``outputs/baseline_J/ours.pt`` next to ``last.pt``.
"""

from __future__ import annotations

import copy
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


def _find_project_root(script_path: Path) -> Path:
    """Find the repository root even when this script is copied one level deeper."""
    searched = []
    for candidate in (script_path.parent, *script_path.parents):
        candidate = candidate.resolve()
        if candidate in searched:
            continue
        searched.append(candidate)
        if (
            (candidate / "models" / "student" / "da3_small_student.py").is_file()
            and (candidate / "configs" / "baselines" / "J.yaml").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Cannot locate the vggtoda3 project root from {}. Searched:\n{}".format(
            script_path,
            "\n".join("  - {}".format(candidate) for candidate in searched),
        )
    )


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
# Tests or advanced callers may override this module variable.  Normal CLI use
# writes next to the discovered Baseline-J last.pt as baseline_J/ours.pt.
OUTPUT_PATH: Path | None = None
J_CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines" / "J.yaml"
BASELINE_J_CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "baseline-J" / "outputs" / "baseline_J" / "last.pt",
    PROJECT_ROOT / "outputs" / "baseline_J" / "last.pt",
    PROJECT_ROOT.parent / "baseline-J" / "outputs" / "baseline_J" / "last.pt",
)
LOCAL_BASE_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "da3-small" / "model.safetensors"
LOCAL_BASE_CONFIG_PATH = PROJECT_ROOT / "checkpoints" / "da3-small" / "config.json"

# Make ``python outputs/merge_baseline_j.py`` work without an editable install.
sys.path.insert(0, str(PROJECT_ROOT))
vendored_da3 = PROJECT_ROOT / "external" / "Depth-Anything-3" / "src"
if vendored_da3.is_dir():
    sys.path.insert(0, str(vendored_da3))

import torch
import torch.nn.functional as F

from models.student.da3_small_student import DA3SmallStudent
from utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(size: int) -> str:
    return "{:,} bytes ({:.2f} MiB)".format(size, size / (1024.0 * 1024.0))


def _existing_path(candidates: Iterable[Path], description: str) -> Path:
    checked = []
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        checked.append(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "{} was not found. Checked:\n{}".format(
            description, "\n".join("  - {}".format(path) for path in checked)
        )
    )


def _find_baseline_j_checkpoint() -> Path:
    env_value = os.environ.get("BASELINE_J_CHECKPOINT")
    candidates = ([Path(env_value)] if env_value else []) + list(
        BASELINE_J_CHECKPOINT_CANDIDATES
    )
    return _existing_path(candidates, "Baseline-J checkpoint")


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(
            str(path), map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _describe_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    print("Baseline-J checkpoint structure:")
    for key in sorted(checkpoint):
        value = checkpoint[key]
        if isinstance(value, Mapping):
            tensor_count = sum(torch.is_tensor(item) for item in value.values())
            print(
                "  {}: {} ({} entries, {} direct tensors)".format(
                    key, type(value).__name__, len(value), tensor_count
                )
            )
        else:
            print("  {}: {}".format(key, type(value).__name__))


def _extract_model_state(checkpoint: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], str]:
    for container_key in ("model", "state_dict", "student"):
        value = checkpoint.get(container_key)
        if isinstance(value, Mapping) and value and all(
            isinstance(key, str) and torch.is_tensor(tensor)
            for key, tensor in value.items()
        ):
            return dict(value), container_key
    if checkpoint and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in checkpoint.items()
    ):
        return dict(checkpoint), "<root state_dict>"
    raise ValueError(
        "Baseline-J checkpoint has no tensor state_dict under model/state_dict/student"
    )


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return dict(state)
    if not all(key.startswith(prefix) for key in state):
        return {}
    return {key[len(prefix) :]: value for key, value in state.items()}


def _normalize_state_keys(
    state: Mapping[str, torch.Tensor], expected_keys: Iterable[str]
) -> Tuple[Dict[str, torch.Tensor], str]:
    expected = set(expected_keys)
    prefixes = (
        "",
        "module.",
        "model.",
        "student.",
        "module.model.",
        "module.student.",
        "model.student.",
        "student.model.",
    )
    best: Tuple[int, str, Dict[str, torch.Tensor], set[str], set[str]] | None = None
    for prefix in prefixes:
        candidate = _strip_prefix(state, prefix)
        if not candidate:
            continue
        actual = set(candidate)
        missing = expected - actual
        unexpected = actual - expected
        score = len(missing) + len(unexpected)
        if best is None or score < best[0]:
            best = (score, prefix, candidate, missing, unexpected)
        if not missing and not unexpected:
            return candidate, prefix or "<none>"
    assert best is not None
    _, prefix, _, missing, unexpected = best
    raise RuntimeError(
        "Checkpoint key mapping is not exact after best prefix {!r}. "
        "missing_keys={} unexpected_keys={}".format(
            prefix or "<none>", sorted(missing), sorted(unexpected)
        )
    )


def _path_candidates(value: Any, local_fallback: Path) -> list[Path]:
    result = []
    if value:
        supplied = Path(str(value)).expanduser()
        result.append(supplied if supplied.is_absolute() else PROJECT_ROOT / supplied)
    result.append(local_fallback)
    return result


def _resolved_student_config(
    checkpoint: Mapping[str, Any], project_config: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Path, Path]:
    checkpoint_config = checkpoint.get("config")
    checkpoint_student = (
        checkpoint_config.get("student")
        if isinstance(checkpoint_config, Mapping)
        else None
    )
    if not isinstance(checkpoint_student, Mapping):
        raise ValueError("Baseline-J checkpoint config.student is missing")
    student = copy.deepcopy(dict(checkpoint_student))
    project_student = project_config.get("student", {})
    base_path = _existing_path(
        _path_candidates(student.get("checkpoint"), LOCAL_BASE_MODEL_PATH)
        + _path_candidates(project_student.get("checkpoint"), LOCAL_BASE_MODEL_PATH),
        "original DA3-Small checkpoint",
    )
    base_config_path = _existing_path(
        _path_candidates(student.get("config_path"), LOCAL_BASE_CONFIG_PATH)
        + _path_candidates(project_student.get("config_path"), LOCAL_BASE_CONFIG_PATH),
        "original DA3-Small config.json",
    )
    student["checkpoint"] = str(base_path)
    student["config_path"] = str(base_config_path)
    return student, base_path, base_config_path


def _tensor_difference(left: torch.Tensor, right: torch.Tensor) -> Tuple[float, float, int]:
    if left.shape != right.shape:
        raise RuntimeError("Tensor shape mismatch: {} != {}".format(left.shape, right.shape))
    difference = (left.float() - right.float()).abs()
    return float(difference.max().item()), float(difference.sum().item()), difference.numel()


def _copy_and_merge_state(
    trained_model: DA3SmallStudent,
    trained_state: Mapping[str, torch.Tensor],
    merged_model: DA3SmallStudent,
) -> Tuple[Dict[str, torch.Tensor], list[str], float, float]:
    merged_state = merged_model.state_dict()
    target_prefixes = [
        "network.backbone.{}".format(name) for name in trained_model.lora_modules
    ]

    # All ordinary student tensors, including trained heads, copy one-to-one.
    for key, value in trained_state.items():
        if any(key == prefix or key.startswith(prefix + ".") for prefix in target_prefixes):
            continue
        if key not in merged_state:
            raise RuntimeError("Unmapped non-LoRA student tensor: {}".format(key))
        if merged_state[key].shape != value.shape:
            raise RuntimeError(
                "Shape mismatch for {}: checkpoint={} merged_model={}".format(
                    key, tuple(value.shape), tuple(merged_state[key].shape)
                )
            )
        merged_state[key] = value.detach().cpu()

    merged_names = []
    sanity_max = 0.0
    sanity_sum = 0.0
    sanity_count = 0
    generator = torch.Generator(device="cpu").manual_seed(20260917)
    for name, module in trained_model.lora_modules.items():
        source_prefix = "network.backbone.{}".format(name)
        destination_prefix = source_prefix
        required = {
            "base_weight": source_prefix + ".base_layer.weight",
            "lora_A": source_prefix + ".lora_A",
            "lora_B": source_prefix + ".lora_B",
        }
        missing = [key for key in required.values() if key not in trained_state]
        if missing:
            raise RuntimeError("LoRA layer {} is missing {}".format(name, missing))
        base_weight = trained_state[required["base_weight"]].detach().cpu()
        lora_a = trained_state[required["lora_A"]].detach().cpu()
        lora_b = trained_state[required["lora_B"]].detach().cpu()
        if tuple(lora_a.shape) != (module.rank, module.base_layer.in_features):
            raise RuntimeError("Unexpected lora_A layout at {}: {}".format(name, lora_a.shape))
        if tuple(lora_b.shape) != (module.base_layer.out_features, module.rank):
            raise RuntimeError("Unexpected lora_B layout at {}: {}".format(name, lora_b.shape))
        delta = torch.matmul(lora_b.float(), lora_a.float()) * float(module.scaling)
        merged_weight = (base_weight.float() + delta).to(base_weight.dtype)
        weight_key = destination_prefix + ".weight"
        if weight_key not in merged_state or merged_state[weight_key].shape != merged_weight.shape:
            raise RuntimeError("Merged destination weight mismatch at {}".format(weight_key))
        merged_state[weight_key] = merged_weight

        source_bias_key = source_prefix + ".base_layer.bias"
        destination_bias_key = destination_prefix + ".bias"
        bias = trained_state.get(source_bias_key)
        if bias is not None:
            if destination_bias_key not in merged_state:
                raise RuntimeError("Merged destination bias missing at {}".format(destination_bias_key))
            merged_state[destination_bias_key] = bias.detach().cpu()
        elif destination_bias_key in merged_state:
            raise RuntimeError("Checkpoint omits required base bias at {}".format(source_bias_key))

        # Eval-mode LoRA dropout is the identity.  Check the exact implemented
        # orientation B @ A against the merged plain nn.Linear layer.
        inputs = torch.randn(
            (3, module.base_layer.in_features), generator=generator, dtype=torch.float32
        )
        bias_float = None if bias is None else bias.detach().cpu().float()
        before = F.linear(inputs, base_weight.float(), bias_float)
        before = before + float(module.scaling) * F.linear(
            F.linear(inputs, lora_a.float()), lora_b.float()
        )
        after = F.linear(inputs, merged_weight.float(), bias_float)
        layer_max, layer_sum, layer_count = _tensor_difference(before, after)
        sanity_max = max(sanity_max, layer_max)
        sanity_sum += layer_sum
        sanity_count += layer_count
        merged_names.append(name)

    return (
        merged_state,
        merged_names,
        sanity_max,
        sanity_sum / max(sanity_count, 1),
    )


def _prefix_exact_copy_audit(
    source: Mapping[str, torch.Tensor],
    destination: Mapping[str, torch.Tensor],
    prefix: str,
) -> Tuple[int, float]:
    keys = [key for key in source if key.startswith(prefix)]
    if not keys:
        raise RuntimeError("No checkpoint tensors found for {}".format(prefix))
    maximum = 0.0
    for key in keys:
        if key not in destination:
            raise RuntimeError("Restored state is missing {}".format(key))
        left, right = source[key], destination[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError("Restored tensor metadata differs for {}".format(key))
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            diff = (left.detach().cpu().float() - right.detach().cpu().float()).abs().max()
            maximum = max(maximum, float(diff.item()))
    return len(keys), maximum


def _portable_path(path: Path) -> str:
    try:
        return "./" + path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    checkpoint_path = _find_baseline_j_checkpoint()
    output_path = OUTPUT_PATH or checkpoint_path.with_name("ours.pt")
    output_path = output_path.resolve()
    if output_path == checkpoint_path.resolve():
        raise RuntimeError("Output path must not overwrite the Baseline-J checkpoint")

    project_config = load_config(J_CONFIG_PATH)
    checkpoint_sha_before = _sha256(checkpoint_path)
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Baseline-J checkpoint root must be a mapping")
    _describe_checkpoint(checkpoint)
    if checkpoint.get("objective_protocol") != "direct_teacher_distillation_v1":
        raise ValueError(
            "Checkpoint objective_protocol is not direct_teacher_distillation_v1"
        )
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Baseline-J checkpoint config is missing")
    experiment_config = checkpoint_config.get("experiment", {})
    teacher_source_config = checkpoint_config.get("teacher", {})
    if experiment_config.get("baseline_id") != "J":
        raise ValueError(
            "Checkpoint is not Baseline J: baseline_id={!r}".format(
                experiment_config.get("baseline_id")
            )
        )
    if (
        teacher_source_config.get("mode") != "full_online"
        or bool(teacher_source_config.get("use_cache", True))
    ):
        raise ValueError("Baseline-J checkpoint must use the cache-free full_online Teacher")
    raw_state, state_container = _extract_model_state(checkpoint)
    student_config, base_path, base_config_path = _resolved_student_config(
        checkpoint, project_config
    )
    if output_path == base_path.resolve():
        raise RuntimeError("Output path must not overwrite the original DA3-Small checkpoint")

    print("Base DA3-Small checkpoint: {}".format(base_path))
    print("Base DA3-Small config: {}".format(base_config_path))
    print("Baseline-J checkpoint: {}".format(checkpoint_path))
    print("Output checkpoint: {}".format(output_path))
    print("Checkpoint model state container: {}".format(state_container))
    base_sha_before = _sha256(base_path)

    trained_model = DA3SmallStudent(student_config, device=torch.device("cpu"))
    normalized_state, stripped_prefix = _normalize_state_keys(
        raw_state, trained_model.state_dict().keys()
    )
    print("Checkpoint key prefix removed: {}".format(stripped_prefix))
    trained_model.load_state_dict(normalized_state, strict=True)
    trained_model.eval()
    print("missing_keys: []")
    print("unexpected_keys: []")

    if not trained_model.lora_modules:
        raise RuntimeError("Baseline-J student has no injected LoRA modules")
    scalings = sorted({float(module.scaling) for module in trained_model.lora_modules.values()})
    formulas = sorted(
        {(float(module.alpha), int(module.rank), float(module.scaling))
         for module in trained_model.lora_modules.values()}
    )
    print("LoRA implementation: custom models.student.lora.LoRALinear")
    print("LoRA forward: base_layer(x) + scaling * B(A(dropout(x)))")
    print("LoRA scaling: {} (alpha, rank, scaling={})".format(formulas, scalings))
    print("LoRA merge formula: W_merged = W + scaling * (B @ A)")

    trainable_names = [name for name, parameter in trained_model.named_parameters() if parameter.requires_grad]
    absent_trainable = [name for name in trainable_names if name not in normalized_state]
    if absent_trainable:
        raise RuntimeError("Trainable student parameters absent from checkpoint: {}".format(absent_trainable))
    categories = {
        "LoRA": [name for name in trainable_names if ".lora_" in name],
        "depth_head": [name for name in trainable_names if name.startswith("network.head.")],
        "camera_decoder": [name for name in trainable_names if name.startswith("network.cam_dec.")],
    }
    categorized = {name for values in categories.values() for name in values}
    categories["other"] = [name for name in trainable_names if name not in categorized]
    print(
        "Restored trainable student modules: LoRA={} tensors, depth_head={} tensors, "
        "camera_decoder={} tensors, other={} tensors".format(
            len(categories["LoRA"]), len(categories["depth_head"]),
            len(categories["camera_decoder"]), len(categories["other"]),
        )
    )

    merged_config = copy.deepcopy(student_config)
    merged_config.update(
        {
            "use_backbone_lora": False,
            "freeze_backbone": True,
            "freeze_depth_head": True,
            "freeze_camera_encoder": True,
            "freeze_camera_decoder": True,
        }
    )
    merged_model = DA3SmallStudent(merged_config, device=torch.device("cpu"))
    merged_model.eval()
    if merged_model.lora_modules:
        raise RuntimeError("Merged inference model unexpectedly contains LoRA branches")
    merged_state, merged_names, layer_max, layer_mean = _copy_and_merge_state(
        trained_model, normalized_state, merged_model
    )

    expected_final_keys = set(merged_model.state_dict())
    actual_final_keys = set(merged_state)
    final_missing = sorted(expected_final_keys - actual_final_keys)
    final_unexpected = sorted(actual_final_keys - expected_final_keys)
    print("missing_keys: {}".format(final_missing))
    print("unexpected_keys: {}".format(final_unexpected))
    if final_missing or final_unexpected:
        raise RuntimeError("Merged inference state has unresolved keys")
    merged_model.load_state_dict(merged_state, strict=True)

    depth_count, depth_diff = _prefix_exact_copy_audit(
        normalized_state, merged_state, "network.head."
    )
    camera_count, camera_diff = _prefix_exact_copy_audit(
        normalized_state, merged_state, "network.cam_dec."
    )
    if depth_diff != 0.0 or camera_diff != 0.0:
        raise RuntimeError("A trained inference head was not copied exactly")

    print("Number of merged LoRA layers: {}".format(len(merged_names)))
    print("Merged LoRA layers:")
    for name in merged_names:
        print("  - {}".format(name))
    print("Depth head restored exactly: True ({} state tensors)".format(depth_count))
    print("Camera decoder restored exactly: True ({} state tensors)".format(camera_count))
    print("LoRA layer merge max_abs_diff: {:.9g}".format(layer_max))
    print("LoRA layer merge mean_abs_diff: {:.9g}".format(layer_mean))
    layer_tolerance = (
        5.0e-5
        if all(tensor.dtype == torch.float32 for tensor in normalized_state.values())
        else 5.0e-3
    )
    print("LoRA layer merge tolerance: {:.9g}".format(layer_tolerance))
    if layer_max > layer_tolerance:
        raise RuntimeError(
            "LoRA merge numerical check failed: max_abs_diff={} > {}".format(
                layer_max, layer_tolerance
            )
        )
    print(
        "Forward equivalence test: not run; no real input was loaded and the "
        "per-layer eval-mode LoRA equivalence check was used instead."
    )

    teacher_config = checkpoint_config.get("teacher", {})
    cache_protocol = teacher_config.get("cache_protocol")
    if not isinstance(cache_protocol, str) or not cache_protocol:
        raise ValueError(
            "Checkpoint config.teacher.cache_protocol is required by the current inference loader"
        )
    merged_config_for_save = copy.deepcopy(merged_config)
    merged_config_for_save["checkpoint"] = _portable_path(base_path)
    merged_config_for_save["config_path"] = _portable_path(base_config_path)
    output_checkpoint = {
        "objective_protocol": checkpoint.get("objective_protocol"),
        "model": {key: value.detach().cpu() for key, value in merged_state.items()},
        "config": {
            "student": merged_config_for_save,
            # Kept only because the current evaluation loader validates this
            # provenance field before strict-loading the student state.
            "teacher": {"cache_protocol": cache_protocol},
        },
        "merge_metadata": {
            "format": "merged_da3_small_student_v1",
            "source_checkpoint_sha256": checkpoint_sha_before,
            "base_checkpoint_sha256": base_sha_before,
            "lora_implementation": "models.student.lora.LoRALinear",
            "lora_formula": "W_merged = W + scaling * (B @ A)",
            "lora_scaling": scalings,
            "merged_lora_layers": merged_names,
            "depth_head_restored": True,
            "camera_decoder_restored": True,
        },
    }
    forbidden = {"optimizer", "scheduler", "scaler", "cuda_rng_state_all", "teacher_q", "teacher_k"}
    leaked = forbidden.intersection(output_checkpoint)
    if leaked:
        raise RuntimeError("Training-only fields leaked into output: {}".format(sorted(leaked)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.pt")
    torch.save(output_checkpoint, temporary)
    temporary.replace(output_path)

    verification = _load_torch_checkpoint(output_path)
    if set(verification) != {"objective_protocol", "model", "config", "merge_metadata"}:
        raise RuntimeError("Saved inference checkpoint has an unexpected top-level format")
    verify_state = verification["model"]
    verify_missing = sorted(expected_final_keys - set(verify_state))
    verify_unexpected = sorted(set(verify_state) - expected_final_keys)
    if verify_missing or verify_unexpected:
        raise RuntimeError(
            "Saved checkpoint verification failed: missing={} unexpected={}".format(
                verify_missing, verify_unexpected
            )
        )

    base_sha_after = _sha256(base_path)
    checkpoint_sha_after = _sha256(checkpoint_path)
    base_unchanged = base_sha_before == base_sha_after
    checkpoint_unchanged = checkpoint_sha_before == checkpoint_sha_after
    print("Base checkpoint SHA256 before: {}".format(base_sha_before))
    print("Base checkpoint SHA256 after : {}".format(base_sha_after))
    print("Base checkpoint unchanged: {}".format(base_unchanged))
    print("Baseline-J checkpoint unchanged: {}".format(checkpoint_unchanged))
    print("ours.pt file size: {}".format(_format_bytes(output_path.stat().st_size)))
    if not base_unchanged or not checkpoint_unchanged:
        raise RuntimeError("A source checkpoint changed during merge")
    print("Merge completed successfully.")
    try:
        displayed_output = output_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        displayed_output = str(output_path)
    print("Saved merged student to: {}".format(displayed_output))


if __name__ == "__main__":
    main()
