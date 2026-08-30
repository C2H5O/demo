"""Official pretrained DA3-Small with a true depth-only dense forward path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student.lora import LoRALinear, inject_da3_mlp_lora
from utils.da3_geometry import depth_intrinsics_to_local_points, local_to_global_points


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DA3SmallConfig:
    architecture: str = "da3_small"
    model_name: str = "da3-small"
    image_height: int = 448
    image_width: int = 560
    patch_size: int = 14
    checkpoint: str = "./checkpoints/da3-small/model.safetensors"
    config_path: str = "./checkpoints/da3-small/config.json"
    normalize_mode: str = "zero_one"
    use_ray: bool = False
    use_ray_pose: bool = False
    use_camera_head: bool = True
    freeze_backbone: bool = False
    use_backbone_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_expected_modules: int = 24
    freeze_depth_head: bool = False
    freeze_camera_encoder: bool = False
    freeze_camera_decoder: bool = False
    head_chunk_size: int = 8
    ref_view_strategy: str = "saddle_balanced"

    def validate(self) -> None:
        if self.architecture != "da3_small" or self.model_name != "da3-small":
            raise ValueError("student must select the official da3-small architecture")
        if (self.image_height, self.image_width) != (448, 560):
            raise ValueError("DA3 student input must be exactly 448x560")
        if self.patch_size != 14 or (448 // self.patch_size, 560 // self.patch_size) != (32, 40):
            raise ValueError("DA3-Small requires the 14px patch grid 32x40")
        if self.normalize_mode != "zero_one":
            raise ValueError("Dataset RGB must be zero_one; DA3 ImageNet normalization is applied in-model")
        if self.use_ray or self.use_ray_pose:
            raise ValueError("ray and ray-pose branches are forbidden")
        if not self.use_camera_head:
            raise ValueError("DA3 native camera head is required")
        if self.use_backbone_lora:
            if not self.freeze_backbone:
                raise ValueError("Standard backbone LoRA requires freeze_backbone=true")
            if self.lora_rank <= 0 or self.lora_alpha <= 0.0:
                raise ValueError("LoRA rank and alpha must be positive")
            if not 0.0 <= self.lora_dropout < 1.0:
                raise ValueError("LoRA dropout must be in [0,1)")
            if self.lora_expected_modules <= 0:
                raise ValueError("lora_expected_modules must be positive")
        if self.head_chunk_size <= 0:
            raise ValueError("head_chunk_size must be positive")


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _require_official_da3() -> tuple[Any, Any, Any]:
    try:
        from omegaconf import OmegaConf
        from depth_anything_3.cfg import create_object
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    except ImportError as error:
        raise RuntimeError(
            "Official ByteDance-Seed Depth-Anything-3 is required. "
            "Run scripts/setup_da3.sh before training."
        ) from error
    return create_object, OmegaConf, pose_encoding_to_extri_intri


def _load_full_pretrained(config: DA3SmallConfig) -> tuple[nn.Module, Dict[str, int]]:
    """Strictly load the complete official checkpoint before disabling ray execution."""
    create_object, OmegaConf, _ = _require_official_da3()
    try:
        from safetensors import safe_open
        from safetensors.torch import load_model
    except ImportError as error:
        raise RuntimeError("safetensors is required to load DA3-Small") from error
    checkpoint_path = _project_path(config.checkpoint)
    config_path = _project_path(config.config_path)
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "DA3-Small needs checkpoint={} and config={}".format(checkpoint_path, config_path)
        )
    serialized_config = json.loads(config_path.read_text(encoding="utf-8"))
    if serialized_config.get("model_name") != "da3-small" or "config" not in serialized_config:
        raise RuntimeError("DA3 config.json does not describe da3-small")
    official_config = serialized_config["config"]
    architecture_audit = {
        "backbone": official_config.get("net", {}).get("name"),
        "head": official_config.get("head", {}).get("__object__", {}).get("name"),
        "camera_encoder": official_config.get("cam_enc", {}).get("__object__", {}).get("name"),
        "camera_decoder": official_config.get("cam_dec", {}).get("__object__", {}).get("name"),
    }
    if architecture_audit != {
        "backbone": "vits", "head": "DualDPT",
        "camera_encoder": "CameraEnc", "camera_decoder": "CameraDec",
    }:
        raise RuntimeError("Unexpected DA3-Small architecture: {}".format(architecture_audit))
    network = create_object(OmegaConf.create(official_config))
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
    if not keys or any(not key.startswith("model.") for key in keys):
        raise RuntimeError("DA3 safetensors keys must all use the official model. prefix")
    # The official DualDPT shares LayerNorm modules across auxiliary levels.
    # Safetensors intentionally stores a shared tensor once, so load_file plus
    # load_state_dict would report its alias names as missing.  Preserve the
    # checkpoint's model.* namespace and use the sharing-aware strict loader.
    checkpoint_container = nn.Module()
    checkpoint_container.add_module("model", network)
    missing, unexpected = load_model(
        checkpoint_container, str(checkpoint_path), strict=True, device="cpu"
    )
    if missing or unexpected:
        raise RuntimeError(
            "Strict DA3 checkpoint load reported missing={} unexpected={}".format(
                missing, unexpected
            )
        )
    counts = {
        "backbone": sum(key.startswith("model.backbone.") for key in keys),
        "depth_head": sum(key.startswith("model.head.") for key in keys),
        "camera_encoder": sum(key.startswith("model.cam_enc.") for key in keys),
        "camera_decoder": sum(key.startswith("model.cam_dec.") for key in keys),
    }
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError("Checkpoint audit failed: {}".format(counts))
    print("DA3-Small checkpoint path: {}".format(checkpoint_path))
    print("DA3-Small config path: {}".format(config_path))
    print("official architecture: DINOv2 ViT-S/14 + DualDPT + CameraEnc/CameraDec")
    print(
        "checkpoint strict load: backbone={} depth_head={} camera_encoder={} camera_decoder={}".format(
            counts["backbone"], counts["depth_head"], counts["camera_encoder"], counts["camera_decoder"]
        )
    )
    print("ray branch disabled: ray-specific DualDPT modules will not execute")
    return network, counts


class DA3SmallStudent(nn.Module):
    """Joint 16-view DA3-Small depth/camera student in W2C convention."""

    def __init__(
        self,
        config: Mapping[str, Any] | DA3SmallConfig,
        device: Optional[torch.device] = None,
        network: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = DA3SmallConfig(**dict(config)) if isinstance(config, Mapping) else config
        self.config.validate()
        self.network, self.load_audit = (
            _load_full_pretrained(self.config) if network is None else (network, {})
        )
        for name in ("backbone", "head", "cam_enc", "cam_dec"):
            if not isinstance(getattr(self.network, name, None), nn.Module):
                raise RuntimeError("Official DA3 network lacks {}".format(name))
        self.lora_modules: Dict[str, LoRALinear] = {}
        if self.config.use_backbone_lora:
            self.lora_modules = inject_da3_mlp_lora(
                self.backbone,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
            if len(self.lora_modules) != self.config.lora_expected_modules:
                raise RuntimeError(
                    "Expected {} DINOv2 MLP LoRA targets, found {}: {}".format(
                        self.config.lora_expected_modules,
                        len(self.lora_modules),
                        sorted(self.lora_modules),
                    )
                )
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        )
        self._ray_forward_count = 0
        self._ray_hooks = []
        self._timing_enabled = False
        self._last_forward_timing_events: Dict[str, torch.cuda.Event] = {}
        self._install_ray_execution_audit()
        self._configure_trainability()
        self.to(device or torch.device("cpu"))

    @property
    def backbone(self) -> nn.Module:
        return self.network.backbone

    @property
    def depth_head(self) -> nn.Module:
        return self.network.head

    @property
    def camera_encoder(self) -> nn.Module:
        return self.network.cam_enc

    @property
    def camera_decoder(self) -> nn.Module:
        return self.network.cam_dec

    def _ray_modules(self) -> list[nn.Module]:
        scratch = self.depth_head.scratch
        names = (
            "refinenet1_aux", "refinenet2_aux", "refinenet3_aux", "refinenet4_aux",
            "output_conv1_aux", "output_conv2_aux",
        )
        return [getattr(scratch, name) for name in names]

    def _install_ray_execution_audit(self) -> None:
        def counted(_module: nn.Module, _inputs: Any, _output: Any) -> None:
            self._ray_forward_count += 1

        audited = []
        for root in self._ray_modules():
            audited.extend(
                module for module in root.modules() if not isinstance(module, nn.ModuleList)
            )
        self._ray_hooks = [module.register_forward_hook(counted) for module in audited]

    def _configure_trainability(self) -> None:
        groups = (
            (self.depth_head, not self.config.freeze_depth_head),
            (self.camera_encoder, not self.config.freeze_camera_encoder),
            (self.camera_decoder, not self.config.freeze_camera_decoder),
        )
        for module, trainable in groups:
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.config.freeze_backbone)
        if self.config.use_backbone_lora:
            for module in self.lora_modules.values():
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
        # The checkpoint was loaded strictly first. Ray-only weights are retained
        # for checkpoint compatibility, but are frozen and never placed in an optimizer.
        for module in self._ray_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def assert_trainability_contract(self) -> None:
        if not any(parameter.requires_grad for parameter in self.depth_head.parameters()):
            raise RuntimeError("DA3 pretrained depth branch is not trainable")
        if not any(parameter.requires_grad for parameter in self.camera_decoder.parameters()):
            raise RuntimeError("DA3 native camera decoder is not trainable")
        if any(parameter.requires_grad for module in self._ray_modules() for parameter in module.parameters()):
            raise RuntimeError("Ray-only parameters must be frozen")
        if self.config.use_backbone_lora:
            lora_ids = {
                id(parameter)
                for module in self.lora_modules.values()
                for parameter in (module.lora_A, module.lora_B)
            }
            trainable_backbone_ids = {
                id(parameter) for parameter in self.backbone.parameters() if parameter.requires_grad
            }
            if trainable_backbone_ids != lora_ids:
                raise RuntimeError("DINOv2 LoRA mode exposed non-LoRA backbone parameters")

    def parameter_groups(self) -> Dict[str, list[nn.Parameter]]:
        ray_ids = {id(parameter) for module in self._ray_modules() for parameter in module.parameters()}
        def selected(module: nn.Module) -> list[nn.Parameter]:
            return [p for p in module.parameters() if p.requires_grad and id(p) not in ray_ids]
        return {
            "backbone": selected(self.backbone),
            "depth_head": selected(self.depth_head),
            "camera_encoder": selected(self.camera_encoder),
            "camera_decoder": selected(self.camera_decoder),
        }

    def parameter_statistics(self) -> Dict[str, int]:
        result = {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        for name, parameters in self.parameter_groups().items():
            result[name + "_trainable"] = sum(p.numel() for p in parameters)
        result["ray_trainable"] = sum(
            p.numel() for module in self._ray_modules() for p in module.parameters() if p.requires_grad
        )
        result["backbone_lora_trainable"] = sum(
            parameter.numel()
            for module in self.lora_modules.values()
            for parameter in (module.lora_A, module.lora_B)
            if parameter.requires_grad
        )
        result["lora_modules"] = len(self.lora_modules)
        return result

    def enable_cuda_timing(self, enabled: bool) -> None:
        self._timing_enabled = bool(enabled)

    def _record_cuda_timing(self, name: str, device: torch.device) -> None:
        if not self._timing_enabled or device.type != "cuda":
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._last_forward_timing_events[name] = event

    def forward_cuda_timings_ms(self) -> Dict[str, float]:
        events = self._last_forward_timing_events
        pairs = {
            "input_checks": ("start", "backbone_start"),
            "backbone": ("backbone_start", "backbone_end"),
            "depth_head": ("backbone_end", "depth_end"),
            "camera_decoder": ("depth_end", "camera_end"),
            "geometry": ("camera_end", "geometry_end"),
        }
        return {
            name: events[start].elapsed_time(events[end])
            for name, (start, end) in pairs.items()
            if start in events and end in events
        }

    def _fuse_depth_main(self, resized: list[torch.Tensor]) -> torch.Tensor:
        head = self.depth_head
        l1, l2, l3, l4 = resized
        l1 = head.scratch.layer1_rn(l1)
        l2 = head.scratch.layer2_rn(l2)
        l3 = head.scratch.layer3_rn(l3)
        l4 = head.scratch.layer4_rn(l4)
        out = head.scratch.refinenet4(l4, size=l3.shape[2:])
        out = head.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = head.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = head.scratch.refinenet1(out, l1)
        return head.scratch.output_conv1(out)

    def _depth_main_chunk(
        self, tokens: list[torch.Tensor], height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.depth_head
        batch_frames, _, channels = tokens[0].shape
        patch_h, patch_w = height // head.patch_size, width // head.patch_size
        resized = []
        for stage, take in enumerate(head.intermediate_layer_idx):
            value = head.norm(tokens[take])
            value = value.permute(0, 2, 1).reshape(batch_frames, channels, patch_h, patch_w)
            value = head.projects[stage](value)
            if head.pos_embed:
                value = head._add_pos_embed(value, width, height)
            resized.append(head.resize_layers[stage](value))
        fused = self._fuse_depth_main(resized)
        fused = F.interpolate(fused, (height, width), mode="bilinear", align_corners=True)
        if head.pos_embed:
            fused = head._add_pos_embed(fused, width, height)
        logits = head.scratch.output_conv2(fused).permute(0, 2, 3, 1)
        depth = head._apply_activation_single(logits[..., :-1], head.activation).squeeze(-1)
        confidence = head._apply_activation_single(logits[..., -1], head.conf_activation)
        return depth, confidence

    def _forward_depth_main(
        self, feats: list[Any], height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, tokens, channels = feats[0][0].shape
        flat = [feature[0].reshape(batch * frames, tokens, channels) for feature in feats]
        depths, confidences = [], []
        chunk = self.config.head_chunk_size
        for start in range(0, batch * frames, chunk):
            depth, confidence = self._depth_main_chunk(
                [value[start : start + chunk] for value in flat], height, width
            )
            depths.append(depth)
            confidences.append(confidence)
        return (
            torch.cat(depths).reshape(batch, frames, height, width),
            torch.cat(confidences).reshape(batch, frames, height, width),
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._last_forward_timing_events = {}
        self._record_cuda_timing("start", images.device)
        if tuple(images.shape[1:]) != (16, 3, 448, 560):
            raise ValueError("DA3 student requires [B,16,3,448,560], got {}".format(tuple(images.shape)))
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("DA3 dataset RGB must be finite in [0,1]")
        normalized = (images - self.imagenet_mean) / self.imagenet_std
        self._ray_forward_count = 0
        self._record_cuda_timing("backbone_start", images.device)
        feats, _ = self.backbone(
            normalized,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy=self.config.ref_view_strategy,
        )
        self._record_cuda_timing("backbone_end", images.device)
        with torch.autocast(device_type=images.device.type, enabled=False):
            depth, depth_conf = self._forward_depth_main(feats, 448, 560)
            self._record_cuda_timing("depth_end", images.device)
            pose_encoding = self.camera_decoder(feats[-1][1])
            _, _, pose_encoding_to_extri_intri = _require_official_da3()
            c2w, intrinsics = pose_encoding_to_extri_intri(pose_encoding, (448, 560))
            c2w_h = torch.eye(4, device=c2w.device, dtype=c2w.dtype).view(1, 1, 4, 4).repeat(
                c2w.shape[0], c2w.shape[1], 1, 1
            )
            c2w_h[..., :3, :] = c2w
            extrinsics = torch.linalg.inv(c2w_h)[..., :3, :]
            self._record_cuda_timing("camera_end", images.device)
            xyz_local = depth_intrinsics_to_local_points(depth, intrinsics)
            xyz_global = local_to_global_points(xyz_local, extrinsics)
            self._record_cuda_timing("geometry_end", images.device)
        expected = {
            "depth": (images.shape[0], 16, 448, 560),
            "intrinsics": (images.shape[0], 16, 3, 3),
            "extrinsics": (images.shape[0], 16, 3, 4),
            "xyz_local": (images.shape[0], 16, 448, 560, 3),
            "xyz_global": (images.shape[0], 16, 448, 560, 3),
        }
        actual = {
            "depth": tuple(depth.shape), "intrinsics": tuple(intrinsics.shape),
            "extrinsics": tuple(extrinsics.shape), "xyz_local": tuple(xyz_local.shape),
            "xyz_global": tuple(xyz_global.shape),
        }
        if actual != expected:
            raise RuntimeError("DA3 output contract mismatch: {} != {}".format(actual, expected))
        if self._ray_forward_count != 0:
            raise RuntimeError("Ray-specific branch executed {} modules".format(self._ray_forward_count))
        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "xyz_local": xyz_local,
            "xyz_global": xyz_global,
            # Compatibility alias for the unchanged highlight/smoothness loss API.
            "pts3d_local": xyz_local,
        }


__all__ = ["DA3SmallConfig", "DA3SmallStudent"]
