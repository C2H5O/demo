"""Scoped inference adapter for DA3 global-attention frame K/V policies."""

from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import ExitStack
from functools import wraps

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from inference.kv_sampling import (
    KVSamplingConfig,
    WindowFrameMetadata,
    select_kv_frames,
)
from inference.query_group_kv import (
    build_group_token_indices,
    build_query_group_providers,
    grouped_scaled_dot_product_attention,
    provider_token_indices,
)
from models.attention_capture import _blocks


def frame_slots_to_token_indices(
    selected_slots,
    *,
    num_frames: int,
    tokens_per_frame: int,
    special_tokens: int,
    reference_indices: torch.Tensor,
    patch_indices_by_slot=None,
) -> torch.Tensor:
    """Compatibility helper for full-spatial shared-provider F/G policies."""
    if patch_indices_by_slot is not None:
        raise ValueError("Spatial token pruning is not supported by this adapter")
    return provider_token_indices(
        selected_slots,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        special_tokens=special_tokens,
        reference_indices=reference_indices,
        keep_all_special_tokens=True,
    )


class _KVSDPAMode(TorchFunctionMode):
    def __init__(self, owner, layer):
        super().__init__()
        self.owner, self.layer = owner, layer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention:
            return self.owner.attend(self.layer, func, *args, **kwargs)
        return func(*args, **kwargs)


class DA3KVAttention:
    """Temporarily replace selected DA3 global SDPA calls during one sequence."""

    def __init__(self, model, config: KVSamplingConfig, window_length: int):
        self.model, self.config = model, config
        self.budget = config.frame_budget(window_length)
        self.encoder = getattr(getattr(model, "backbone", None), "pretrained", None)
        self.observing = config.enabled or config.debug or config.profile_attention
        self.stack = ExitStack()
        self.layers: list[int] = []
        self.shape_counts = Counter()
        self.audit_examples = []
        self.kernel_call_counts = Counter()
        self.attention_seconds = 0.0
        self.profiled_calls = 0
        self.metadata = None
        self.events = []
        if self.encoder is None:
            if self.observing:
                raise RuntimeError("K/V sampling requires the real DA3 encoder")
            return

        self.patch_size = int(self.encoder.patch_size)
        self.special_tokens = 1 + int(self.encoder.num_register_tokens)
        self.blocks = _blocks(self.encoder)
        alt_start = int(self.encoder.alt_start)
        global_layers = [
            layer
            for layer in range(len(self.blocks))
            if alt_start != -1 and layer >= alt_start and layer % 2 == 1
        ]
        if self.observing and (not global_layers or not 0 <= alt_start - 2 < len(self.blocks)):
            raise RuntimeError("DA3 alt_start does not identify global attention")
        if config.enabled and config.method == "query_group" and config.apply_layers is not None:
            requested = list(dict.fromkeys(config.apply_layers))
            if any(layer not in global_layers for layer in requested):
                raise ValueError("apply_layers must contain DA3 global encoder blocks")
            self.layers = requested
        else:
            self.layers = global_layers

    def __enter__(self):
        if not self.observing:
            return self
        if self.model.training or getattr(self.model, "attention_capture", None) is not None:
            raise ValueError("K/V adapter requires eval mode without attention capture")
        try:
            source = self.blocks[int(self.encoder.alt_start) - 2]
            handle = source.register_forward_hook(self._capture_reference)
            self.stack.callback(handle.remove)
            handle = self.model.backbone.register_forward_pre_hook(
                self._backbone_input, with_kwargs=True
            )
            self.stack.callback(handle.remove)
            for layer in self.layers:
                attention = self.blocks[layer].attn
                if type(attention).__module__ != "depth_anything_3.model.dinov2.layers.attention":
                    raise RuntimeError(
                        "Unsupported DA3 attention implementation at layer " + str(layer)
                    )
                if self.config.enabled:
                    self._replace(attention, "fused_attn", True)
                elif not attention.fused_attn:
                    raise RuntimeError("Dense attention audit requires the SDPA path")
                self._replace(attention, "forward", self._wrap(attention.forward, layer))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self._clear_window()
        return self.stack.__exit__(*exc)

    def _replace(self, module, name, value) -> None:
        existed = name in vars(module)
        previous = vars(module).get(name)

        def restore():
            if existed:
                setattr(module, name, previous)
            else:
                delattr(module, name)

        self.stack.callback(restore)
        setattr(module, name, value)

    def _wrap(self, original, layer):
        @wraps(original)
        def forward(*args, **kwargs):
            if self.metadata is None or self.model.training or torch.is_grad_enabled():
                raise RuntimeError("K/V attention executed outside its inference window")
            with _KVSDPAMode(self, layer):
                return original(*args, **kwargs)

        return forward

    def begin_window(self, metadata: WindowFrameMetadata, images: torch.Tensor) -> None:
        """Build provider sets inside the caller's synchronized model timer."""
        if images.ndim != 5 or images.shape[1] != len(metadata.frame_positions):
            raise ValueError("Window images and metadata must describe the same frames")
        self.metadata = metadata
        self.batch, self.frames = images.shape[:2]
        self.grid_height = images.shape[-2] // self.patch_size if self.encoder is not None else 0
        self.grid_width = images.shape[-1] // self.patch_size if self.encoder is not None else 0
        self.patches = self.grid_height * self.grid_width
        self.tokens_per_frame = self.special_tokens + self.patches if self.encoder is not None else 0
        self.calls, self.events = [], []
        self.reference_indices = None
        self.token_indices = None
        self.query_indices = self.provider_indices = None
        self.group_kernel_calls = {}
        if not self.config.enabled:
            return
        if self.config.method == "query_group":
            self.query_groups, self.provider_groups = build_query_group_providers(
                metadata,
                self.config.query_group_size,
                self.config.kv_frames,
                preserve_vda_history=self.config.preserve_vda_history,
            )
        else:
            self.selected = select_kv_frames(metadata, self.config, self.budget)

    def _backbone_input(self, module, args, kwargs) -> None:
        self.ref_strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
        self.camera_conditioned = kwargs.get("cam_token") is not None

    def _capture_reference(self, module, args, output) -> None:
        from depth_anything_3.model.reference_view_selector import select_reference_view
        from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

        if self.frames >= THRESH_FOR_REF_SELECTION and not self.camera_conditioned:
            value = output.detach().reshape(self.batch, self.frames, *output.shape[1:])
            self.reference_indices = select_reference_view(
                value, strategy=self.ref_strategy
            ).detach()
        else:
            self.reference_indices = torch.zeros(
                self.batch, dtype=torch.long, device=output.device
            )
        if not self.config.enabled:
            return
        if self.config.method == "query_group":
            self.query_indices, self.provider_indices = build_group_token_indices(
                self.query_groups,
                self.provider_groups,
                num_frames=self.frames,
                tokens_per_frame=self.tokens_per_frame,
                special_tokens=self.special_tokens,
                reference_indices=self.reference_indices,
                keep_all_special_tokens=self.config.keep_all_special_tokens,
            )
        else:
            self.token_indices = frame_slots_to_token_indices(
                self.selected,
                num_frames=self.frames,
                tokens_per_frame=self.tokens_per_frame,
                special_tokens=self.special_tokens,
                reference_indices=self.reference_indices,
            )

    def attend(
        self,
        layer,
        kernel,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        **kwargs,
    ):
        expected = self.frames * self.tokens_per_frame
        if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape:
            raise RuntimeError("Expected full DA3 Q/K/V [B,H,F*tokens,D]")
        if query.shape[0] != self.batch or query.shape[-2] != expected:
            raise RuntimeError("Global attention tokens do not match the VDA window")
        if self.config.enabled and (attn_mask is not None or is_causal or dropout_p):
            raise ValueError("Inference K/V sampling expects unmasked noncausal SDPA")

        event_pair = None
        if self.config.profile_attention:
            if query.is_cuda:
                event_pair = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                event_pair[0].record()
            else:
                tick = time.perf_counter()

        if self.config.enabled and self.config.method == "query_group":
            if self.query_indices is None or self.provider_indices is None:
                raise RuntimeError("DA3 reference permutation was not captured")
            result, grouped_calls = grouped_scaled_dot_product_attention(
                kernel,
                query,
                key,
                value,
                self.query_indices,
                self.provider_indices,
                batched_sdpa=self.config.batched_sdpa,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
            self.group_kernel_calls[layer] = grouped_calls
            kv_tokens = self.provider_indices[0].shape[1]
            self.kernel_call_counts[layer] += len(grouped_calls)
        else:
            if self.config.enabled:
                if self.token_indices is None:
                    raise RuntimeError("DA3 reference permutation was not captured")
                indices = self.token_indices[:, None, :, None].expand(
                    self.batch, key.shape[1], -1, key.shape[-1]
                )
                key = torch.gather(key, 2, indices)
                value = torch.gather(value, 2, indices)
            kv_tokens = key.shape[-2]
            result = kernel(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
            self.kernel_call_counts[layer] += 1

        if self.config.profile_attention:
            if event_pair is not None:
                event_pair[1].record()
                self.events.append(event_pair)
            else:
                self.attention_seconds += time.perf_counter() - tick
            self.profiled_calls += 1
        if result.shape != query.shape:
            raise RuntimeError("Attention must return every original query token")
        self.calls.append((layer, int(query.shape[-2]), int(kv_tokens)))
        return result

    def finish_window(self) -> None:
        metadata = self.metadata
        if self.observing:
            if sorted(layer for layer, _, _ in self.calls) != sorted(self.layers):
                raise RuntimeError("Not every configured global layer executed once")
            self.shape_counts.update(self.calls)
            self.attention_seconds += sum(
                start.elapsed_time(end) / 1000 for start, end in self.events
            )
        if self.config.enabled and self.config.method == "query_group" and (
            self.config.debug or self.config.diagnostics
        ):
            audit = {
                "window_id": metadata.window_id,
                "first_window": metadata.first_window,
                "query_groups": self.query_groups,
                "provider_groups": self.provider_groups,
                "provider_counts": [len(group) for group in self.provider_groups],
                "provider_roles": [
                    [metadata.frame_roles[slot] for slot in group]
                    for group in self.provider_groups
                ],
                "keep_all_special_tokens": self.config.keep_all_special_tokens,
                "batched_sdpa": self.config.batched_sdpa,
                "sdpa_calls_by_layer": self.group_kernel_calls,
            }
            if len(self.audit_examples) < self.config.debug_max_windows:
                self.audit_examples.append(audit)
            if self.config.diagnostics:
                print("QG-KV audit: " + json.dumps(audit), flush=True)
        self._clear_window()

    def _clear_window(self) -> None:
        self.metadata = None
        self.reference_indices = None
        self.token_indices = None
        self.query_indices = self.provider_indices = None
        self.events = []

    def summary(self) -> dict:
        kv_count_name = (
            "kv_token_count_per_group"
            if self.config.method == "query_group"
            else "kv_token_count"
        )
        return {
            "kv_sampling": self.config.as_dict(),
            "global_attention_layers": self.layers,
            "attention_shapes": [
                {
                    "layer": layer,
                    "q_token_count": q_tokens,
                    kv_count_name: kv_tokens,
                    "calls": count,
                }
                for (layer, q_tokens, kv_tokens), count in sorted(
                    self.shape_counts.items()
                )
            ],
            "grouped_sdpa_kernel_calls": dict(self.kernel_call_counts),
            "kv_selection_examples": self.audit_examples,
            "grouped_attention_seconds": (
                self.attention_seconds if self.config.profile_attention else None
            ),
            "grouped_attention_profiled_layers": self.profiled_calls,
            "attention_timing_scope": (
                "optional grouped gather + reshape + SDPA + restore time; "
                "model_forward_seconds remains the primary end-to-end model timing"
            ),
        }
