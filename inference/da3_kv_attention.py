"""Scoped DA3 global-attention KV gathering, after upstream normalization/RoPE.

Reuse the original attention forward and intercept only its SDPA call, following
the project's existing instance-local SDPA interception pattern. No checkpoint,
module parameters, upstream source files or process-global functions are changed.
"""
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
    KVSamplingConfig, WindowFrameMetadata, eligible_frame_slots, select_kv_frames,
)
from models.attention_capture import _blocks
from inference.lightweight_highlight import compute_lightweight_highlight_scores


def frame_slots_to_token_indices(selected_slots, *, num_frames: int,
                                  tokens_per_frame: int, special_tokens: int,
                                  reference_indices: torch.Tensor) -> torch.Tensor:
    """Map original window slots into DA3's [reference, remaining] token layout.

    Return [B, K] indices: all frames' special-token prefixes, plus every patch
    token of each selected frame. These exact indices are shared by K and V.
    Positions have already been applied to full Q/K before these indices are used.
    """
    selected_slots = list(selected_slots)
    if not selected_slots or len(selected_slots) != len(set(selected_slots)):
        raise ValueError("Selected frame slots must be nonempty and unique")
    if any(type(slot) is not int or not 0 <= slot < num_frames for slot in selected_slots):
        raise ValueError("Selected frame slot outside token layout")
    if not 0 <= special_tokens < tokens_per_frame:
        raise ValueError("Token layout must contain a special prefix and nonempty patch range")
    if reference_indices.ndim != 1 or reference_indices.dtype != torch.long:
        raise ValueError("Reference indices must be a one-dimensional LongTensor")
    device = reference_indices.device
    slots = torch.tensor(selected_slots, dtype=torch.long, device=device)[None, :]
    reference = reference_indices[:, None]
    internal = torch.where(slots < reference, slots + 1, slots)
    internal = torch.where(slots == reference, 0, internal)
    patches = torch.arange(special_tokens, tokens_per_frame, device=device)
    patch_indices = (internal[:, :, None] * tokens_per_frame + patches).flatten(1)
    special = (torch.arange(num_frames, device=device)[:, None] * tokens_per_frame
               + torch.arange(special_tokens, device=device)).flatten()
    special = special[None, :].expand(len(reference_indices), -1)
    return torch.cat((special, patch_indices), dim=1)


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
    """Temporary inference adapter with explicit per-window metadata.

    Only selected global layers are wrapped, and every wrapper/hook is restored
    even when inference raises. Dense without debug/profiling installs no hooks.
    The current schedule uses one shared frame budget at every global layer.
    """

    def __init__(self, model, config: KVSamplingConfig, window_length: int):
        self.model, self.config = model, config
        self.budget = config.frame_budget(window_length)
        self.encoder = getattr(getattr(model, "backbone", None), "pretrained", None)
        self.observing = config.enabled or config.debug or config.profile_attention
        self.stack = ExitStack()
        self.layers = []
        self.shape_counts = Counter()
        self.audit_examples = []
        self.attention_seconds = 0.0
        self.profiled_calls = 0
        self.metadata = None
        self.highlight_processor = None
        if config.enabled and config.method == "vda_role_highlight":
            from datasets.highlight import HighlightDetectionConfig, SpecularHighlightProcessor
            self.highlight_processor = SpecularHighlightProcessor(
                HighlightDetectionConfig(**config.highlight_detection))
        if self.encoder is None:
            if self.observing:
                raise RuntimeError("KV sampling requires the real DA3 backbone.pretrained encoder")
            return
        self.patch_size = int(self.encoder.patch_size)
        self.special_tokens = 1 + int(self.encoder.num_register_tokens)
        self.blocks = _blocks(self.encoder)
        alt_start = int(self.encoder.alt_start)
        self.layers = [i for i in range(len(self.blocks))
                       if alt_start != -1 and i >= alt_start and i % 2 == 1]
        if self.observing and (not self.layers or not 0 <= alt_start - 2 < len(self.blocks)):
            raise RuntimeError("Unsupported DA3 global/reference-selection block layout")

    def __enter__(self):
        if not self.observing:
            return self
        if self.model.training or getattr(self.model, "attention_capture", None) is not None:
            raise ValueError("KV adapter requires eval mode without training attention capture")
        try:
            # Same reference-source hook convention as DA3AttentionCapture. The
            # deterministic upstream selector is repeated solely to recover its
            # permutation; it never chooses the KV frame set.
            source = self.blocks[int(self.encoder.alt_start) - 2]
            handle = source.register_forward_hook(self._capture_reference)
            self.stack.callback(handle.remove)
            handle = self.model.backbone.register_forward_pre_hook(self._backbone_input, with_kwargs=True)
            self.stack.callback(handle.remove)
            for layer in self.layers:
                attention = self.blocks[layer].attn
                if type(attention).__module__ != "depth_anything_3.model.dinov2.layers.attention":
                    raise RuntimeError("Unsupported attention implementation at layer " + str(layer))
                if not self.config.enabled and not attention.fused_attn:
                    raise RuntimeError("Dense attention audit requires the existing SDPA path")
                if self.config.enabled:
                    self._replace(attention, "fused_attn", True)
                self._replace(attention, "forward", self._wrap(attention.forward, layer))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.metadata = None
        self.token_indices = None
        self.reference_indices = None
        self.events = []
        return self.stack.__exit__(*exc)

    def _replace(self, module, name, value):
        """Restore class-dispatched methods as well as existing instance overrides."""
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
                raise RuntimeError("KV attention executed outside its inference-window scope")
            with _KVSDPAMode(self, layer):
                return original(*args, **kwargs)
        return forward

    def begin_window(self, metadata: WindowFrameMetadata, images: torch.Tensor):
        """Receive provenance from window construction, never infer roles in attention."""
        self.metadata = metadata
        self.highlight_scores = None
        self.selection_audit = {}
        if self.config.enabled and self.config.method == "vda_role_bucket_highlight":
            if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] != len(metadata.frame_positions):
                raise ValueError("Bucket highlight selection requires [1,F,3,H,W] matching metadata")
            new_slots = [slot for slot in eligible_frame_slots(metadata)
                         if metadata.frame_roles[slot] == "new"]
            self.highlight_scores = {}
            if new_slots:
                indices = torch.tensor(new_slots, dtype=torch.long, device=images.device)
                new_images = images[0].index_select(0, indices)
                scores = compute_lightweight_highlight_scores(new_images, self.config.lightweight_highlight)
                # One tiny score-vector host transfer per window, never RGB/masks.
                # Selection stays inside the existing synchronized forward timer.
                self.highlight_scores = dict(zip(new_slots, scores.detach().cpu().tolist()))
        elif self.highlight_processor is not None:
            if images.shape[0] != 1:
                raise ValueError("Highlight window selection requires the sequence inference batch size of one")
            # begin_window is inside synchronized inference timing. Detection,
            # device-to-host copies, pixel ratios and ranking are all included.
            self.highlight_scores = {
                slot: float(self.highlight_processor.detect_mask_numpy(images[0, slot]).mean())
                for slot in eligible_frame_slots(metadata) if metadata.frame_roles[slot] == "new"
            }
        self.selected = select_kv_frames(metadata, self.config, self.budget,
                                         self.highlight_scores, self.selection_audit)
        self.calls, self.events = [], []
        self.reference_indices = None
        self.token_indices = None
        self.batch, self.frames = images.shape[:2]
        if self.frames != len(metadata.frame_positions):
            raise RuntimeError("Metadata does not match model input frames")
        if self.encoder is not None:
            self.patches = (images.shape[-2] // self.patch_size) * (images.shape[-1] // self.patch_size)
            self.tokens_per_frame = self.special_tokens + self.patches

    def _backbone_input(self, module, args, kwargs):
        self.ref_strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
        self.camera_conditioned = kwargs.get("cam_token") is not None

    def _capture_reference(self, module, args, output):
        from depth_anything_3.model.reference_view_selector import select_reference_view
        from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

        if self.frames >= THRESH_FOR_REF_SELECTION and not self.camera_conditioned:
            value = output.detach().reshape(self.batch, self.frames, *output.shape[1:])
            self.reference_indices = select_reference_view(value, strategy=self.ref_strategy).detach()
        else:
            self.reference_indices = torch.zeros(self.batch, dtype=torch.long, device=output.device)
        if self.config.enabled:
            self.token_indices = frame_slots_to_token_indices(
                self.selected, num_frames=self.frames, tokens_per_frame=self.tokens_per_frame,
                special_tokens=self.special_tokens, reference_indices=self.reference_indices,
            )

    def attend(self, layer, kernel, query, key, value, attn_mask=None,
               dropout_p=0.0, is_causal=False, **kwargs):
        """Gather full-frame patch K/V before SDPA; leave full Q and its order intact."""
        expected = self.frames * self.tokens_per_frame
        if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape:
            raise RuntimeError("Expected full DA3 Q/K/V [B,heads,F*tokens,head_dim]")
        if query.shape[0] != self.batch or query.shape[-2] != expected:
            raise RuntimeError("Global attention token count does not match window metadata")
        if self.config.enabled:
            if attn_mask is not None or is_causal or dropout_p:
                raise ValueError("VDA KV sampling expects unmasked, noncausal eval attention")
            if self.token_indices is None:
                raise RuntimeError("DA3 reference permutation was not captured before global attention")
            indices = self.token_indices[:, None, :, None].expand(
                self.batch, key.shape[1], -1, key.shape[-1])
            key = torch.gather(key, dim=2, index=indices)
            value = torch.gather(value, dim=2, index=indices)
        expected_kv = len(self.selected) * self.patches + self.frames * self.special_tokens
        if key.shape[-2] != expected_kv or key.shape != value.shape:
            raise RuntimeError("K/V gather did not preserve the shared selected-token budget")
        self.calls.append((layer, int(query.shape[-2]), int(key.shape[-2])))
        event_pair = None
        if self.config.profile_attention:
            if query.is_cuda:
                event_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                event_pair[0].record()
            else:
                tick = time.perf_counter()
        result = kernel(query, key, value, attn_mask=attn_mask,
                        dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        if self.config.profile_attention:
            if event_pair is not None:
                event_pair[1].record()
                self.events.append(event_pair)
            else:
                self.attention_seconds += time.perf_counter() - tick
            self.profiled_calls += 1
        if result.shape != query.shape:
            raise RuntimeError("Attention must return every original query token")
        return result

    def finish_window(self):
        """Audit actual SDPA dimensions after the caller's CUDA synchronization."""
        metadata = self.metadata
        if self.observing:
            if sorted(layer for layer, _, _ in self.calls) != self.layers:
                raise RuntimeError("Not every target global layer executed exactly one audited SDPA")
            self.shape_counts.update(self.calls)
            self.attention_seconds += sum(start.elapsed_time(end) / 1000 for start, end in self.events)
        if self.encoder is not None:
            q_tokens = self.frames * self.tokens_per_frame
            kv_tokens = len(self.selected) * self.patches + self.frames * self.special_tokens
            if not self.observing:
                self.shape_counts.update((layer, q_tokens, kv_tokens) for layer in self.layers)
            selected_roles = {role: [slot for slot in self.selected if metadata.frame_roles[slot] == role]
                              for role in ("key", "overlap", "new")}
            audit = {
                "window_id": metadata.window_id,
                "window_slots": list(range(self.frames)),
                "sequence_positions": list(metadata.frame_positions),
                "absolute_frame_ids": list(metadata.absolute_frame_ids),
                "absolute_id_source": metadata.absolute_id_source,
                "frame_roles": list(metadata.frame_roles),
                "is_padding": list(metadata.is_padding),
                "role_counts": dict(Counter(metadata.frame_roles)),
                "selected_kv_window_slots": self.selected,
                "selected_kv_absolute_frame_ids": [metadata.absolute_frame_ids[slot] for slot in self.selected],
                "selected_by_role": selected_roles,
                "selected_role_counts": {role: len(slots) for role, slots in selected_roles.items()},
                "total_kv_frame_count": len(self.selected),
                "input_frame_count": self.frames,
                "q_patch_token_count": self.frames * self.patches,
                "kv_patch_token_count": len(self.selected) * self.patches,
                "special_tokens_per_frame": self.special_tokens,
                "q_token_count": q_tokens, "kv_token_count": kv_tokens,
                "frame_retention_ratio": len(self.selected) / self.frames,
                "token_retention_ratio": kv_tokens / q_tokens,
                "token_count_source": "observed_sdpa_inputs" if self.observing else "encoder_layout_dense",
            }
            if self.highlight_scores is not None:
                audit["new_highlight_scores"] = self.highlight_scores
                audit["new_candidate_count"] = len(self.highlight_scores)
                audit["highlight_score_type"] = "highlight_pixels / all_RGB_pixels"
                if self.config.method == "vda_role_bucket_highlight":
                    audit["highlight_score_type"] = "gpu_brightness_low_saturation_ratio"
                    audit.update(self.selection_audit)
            audit_limit = (self.config.debug_max_windows
                           if self.config.debug and self.config.method == "vda_role_bucket_highlight" else 2)
            if len(self.audit_examples) < audit_limit:
                self.audit_examples.append(audit)
            if self.config.debug and metadata.window_id < self.config.debug_max_windows:
                # Logging is outside model-forward timing. No GPU->CPU reference
                # transfer is needed during the timed attention path.
                audit = dict(audit)
                audit["da3_reference_window_slots"] = self.reference_indices.cpu().tolist()
                print("KV selection audit: " + json.dumps(audit, ensure_ascii=False), flush=True)
        self.metadata = None
        self.token_indices = None
        self.reference_indices = None
        self.events = []

    def summary(self):
        """Bounded sequence-level shape audit and optional measured SDPA time."""
        return {
            "kv_sampling": self.config.as_dict(),
            "global_attention_layers": self.layers,
            "attention_token_count_source": ("observed_sdpa_inputs" if self.observing else
                                              "encoder_layout_dense" if self.encoder is not None else "unavailable"),
            "attention_shapes": [{"layer": layer, "q_token_count": q, "kv_token_count": k, "calls": count}
                                 for (layer, q, k), count in sorted(self.shape_counts.items())],
            "kv_selection_examples": self.audit_examples,
            "global_sdpa_seconds": self.attention_seconds if self.config.profile_attention else None,
            "global_sdpa_profiled_calls": self.profiled_calls,
            "attention_timing_scope": "optional sum of global SDPA calls only; excludes QKV projection, normalization, RoPE, frame selection and K/V gather",
        }
