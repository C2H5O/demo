"""Persistent sequence-scoped K/V from fixed global DA3 anchors."""
from __future__ import annotations

import hashlib
import json
import time
from functools import wraps

import torch

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.global_anchor_bank import extract_frame_descriptors, select_diverse_anchors
from inference.kv_sampling import (
    build_spatial_patch_indices, resolve_fixed_global_spatial_sampling_options,
)


class HybridGlobalKVAttention(DA3KVAttention):
    """Build anchor K/V once, then serve every 32-query VDA window."""

    def __init__(self, model, config, window_length):
        super().__init__(model, config, window_length)
        if len(self.blocks) != 12 or self.layers != [5, 7, 9, 11]:
            raise RuntimeError("Fixed Global KV requires DA3-Small blocks 5/7/9/11")
        self.schedule = resolve_fixed_global_spatial_sampling_options(config.spatial_sampling)
        self.mode, self.bank = "idle", None
        self.global_kv_cache = {}
        self.cache_build_count = 0
        self.cache_build_seconds = 0.0
        self.cache_bytes = 0

    def __exit__(self, *exc):
        self.global_kv_cache.clear()
        self.bank, self.mode = None, "idle"
        return super().__exit__(*exc)

    def _wrap(self, original, layer):
        wrapped = super()._wrap(original, layer)
        @wraps(original)
        def forward(*args, **kwargs):
            pos = kwargs.get("pos", args[1] if len(args) > 1 else None)
            frames = self.anchor_frames if self.mode == "cache_build" else self.frames
            if self.blocks[layer].attn.rope is not None:
                expected = (1, frames * self.tokens_per_frame, 2)
                if pos is None or tuple(pos.shape) != expected:
                    raise RuntimeError("Fixed Global KV RoPE position layout changed")
                per_frame = pos.reshape(1, frames, self.tokens_per_frame, 2)
                if not torch.equal(per_frame, per_frame[:, :1].expand_as(per_frame)):
                    raise RuntimeError("Frame-dependent global RoPE forbids persistent K/V reuse")
            return wrapped(*args, **kwargs)
        return forward

    def prepare_sequence(self, frames, *, device, amp=True):
        self.global_kv_cache.clear()
        self.bank = None
        self.cache_build_count = 0
        self.cache_build_seconds = 0.0
        self.cache_bytes = 0
        self.amp, device = amp, torch.device(device)
        tick = time.perf_counter()
        self.mode = "descriptor"
        try:
            descriptors = extract_frame_descriptors(self.model, frames, device=device)
        finally:
            self.mode = "idle"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.descriptor_seconds = time.perf_counter() - tick
        self.bank = select_diverse_anchors(descriptors, anchor_count=50)
        self.selection_seconds = self.bank.selection_seconds
        self.anchor_positions = self.bank.anchor_positions
        expected = min(50, len(frames))
        if len(self.anchor_positions) != expected or len(set(self.anchor_positions)) != expected:
            raise RuntimeError("Fixed Global KV anchor-count contract failed")
        self.grid_height = frames[0].shape[-2] // self.patch_size
        self.grid_width = frames[0].shape[-1] // self.patch_size
        self.patches = self.grid_height * self.grid_width
        self.tokens_per_frame = self.special_tokens + self.patches
        self.anchor_frames = expected
        self._build_cache_once(frames, device)
        if self.cache_build_count != 1 or set(self.global_kv_cache) != set(self.layers):
            raise RuntimeError("Fixed Global KV cache must be built exactly once per sequence")
        self.anchor_hash = hashlib.sha256(
            ",".join(map(str, self.anchor_positions)).encode("ascii")
        ).hexdigest()[:16]
        print("Fixed Global KV audit: " + json.dumps({
            "sequence_frames": len(frames), "anchor_count": expected,
            "anchor_provider_positions": self.anchor_positions,
            "anchor_hash": self.anchor_hash, "provider_policy": "fixed_sequence_global",
            "inference_resolution": [self.grid_height * self.patch_size,
                                     self.grid_width * self.patch_size],
            "patch_grid": [self.grid_height, self.grid_width],
            "cache_build_count": self.cache_build_count,
            "global_kv_cache_bytes": self.cache_bytes,
            "global_stride_by_block": {str(k): v["global_stride"]
                                       for k, v in self.schedule.items()},
        }), flush=True)

    def _build_cache_once(self, frames, device):
        if self.cache_build_count or self.global_kv_cache:
            raise RuntimeError("Attempted to rebuild sequence-global K/V cache")
        tick = time.perf_counter()
        rgb = torch.stack([frames[p] for p in self.anchor_positions]).unsqueeze(0).to(device)
        normalized = (rgb - self.model.imagenet_mean) / self.model.imagenet_std
        source_tokens = self.encoder.camera_token[:, 1:2].expand(1, self.anchor_frames, -1)
        self.mode = "cache_build"
        self.metadata = "cache_build"
        try:
            with torch.autocast(device_type=device.type, enabled=bool(self.amp and device.type == "cuda")):
                self.model.backbone(normalized, cam_token=source_tokens, export_feat_layers=[],
                                    ref_view_strategy=self.model.config.ref_view_strategy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            self.mode = "idle"
            self.metadata = None
        self.cache_build_seconds = time.perf_counter() - tick
        self.cache_build_count = 1
        if set(self.global_kv_cache) != set(self.layers):
            raise RuntimeError("Anchor cache build missed a global attention block")
        self.cache_bytes = sum(t.numel() * t.element_size()
                               for pair in self.global_kv_cache.values() for t in pair)

    def _anchor_token_indices(self, layer, device):
        stride = self.schedule[layer]["global_stride"]
        patches = {}
        for rank in range(self.anchor_frames):
            phase = (rank + (2 if layer == 7 else 0)) % 4
            offsets = ((0, 0), (0, 1), (1, 0), (1, 1))[phase] if stride == 2 else (0, 0)
            patches[rank] = build_spatial_patch_indices(
                self.grid_height, self.grid_width, stride, *offsets)
        return frame_slots_to_token_indices(
            list(range(self.anchor_frames)), num_frames=self.anchor_frames,
            tokens_per_frame=self.tokens_per_frame, special_tokens=self.special_tokens,
            reference_indices=torch.zeros(1, dtype=torch.long, device=device),
            patch_indices_by_slot=patches)

    def _capture_reference(self, module, args, output):
        # The current pass keeps its full Q, and provider K/V comes exclusively
        # from the fixed cache, so no current-window gather layout is needed.
        return

    def begin_window(self, metadata, images):
        if self.cache_build_count != 1 or set(self.global_kv_cache) != set(self.layers):
            raise RuntimeError("Window started without one complete persistent K/V cache")
        self.metadata = metadata
        self.batch, self.frames = images.shape[:2]
        if self.batch != 1 or self.frames != 32 or images.shape[-2:] != (448, 560):
            raise ValueError("Fixed Global KV requires [1,32,3,448,560] queries")
        if len(self.anchor_positions) != self.anchor_frames:
            raise RuntimeError("Fixed anchor positions changed between windows")
        self.calls, self.events = [], []
        self.reference_indices = None
        self.token_indices_by_layer = None
        self.mode = "window"

    def attend(self, layer, kernel, query, key, value, attn_mask=None,
               dropout_p=0.0, is_causal=False, **kwargs):
        if self.mode == "cache_build":
            if layer in self.global_kv_cache:
                raise RuntimeError("Global block executed twice during cache build")
            indices = self._anchor_token_indices(layer, key.device)
            gather = indices[:, None, :, None].expand(key.shape[0], key.shape[1], -1, key.shape[-1])
            self.global_kv_cache[layer] = (key.gather(2, gather).detach(),
                                           value.gather(2, gather).detach())
            return kernel(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        if self.mode != "window":
            raise RuntimeError("Fixed Global KV attention executed outside cache/window scope")
        if attn_mask is not None or dropout_p or is_causal:
            raise ValueError("Fixed Global KV requires unmasked eval SDPA")
        if query.shape[-2] != 32 * self.tokens_per_frame:
            raise RuntimeError("Fixed Global KV must retain all 32 query frames")
        cached_key, cached_value = self.global_kv_cache[layer]
        if cached_key.dtype != key.dtype or cached_key.device != key.device:
            raise RuntimeError("Persistent K/V cache dtype/device differs from current inference")
        self.calls.append((layer, int(query.shape[-2]), int(cached_key.shape[-2])))
        result = kernel(query, cached_key, cached_value, attn_mask=None,
                        dropout_p=0.0, is_causal=False, **kwargs)
        if result.shape != query.shape:
            raise RuntimeError("Fixed Global KV changed the query length")
        return result

    def finish_window(self):
        if sorted(layer for layer, _, _ in self.calls) != self.layers:
            raise RuntimeError("Fixed Global KV missed a current-window global block")
        if self.cache_build_count != 1:
            raise RuntimeError("Persistent cache was rebuilt during window inference")
        self.shape_counts.update(self.calls)
        audit = {
            "window_id": self.metadata.window_id, "q_frame_count": 32,
            "anchor_provider_positions": self.anchor_positions,
            "provider_count": self.anchor_frames,
            "provider_policy": "fixed_sequence_global", "anchor_hash": self.anchor_hash,
            "cache_build_count": self.cache_build_count,
            "q_tokens": 32 * self.tokens_per_frame,
            "kv_tokens_by_block": {f"block{layer}": kv for layer, _, kv in self.calls},
        }
        if len(self.audit_examples) < 3:
            self.audit_examples.append(audit)
            print("Fixed Global KV window audit: " + json.dumps(audit), flush=True)
        self.metadata, self.mode, self.events = None, "idle", []

    def summary(self):
        result = super().summary()
        result.update({
            "global_anchor_descriptor_seconds": self.descriptor_seconds,
            "global_anchor_selection_seconds": self.selection_seconds,
            "global_anchor_cache_build_seconds": self.cache_build_seconds,
            "global_kv_cache_bytes": self.cache_bytes,
            "global_anchor_sequence_positions": self.anchor_positions,
            "cache_build_count": self.cache_build_count,
            "provider_policy": "fixed_sequence_global",
            "persistent_kv_cache": True,
        })
        return result
