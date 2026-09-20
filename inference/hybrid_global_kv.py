"""Sequence-wide external K/V for DA3-Small global attention, without upstream edits.

V0 recomputes anchor backbone states for each window. The separate anchor pass
uses the same model weights with source-view camera tokens for every external
anchor, and captures K/V after Q/K normalization and 2D RoPE. Only the 32-frame
current pass produces depth/camera outputs; its full Q is never pruned.
"""
from __future__ import annotations

import json
import time
from functools import wraps

import torch

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.global_anchor_bank import (
    extract_frame_descriptors, select_diverse_anchors, select_window_providers,
)
from inference.kv_sampling import build_spatial_patch_indices, resolve_hybrid_spatial_sampling_options


class HybridGlobalKVAttention(DA3KVAttention):
    def __init__(self, model, config, window_length):
        super().__init__(model, config, window_length)
        if len(self.blocks) != 12 or self.layers != [5, 7, 9, 11]:
            raise RuntimeError("Hybrid Global KV requires DA3-Small global blocks 5/7/9/11")
        self.schedule = resolve_hybrid_spatial_sampling_options(config.spatial_sampling)
        self.bank = None
        self.anchor_forward_seconds = 0.0
        self.amp = True
        self.mode = "current"

    def _wrap(self, original, layer):
        wrapped = super()._wrap(original, layer)

        @wraps(original)
        def forward(*args, **kwargs):
            pos = kwargs.get("pos", args[1] if len(args) > 1 else None)
            frames = self.anchor_frames if self.mode == "anchor" else self.frames
            if self.blocks[layer].attn.rope is not None:
                expected = (self.batch, frames * self.tokens_per_frame, 2)
                if pos is None or tuple(pos.shape) != expected:
                    raise RuntimeError("Hybrid global RoPE position layout changed")
                per_frame = pos.reshape(self.batch, frames, self.tokens_per_frame, 2)
                if not torch.equal(per_frame, per_frame[:, :1].expand_as(per_frame)):
                    raise RuntimeError("Frame-dependent global RoPE needs explicit external position mapping")
            return wrapped(*args, **kwargs)

        return forward

    def prepare_sequence(self, frames, *, device, amp=True):
        self.amp = amp
        self.sequence_frames = frames
        tick = time.perf_counter()
        self.mode = "descriptor"
        try:
            descriptors = extract_frame_descriptors(self.model, frames, device=device)
        finally:
            self.mode = "current"
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        descriptor_seconds = time.perf_counter() - tick
        self.bank = select_diverse_anchors(descriptors)
        self.descriptor_seconds = descriptor_seconds
        self.selection_seconds = self.bank.selection_seconds
        grid_h = frames[0].shape[-2] // self.patch_size
        grid_w = frames[0].shape[-1] // self.patch_size
        print("Hybrid Global KV audit: " + json.dumps({
            "sequence_frames": len(frames), "global_bank_size": len(self.bank.bank_positions),
            "global_active_size": len(self.bank.active_positions),
            "inference_resolution": [grid_h * self.patch_size, grid_w * self.patch_size],
            "patch_grid": [grid_h, grid_w], "patches_per_frame": grid_h * grid_w,
            "vda_window": 32, "local_key": 2, "local_overlap": 8,
            "global_active": 15, "target_unique_provider_count": 25,
            "global_bank_sequence_positions": self.bank.bank_positions,
            "global_active_sequence_positions": self.bank.active_positions,
        }), flush=True)

    def begin_window(self, metadata, images):
        if self.bank is None:
            raise RuntimeError("Hybrid anchor bank must be prepared before VDA windows")
        self.metadata = metadata
        self.batch, self.frames = images.shape[:2]
        if self.batch != 1 or self.frames != 32 or images.shape[-2:] != (448, 560):
            raise ValueError("Hybrid input requires [1,32,3,448,560]")
        self.grid_height, self.grid_width = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        self.patches = self.grid_height * self.grid_width
        self.tokens_per_frame = self.special_tokens + self.patches
        self.local_positions, self.global_positions = select_window_providers(metadata, self.bank)
        current_slots = {}
        for slot, (position, padded) in enumerate(zip(metadata.frame_positions, metadata.is_padding)):
            if not padded:
                current_slots.setdefault(position, slot)
        self.current_provider_slots = [current_slots[position] for position in
                                       (*self.local_positions, *self.global_positions) if position in current_slots]
        self.external_positions = [position for position in self.global_positions if position not in current_slots]
        self.anchor_kv = {}
        self.anchor_reference = None
        self.reference_indices = None
        self.token_indices_by_layer = None
        self.calls, self.events = [], []
        if self.external_positions:
            self._prepare_external(images.device)
        self.mode = "current"
        self.reference_indices = None

    def _prepare_external(self, device):
        tick = time.perf_counter()
        rgb = torch.stack([self.sequence_frames[p] for p in self.external_positions]).unsqueeze(0).to(device)
        normalized = (rgb - self.model.imagenet_mean) / self.model.imagenet_std
        self.mode = "anchor"
        self.anchor_frames = len(self.external_positions)
        # External anchors cannot become the current window's reference view.
        # The official cam_token input gives every anchor the native source token
        # and disables anchor-only reference selection/permutation.
        source_tokens = self.encoder.camera_token[:, 1:2].expand(1, self.anchor_frames, -1)
        try:
            with torch.autocast(device_type=device.type, enabled=bool(self.amp and device.type == "cuda")):
                self.model.backbone(
                    normalized, cam_token=source_tokens, export_feat_layers=[],
                    ref_view_strategy=self.model.config.ref_view_strategy,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            self.mode = "current"
        self.anchor_forward_seconds += time.perf_counter() - tick
        if (set(self.anchor_kv) != set(self.layers) or self.anchor_reference is None
            or torch.count_nonzero(self.anchor_reference).item()):
            raise RuntimeError("External anchor backbone pass missed a global layer or reference")

    def _capture_reference(self, module, args, output):
        if self.mode == "descriptor":
            return
        from depth_anything_3.model.reference_view_selector import select_reference_view
        from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

        frames = self.anchor_frames if self.mode == "anchor" else self.frames
        if frames >= THRESH_FOR_REF_SELECTION and not self.camera_conditioned:
            reference = select_reference_view(
                output.detach().reshape(self.batch, frames, *output.shape[1:]),
                strategy=self.ref_strategy,
            ).detach()
        else:
            reference = torch.zeros(self.batch, dtype=torch.long, device=output.device)
        if self.mode == "anchor":
            self.anchor_reference = reference
            return
        self.reference_indices = reference
        self.token_indices_by_layer = {}
        for layer in self.layers:
            current_patches, external_patches = self._patch_layout(layer)
            if self.current_provider_slots:
                current_indices = frame_slots_to_token_indices(
                    self.current_provider_slots, num_frames=self.frames,
                    tokens_per_frame=self.tokens_per_frame,
                    special_tokens=self.special_tokens, reference_indices=reference,
                    patch_indices_by_slot=current_patches,
                )
            else:
                # First-window global providers may all lie outside its 32 Q frames.
                special = (torch.arange(self.frames, device=reference.device)[:, None]
                           * self.tokens_per_frame
                           + torch.arange(self.special_tokens, device=reference.device)).flatten()
                current_indices = special[None, :].expand(self.batch, -1)
            self.token_indices_by_layer[layer] = (
                current_indices,
                frame_slots_to_token_indices(
                    list(range(self.anchor_frames)), num_frames=self.anchor_frames,
                    tokens_per_frame=self.tokens_per_frame,
                    special_tokens=self.special_tokens, reference_indices=self.anchor_reference,
                    patch_indices_by_slot=external_patches,
                ) if self.external_positions else None,
            )

    def _patch_layout(self, layer):
        current, external = {}, {}
        position_to_slot = {pos: slot for slot, (pos, padded) in enumerate(
            zip(self.metadata.frame_positions, self.metadata.is_padding)) if not padded}
        full = build_spatial_patch_indices(self.grid_height, self.grid_width, 1)
        for position in self.local_positions:
            current[position_to_slot[position]] = full
        stride = self.schedule[layer]["global_stride"]
        for rank, position in enumerate(self.global_positions):
            phase = (rank + (2 if layer == 7 else 0)) % 4
            offsets = ((0, 0), (0, 1), (1, 0), (1, 1))[phase] if stride == 2 else (0, 0)
            patches = build_spatial_patch_indices(self.grid_height, self.grid_width, stride, *offsets)
            if position in position_to_slot:
                current[position_to_slot[position]] = patches
            else:
                external[self.external_positions.index(position)] = patches
        return current, external

    def attend(self, layer, kernel, query, key, value, attn_mask=None,
               dropout_p=0.0, is_causal=False, **kwargs):
        if self.mode == "anchor":
            self.anchor_kv[layer] = (key.detach(), value.detach())
            return kernel(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, **kwargs)
        if attn_mask is not None or dropout_p or is_causal:
            raise ValueError("Hybrid attention requires unmasked eval SDPA")
        if query.shape[-2] != self.frames * self.tokens_per_frame:
            raise RuntimeError("Hybrid Q lost a current-window token")
        if self.token_indices_by_layer is None:
            raise RuntimeError("Current DA3 reference permutation was not captured")
        current_indices, external_indices = self.token_indices_by_layer[layer]
        gather = current_indices[:, None, :, None].expand(key.shape[0], key.shape[1], -1, key.shape[-1])
        selected_key = key.gather(2, gather)
        selected_value = value.gather(2, gather)
        if external_indices is not None:
            anchor_key, anchor_value = self.anchor_kv[layer]
            anchor_gather = external_indices[:, None, :, None].expand(
                anchor_key.shape[0], anchor_key.shape[1], -1, anchor_key.shape[-1])
            selected_key = torch.cat((selected_key, anchor_key.gather(2, anchor_gather)), dim=2)
            selected_value = torch.cat((selected_value, anchor_value.gather(2, anchor_gather)), dim=2)
        self.calls.append((layer, int(query.shape[-2]), int(selected_key.shape[-2])))
        event_pair = None
        if self.config.profile_attention:
            if query.is_cuda:
                event_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                event_pair[0].record()
            else:
                tick = time.perf_counter()
        result = kernel(query, selected_key, selected_value,
                        attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs)
        if self.config.profile_attention:
            if event_pair is not None:
                event_pair[1].record()
                self.events.append(event_pair)
            else:
                self.attention_seconds += time.perf_counter() - tick
            self.profiled_calls += 1
        if result.shape != query.shape:
            raise RuntimeError("Hybrid attention changed the query length")
        return result

    def finish_window(self):
        if sorted(layer for layer, _, _ in self.calls) != self.layers:
            raise RuntimeError("Hybrid pass missed a current-window global block")
        self.shape_counts.update(self.calls)
        self.attention_seconds += sum(start.elapsed_time(end) / 1000 for start, end in self.events)
        q = self.frames * self.tokens_per_frame
        audit = {
            "window_id": self.metadata.window_id,
            "local_provider_positions": self.local_positions,
            "global_provider_positions": self.global_positions,
            "deduplicated_provider_count": len(self.local_positions) + len(self.global_positions),
            "external_provider_positions": self.external_positions,
            "q_tokens": q,
            "tokens_by_block": {f"block{layer}": {"q_tokens": query, "kv_tokens": kv}
                                for layer, query, kv in self.calls},
        }
        if len(self.audit_examples) < 3:
            self.audit_examples.append(audit)
            print("Hybrid window audit: " + json.dumps(audit), flush=True)
        self.metadata = None
        self.token_indices_by_layer = None
        self.anchor_kv = {}
        self.events = []

    def summary(self):
        result = super().summary()
        result.update({
            "global_anchor_descriptor_seconds": self.descriptor_seconds,
            "global_anchor_selection_seconds": self.selection_seconds,
            "global_anchor_backbone_seconds": self.anchor_forward_seconds,
            "global_bank_sequence_positions": self.bank.bank_positions,
            "global_active_sequence_positions": self.bank.active_positions,
        })
        return result
