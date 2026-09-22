"""Pre-projection hidden-token K/V sparsity for DA3 global attention."""
from __future__ import annotations

import time
from collections import Counter
from functools import wraps

import torch
import torch.nn.functional as F

from inference.da3_kv_attention import DA3KVAttention, frame_slots_to_token_indices
from inference.kv_sampling import build_spatial_patch_indices, temporal_stride_slots


def split_fused_qkv(qkv, hidden_dim: int):
    """Return parameter views for Q, K and V without cloning or registering weights."""
    if not isinstance(qkv, torch.nn.Linear):
        raise RuntimeError("Layer-stride KV requires the official nn.Linear fused qkv")
    if qkv.in_features != hidden_dim or qkv.out_features != 3 * hidden_dim:
        raise RuntimeError("Fused qkv dimensions do not match the attention hidden size")
    weight = qkv.weight
    bias = qkv.bias
    weights = (weight[:hidden_dim], weight[hidden_dim:2 * hidden_dim], weight[2 * hidden_dim:])
    biases = ((None, None, None) if bias is None else
              (bias[:hidden_dim], bias[hidden_dim:2 * hidden_dim], bias[2 * hidden_dim:]))
    return weights, biases


def project_full_q_sparse_kv(attention, x: torch.Tensor, x_kv: torch.Tensor):
    """Apply Q to every hidden token and K/V only to selected hidden tokens."""
    if x.ndim != 3 or x_kv.ndim != 3 or x.shape[0] != x_kv.shape[0]:
        raise ValueError("Q and sparse K/V hidden states must be [B,tokens,hidden]")
    batch, q_tokens, hidden_dim = x.shape
    if x_kv.shape[-1] != hidden_dim:
        raise ValueError("Q and K/V hidden dimensions differ")
    heads = int(attention.num_heads)
    if hidden_dim % heads:
        raise RuntimeError("Attention hidden size must be divisible by its head count")
    head_dim = hidden_dim // heads
    (weight_q, weight_k, weight_v), (bias_q, bias_k, bias_v) = split_fused_qkv(
        attention.qkv, hidden_dim
    )
    q = F.linear(x, weight_q, bias_q).reshape(batch, q_tokens, heads, head_dim).transpose(1, 2)
    kv_tokens = int(x_kv.shape[1])
    k = F.linear(x_kv, weight_k, bias_k).reshape(
        batch, kv_tokens, heads, head_dim
    ).transpose(1, 2)
    v = F.linear(x_kv, weight_v, bias_v).reshape(
        batch, kv_tokens, heads, head_dim
    ).transpose(1, 2)
    return q, k, v


def sparse_attention_output(attention, x: torch.Tensor, kv_indices: torch.Tensor,
                            pos: torch.Tensor | None = None,
                            attn_mask: torch.Tensor | None = None):
    """Official DA3 attention math with hidden-state selection before K/V Linear."""
    if attention.training:
        raise ValueError("Layer-stride attention is inference-only")
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("Layer-stride attention requires [1,tokens,hidden]")
    if kv_indices.ndim != 1 or kv_indices.dtype != torch.long:
        raise ValueError("K/V indices must be a one-dimensional LongTensor")
    if kv_indices.device != x.device:
        raise ValueError("K/V indices and hidden states must share a device")
    if attn_mask is not None:
        raise ValueError("Layer-stride attention requires unmasked global attention")
    x_kv = x.index_select(1, kv_indices)
    pos_kv = None
    if pos is not None:
        if pos.ndim != 3 or pos.shape[:2] != x.shape[:2]:
            raise RuntimeError("DA3 RoPE position tensor does not match full Q tokens")
        pos_kv = pos.index_select(1, kv_indices)
        if pos_kv.shape[1] != x_kv.shape[1]:
            raise RuntimeError("Sparse K positions do not match sparse hidden tokens")
    q, k, v = project_full_q_sparse_kv(attention, x, x_kv)
    q = attention.q_norm(q)
    k = attention.k_norm(k)
    if attention.rope is not None and pos is not None:
        q = attention.rope(q, pos)
        k = attention.rope(k, pos_kv)
    if attention.fused_attn:
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=attention.attn_drop.p if attention.training else 0.0,
            attn_mask=None,
        )
    else:
        scores = (q * attention.scale) @ k.transpose(-2, -1)
        weights = attention.attn_drop(scores.softmax(dim=-1))
        output = weights @ v
    output = output.transpose(1, 2).reshape(x.shape)
    return attention.proj_drop(attention.proj(output))


class LayerStrideKVAttention(DA3KVAttention):
    """Replace only layer_stride_kv global attention with pre-K/V selection."""

    _STAGES = (
        "kv_selection", "q_projection", "kv_projection", "qk_norm_rope",
        "sdpa", "output_projection",
    )

    def __init__(self, model, config, window_length):
        super().__init__(model, config, window_length)
        self.projection_token_counts = {}
        self.profile_seconds = Counter()
        self.profile_events = []
        self._static_selected_by_layer = None
        self._static_patch_indices_by_layer = None

    def begin_window(self, metadata, images):
        super().begin_window(metadata, images)
        if images.ndim != 5 or tuple(images.shape[:2]) != (1, 32):
            raise ValueError("Layer-stride KV requires one complete 32-slot query window")
        if self._static_selected_by_layer is None:
            spatial_slots = list(range(32))
            temporal_slots = list(temporal_stride_slots(32, self.config.temporal_stride))
            lattice = build_spatial_patch_indices(
                self.grid_height, self.grid_width, self.config.spatial_stride)
            spatial_patches = {slot: lattice for slot in spatial_slots}
            self._static_selected_by_layer = {}
            self._static_patch_indices_by_layer = {}
            for layer, policy in self.layer_stride_policy.items():
                self._static_selected_by_layer[layer] = (
                    spatial_slots if policy == "spatial" else temporal_slots)
                self._static_patch_indices_by_layer[layer] = (
                    spatial_patches if policy == "spatial" else None)
        self.projection_token_counts = {}
        self.profile_events = []

    def _capture_reference(self, module, args, output):
        """Map cached original-slot patterns into this window's reference-first layout."""
        from depth_anything_3.model.reference_view_selector import select_reference_view
        from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

        if self.frames >= THRESH_FOR_REF_SELECTION and not self.camera_conditioned:
            value = output.detach().reshape(self.batch, self.frames, *output.shape[1:])
            self.reference_indices = select_reference_view(
                value, strategy=self.ref_strategy).detach()
        else:
            self.reference_indices = torch.zeros(
                self.batch, dtype=torch.long, device=output.device)
        self.token_indices_by_layer = {}
        self.selected_by_layer = {}
        self.patch_indices_by_layer = {}
        token_layout_cache = {}
        for layer in self.layers:
            selected = self._static_selected_by_layer[layer]
            patches = self._static_patch_indices_by_layer[layer]
            signature = self.layer_stride_policy[layer]
            if signature not in token_layout_cache:
                token_layout_cache[signature] = frame_slots_to_token_indices(
                    selected,
                    num_frames=self.frames,
                    tokens_per_frame=self.tokens_per_frame,
                    special_tokens=self.special_tokens,
                    reference_indices=self.reference_indices,
                    patch_indices_by_slot=patches,
                )
            self.selected_by_layer[layer] = selected
            self.patch_indices_by_layer[layer] = patches
            self.token_indices_by_layer[layer] = token_layout_cache[signature]
        self.token_indices = self.token_indices_by_layer[self.layers[0]]

    def _measure(self, layer, stage, tensor, operation):
        if not self.config.profile_attention:
            return operation()
        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            self.profile_events.append((layer, stage, start, end))
            return result
        started = time.perf_counter()
        result = operation()
        self.profile_seconds[(layer, stage)] += time.perf_counter() - started
        return result

    def _wrap(self, original, layer):
        attention = self.blocks[layer].attn

        @wraps(original)
        def forward(x, pos=None, attn_mask=None):
            if self.metadata is None or self.model.training or torch.is_grad_enabled():
                raise RuntimeError("Layer-stride attention executed outside inference-window scope")
            if x.ndim != 3 or x.shape[0] != 1:
                raise RuntimeError("Layer-stride global attention requires [1,tokens,hidden]")
            if attn_mask is not None:
                raise ValueError("Layer-stride KV requires unmasked global attention")
            if self.token_indices_by_layer is None or layer not in self.token_indices_by_layer:
                raise RuntimeError("Reference permutation was not captured before global attention")
            indices = self.token_indices_by_layer[layer][0]
            q_tokens = int(x.shape[1])
            kv_tokens = int(indices.numel())
            if q_tokens != self.frames * self.tokens_per_frame:
                raise RuntimeError("Layer-stride Q token layout changed")

            def select_kv():
                hidden = x.index_select(1, indices)
                sparse_pos = None if pos is None else pos.index_select(1, indices)
                return hidden, sparse_pos

            x_kv, pos_kv = self._measure(layer, "kv_selection", x, select_kv)
            if pos is not None and (pos.ndim != 3 or pos.shape[:2] != x.shape[:2]
                                    or pos_kv.shape[:2] != x_kv.shape[:2]):
                raise RuntimeError("Full-Q and sparse-K RoPE position shapes are inconsistent")

            hidden_dim = int(x.shape[-1])
            heads = int(attention.num_heads)
            head_dim = hidden_dim // heads
            weights, biases = split_fused_qkv(attention.qkv, hidden_dim)
            weight_q, weight_k, weight_v = weights
            bias_q, bias_k, bias_v = biases

            q = self._measure(
                layer, "q_projection", x,
                lambda: F.linear(x, weight_q, bias_q).reshape(
                    1, q_tokens, heads, head_dim).transpose(1, 2),
            )

            def project_kv():
                k = F.linear(x_kv, weight_k, bias_k).reshape(
                    1, kv_tokens, heads, head_dim).transpose(1, 2)
                v = F.linear(x_kv, weight_v, bias_v).reshape(
                    1, kv_tokens, heads, head_dim).transpose(1, 2)
                return k, v

            k, v = self._measure(layer, "kv_projection", x_kv, project_kv)

            def normalize_and_rope():
                normalized_q = attention.q_norm(q)
                normalized_k = attention.k_norm(k)
                if attention.rope is not None and pos is not None:
                    normalized_q = attention.rope(normalized_q, pos)
                    normalized_k = attention.rope(normalized_k, pos_kv)
                return normalized_q, normalized_k

            q, k = self._measure(layer, "qk_norm_rope", q, normalize_and_rope)

            def sdpa():
                if attention.fused_attn:
                    return F.scaled_dot_product_attention(
                        q, k, v, dropout_p=0.0, attn_mask=None)
                scores = (q * attention.scale) @ k.transpose(-2, -1)
                return attention.attn_drop(scores.softmax(dim=-1)) @ v

            output = self._measure(layer, "sdpa", q, sdpa)
            if self.config.profile_attention:
                self.profiled_calls += 1

            def output_projection():
                value = output.transpose(1, 2).reshape(1, q_tokens, hidden_dim)
                return attention.proj_drop(attention.proj(value))

            result = self._measure(layer, "output_projection", output, output_projection)
            self.calls.append((layer, q_tokens, kv_tokens))
            self.projection_token_counts[layer] = {
                "q_projection_tokens": q_tokens,
                "k_projection_tokens": kv_tokens,
                "v_projection_tokens": kv_tokens,
                "sdpa_q_tokens": int(q.shape[-2]),
                "sdpa_kv_tokens": int(k.shape[-2]),
            }
            if result.shape != x.shape:
                raise RuntimeError("Layer-stride attention changed the full-Q output shape")
            return result

        return forward

    def attend(self, *args, **kwargs):
        raise RuntimeError("Layer-stride KV must not sparsify after QKV projection")

    def finish_window(self):
        for layer, stage, start, end in self.profile_events:
            self.profile_seconds[(layer, stage)] += start.elapsed_time(end) / 1000.0
        self.profile_events = []
        super().finish_window()

    def summary(self):
        result = super().summary()
        result["layer_stride_profile_seconds"] = {
            f"block{layer}": {
                stage: self.profile_seconds.get((layer, stage), 0.0)
                for stage in self._STAGES
            }
            for layer in self.layers
        } if self.config.profile_attention else None
        if self.config.profile_attention:
            result["global_sdpa_seconds"] = sum(
                self.profile_seconds.get((layer, "sdpa"), 0.0) for layer in self.layers)
            result["global_sdpa_profiled_calls"] = self.profiled_calls
        result["layer_stride_optimization_point"] = "hidden-token selection before K/V projection"
        result["full_projected_kv_gather"] = False
        return result
