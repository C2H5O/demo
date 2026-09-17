"""Query-grouped frame-level K/V for DA3 global attention."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import torch

from inference.kv_sampling import WindowFrameMetadata, eligible_frame_slots, uniform_select


def build_query_group_providers(
    metadata: WindowFrameMetadata,
    query_group_size: int,
    kv_frames: int,
    *,
    preserve_vda_history: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build temporal query groups and one deterministic provider set per group."""
    frame_count = len(metadata.frame_positions)
    if query_group_size <= 0 or kv_frames <= 0:
        raise ValueError("query_group_size and kv_frames must be positive")
    if kv_frames > frame_count:
        raise ValueError("kv_frames cannot exceed the input window length")

    query_groups = [
        list(range(start, min(start + query_group_size, frame_count)))
        for start in range(0, frame_count, query_group_size)
    ]
    candidates = eligible_frame_slots(metadata)
    temporal_key = lambda slot: (metadata.frame_positions[slot], slot)
    candidates = sorted(candidates, key=temporal_key)
    target = min(kv_frames, len(candidates))
    history = [
        slot for slot in candidates if metadata.frame_roles[slot] in {"key", "overlap"}
    ]

    providers = []
    for queries in query_groups:
        query_positions = {metadata.frame_positions[slot] for slot in queries}
        mandatory = [
            slot for slot in candidates if metadata.frame_positions[slot] in query_positions
        ]
        if len(mandatory) > target:
            raise ValueError(
                "kv_frames is too small to retain every unique frame in a query group"
            )
        selected = list(mandatory)
        if preserve_vda_history:
            available_history = [slot for slot in history if slot not in selected]
            selected.extend(
                uniform_select(available_history, min(target - len(selected), len(available_history)))
            )
        remaining = [slot for slot in candidates if slot not in selected]
        selected.extend(uniform_select(remaining, target - len(selected)))
        providers.append(sorted(selected, key=temporal_key))
    return query_groups, providers


def _internal_frame_indices(
    frame_slots: Sequence[int], reference_indices: torch.Tensor
) -> torch.Tensor:
    slots = torch.tensor(
        list(frame_slots), dtype=torch.long, device=reference_indices.device
    )[None, :]
    references = reference_indices[:, None]
    internal = torch.where(slots < references, slots + 1, slots)
    return torch.where(slots == references, 0, internal)


def full_frame_token_indices(
    frame_slots: Sequence[int],
    *,
    tokens_per_frame: int,
    reference_indices: torch.Tensor,
) -> torch.Tensor:
    """Map original frame slots to all tokens in DA3's reference-first layout."""
    internal = _internal_frame_indices(frame_slots, reference_indices)
    offsets = torch.arange(tokens_per_frame, device=reference_indices.device)
    return (internal[..., None] * tokens_per_frame + offsets).flatten(1)


def provider_token_indices(
    provider_slots: Sequence[int],
    *,
    num_frames: int,
    tokens_per_frame: int,
    special_tokens: int,
    reference_indices: torch.Tensor,
    keep_all_special_tokens: bool,
) -> torch.Tensor:
    """Select full-resolution patch K/V and configurable special-token K/V."""
    if not 0 <= special_tokens < tokens_per_frame:
        raise ValueError("Invalid DA3 special-token prefix")
    internal = _internal_frame_indices(provider_slots, reference_indices)
    if keep_all_special_tokens:
        special = (
            torch.arange(num_frames, device=reference_indices.device)[:, None]
            * tokens_per_frame
            + torch.arange(special_tokens, device=reference_indices.device)
        ).flatten()[None, :].expand(len(reference_indices), -1)
        patch_offsets = torch.arange(
            special_tokens, tokens_per_frame, device=reference_indices.device
        )
        patches = (internal[..., None] * tokens_per_frame + patch_offsets).flatten(1)
        return torch.cat((special, patches), dim=1)
    offsets = torch.arange(tokens_per_frame, device=reference_indices.device)
    return (internal[..., None] * tokens_per_frame + offsets).flatten(1)


def build_group_token_indices(
    query_groups: Sequence[Sequence[int]],
    provider_groups: Sequence[Sequence[int]],
    *,
    num_frames: int,
    tokens_per_frame: int,
    special_tokens: int,
    reference_indices: torch.Tensor,
    keep_all_special_tokens: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    query_indices = [
        full_frame_token_indices(
            group,
            tokens_per_frame=tokens_per_frame,
            reference_indices=reference_indices,
        )
        for group in query_groups
    ]
    provider_indices = [
        provider_token_indices(
            group,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            special_tokens=special_tokens,
            reference_indices=reference_indices,
            keep_all_special_tokens=keep_all_special_tokens,
        )
        for group in provider_groups
    ]
    flattened = torch.cat(query_indices, dim=1)
    expected = torch.arange(
        num_frames * tokens_per_frame, device=reference_indices.device
    )[None, :].expand(len(reference_indices), -1)
    if not torch.equal(flattened.sort(dim=1).values, expected):
        raise RuntimeError("Query groups must cover every DA3 token exactly once")
    return query_indices, provider_indices


def _gather_grouped(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, heads, _, channels = tensor.shape
    groups, length = indices.shape[1:]
    source = tensor[:, None].expand(batch, groups, heads, -1, channels)
    gather = indices[:, :, None, :, None].expand(
        batch, groups, heads, length, channels
    )
    return torch.gather(source, 3, gather)


def grouped_scaled_dot_product_attention(
    kernel: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_indices: Sequence[torch.Tensor],
    provider_indices: Sequence[torch.Tensor],
    *,
    batched_sdpa: bool,
    **kwargs,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Run one SDPA per shape class; standard 32/8/K uses exactly one call."""
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise RuntimeError("Expected DA3 Q/K/V tensors shaped [B,H,tokens,D]")
    by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
    for group, (q_index, kv_index) in enumerate(zip(query_indices, provider_indices)):
        by_shape[(q_index.shape[1], kv_index.shape[1])].append(group)
    batches = []
    for group_ids in by_shape.values():
        if batched_sdpa:
            batches.append(group_ids)
        else:
            batches.extend([[group] for group in group_ids])

    # Under CUDA autocast SDPA may return bf16/fp16 even when its Q tensor is
    # fp32. Restore into the kernel's output dtype, matching native SDPA, rather
    # than assuming that Q and the attended result share a dtype.
    output = None
    calls = []
    batch, heads, _, channels = query.shape
    for group_ids in batches:
        q_index = torch.stack([query_indices[group] for group in group_ids], dim=1)
        kv_index = torch.stack([provider_indices[group] for group in group_ids], dim=1)
        grouped_q = _gather_grouped(query, q_index)
        grouped_k = _gather_grouped(key, kv_index)
        grouped_v = _gather_grouped(value, kv_index)
        group_count, query_tokens, kv_tokens = (
            len(group_ids), q_index.shape[-1], kv_index.shape[-1]
        )
        attended = kernel(
            grouped_q.reshape(batch * group_count, heads, query_tokens, channels),
            grouped_k.reshape(batch * group_count, heads, kv_tokens, channels),
            grouped_v.reshape(batch * group_count, heads, kv_tokens, channels),
            **kwargs,
        ).reshape(batch, group_count, heads, query_tokens, channels)
        if output is None:
            output = torch.empty(
                query.shape, dtype=attended.dtype, device=attended.device
            )
        elif output.dtype != attended.dtype:
            raise RuntimeError("Grouped SDPA shape classes returned different dtypes")
        flat_indices = q_index.flatten(1)
        flat_values = attended.permute(0, 2, 1, 3, 4).reshape(
            batch, heads, group_count * query_tokens, channels
        )
        output.scatter_(
            2,
            flat_indices[:, None, :, None].expand(
                batch, heads, group_count * query_tokens, channels
            ),
            flat_values,
        )
        calls.append(
            {
                "group_count": group_count,
                "query_tokens_per_group": query_tokens,
                "kv_tokens_per_group": kv_tokens,
            }
        )
    if output is None:
        raise RuntimeError("Query grouping produced no SDPA calls")
    return output, calls
