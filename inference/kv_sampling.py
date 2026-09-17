"""Small, configurable frame-selection surface for inference-time K/V."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class WindowFrameMetadata:
    """VDA window provenance in input order, before DA3 reference permutation."""

    window_id: int
    frame_positions: tuple[int, ...]
    absolute_frame_ids: tuple[int, ...]
    frame_roles: tuple[str, ...]
    is_padding: tuple[bool, ...]
    first_window: bool
    absolute_id_source: str = "sequence_position"

    def __post_init__(self) -> None:
        size = len(self.frame_positions)
        if not size or any(
            len(value) != size
            for value in (self.absolute_frame_ids, self.frame_roles, self.is_padding)
        ):
            raise ValueError("Window metadata fields must have the same nonzero length")
        if any(role not in {"key", "overlap", "new"} for role in self.frame_roles):
            raise ValueError("Unknown VDA frame role")


@dataclass(frozen=True)
class KVSamplingConfig:
    enabled: bool = False
    method: str = "vda_role"

    # QG-K parameters.
    query_group_size: int = 8
    kv_frames: int = 20
    provider_selection: str = "temporal_uniform"
    preserve_vda_history: bool = True
    keep_all_special_tokens: bool = True
    apply_layers: list[int] | None = None
    batched_sdpa: bool = True
    diagnostics: bool = False

    # Compact legacy F/G policy surface retained for their inherited configs.
    retention_ratio: float = 0.25
    key_frames: int = 2
    overlap_frames: int = 2
    new_frames: int = 4
    first_window_method: str = "uniform"
    first_window_num_frames: int = 8
    temporal_stride: int = 4

    debug: bool = False
    debug_max_windows: int = 2
    profile_attention: bool = False

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            return value
        values = dict(value or {})
        role = dict(values.pop("vda_role", {}))
        first = dict(values.pop("first_window", {}))
        stride = dict(values.pop("spark3r_fixed_stride", {}))
        return cls(
            **values,
            **role,
            **{"first_window_" + key: item for key, item in first.items()},
            **stride,
        )

    def frame_budget(self, window_length: int, *, first_window: bool = False) -> int:
        if not self.enabled:
            return window_length
        if self.method == "query_group":
            if self.query_group_size <= 0 or self.kv_frames <= 0:
                raise ValueError("query_group_size and kv_frames must be positive")
            if self.kv_frames > window_length:
                raise ValueError("kv_frames cannot exceed the input window length")
            if self.provider_selection != "temporal_uniform":
                raise ValueError("query_group provider_selection must be temporal_uniform")
            if not all(
                isinstance(value, bool)
                for value in (
                    self.preserve_vda_history,
                    self.keep_all_special_tokens,
                    self.batched_sdpa,
                    self.diagnostics,
                )
            ):
                raise ValueError("Query-group switches must be booleans")
            if self.apply_layers is not None and (
                not isinstance(self.apply_layers, (list, tuple))
                or any(type(layer) is not int or layer < 0 for layer in self.apply_layers)
            ):
                raise ValueError("apply_layers must be null or encoder block indices")
            return self.kv_frames

        if self.method not in {"vda_role", "spark3r_fixed_stride"}:
            raise ValueError("Unknown kv_sampling.method")
        if not isfinite(self.retention_ratio) or not 0 < self.retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0, 1]")
        target = ceil(window_length * self.retention_ratio)
        if self.method == "vda_role":
            if sum((self.key_frames, self.overlap_frames, self.new_frames)) != target:
                raise ValueError("Role quotas must sum to the frame budget")
            return target
        if self.temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive")
        if len(range(0, window_length, self.temporal_stride)) != target:
            raise ValueError("Fixed stride must give the same full-window frame budget")
        return target

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_kv_sampling(config: Mapping) -> KVSamplingConfig:
    if (
        "kv_sampling" not in config
        and config.get("inference", {}).get("acceleration", "none") != "none"
    ):
        raise NotImplementedError("Acceleration requires an explicit kv_sampling config")
    return KVSamplingConfig.from_mapping(config.get("kv_sampling"))


def uniform_select(indices: Sequence[int], count: int) -> list[int]:
    """Choose ordered, unique, approximately uniform temporal samples."""
    candidates = list(dict.fromkeys(indices))
    count = min(max(0, count), len(candidates))
    if count == 0:
        return []
    if count == 1:
        return [candidates[(len(candidates) - 1) // 2]]
    denominator = count - 1
    return [
        candidates[
            (2 * index * (len(candidates) - 1) + denominator)
            // (2 * denominator)
        ]
        for index in range(count)
    ]


def eligible_frame_slots(metadata: WindowFrameMetadata) -> list[int]:
    """Return one non-padding slot per source frame, preferring history roles."""
    seen, slots = set(), []
    for role in ("key", "overlap", "new"):
        for slot, (position, frame_role, padding) in enumerate(
            zip(metadata.frame_positions, metadata.frame_roles, metadata.is_padding)
        ):
            if frame_role == role and not padding and position not in seen:
                seen.add(position)
                slots.append(slot)
    return sorted(slots)


def select_vda_role_kv_frames(
    metadata: WindowFrameMetadata,
    budget: int,
    role_budgets: Mapping[str, int],
) -> list[int]:
    candidates = eligible_frame_slots(metadata)
    target = min(max(0, budget), len(candidates))
    temporal = lambda slots: sorted(
        slots, key=lambda slot: (metadata.frame_positions[slot], slot)
    )
    if metadata.first_window:
        return sorted(uniform_select(temporal(candidates), target))
    selected = []
    for role in ("key", "overlap", "new"):
        available = temporal(
            [slot for slot in candidates if metadata.frame_roles[slot] == role]
        )
        selected.extend(
            uniform_select(available, min(role_budgets[role], target - len(selected)))
        )
    remaining = temporal([slot for slot in candidates if slot not in selected])
    selected.extend(uniform_select(remaining, target - len(selected)))
    return sorted(selected)


def select_kv_frames(
    metadata: WindowFrameMetadata, config: KVSamplingConfig, budget: int
) -> list[int]:
    """Select one shared provider set for the retained F/G policies."""
    if not config.enabled:
        return list(range(len(metadata.frame_positions)))
    if config.method == "vda_role":
        selected = select_vda_role_kv_frames(
            metadata,
            budget,
            {
                "key": config.key_frames,
                "overlap": config.overlap_frames,
                "new": config.new_frames,
            },
        )
    elif config.method == "spark3r_fixed_stride":
        candidates = eligible_frame_slots(metadata)
        target = min(budget, len(candidates))
        selected = [
            slot for slot in candidates if slot % config.temporal_stride == 0
        ][:target]
        remaining = [slot for slot in candidates if slot not in selected]
        selected = sorted(
            selected + uniform_select(remaining, target - len(selected))
        )
    else:
        raise ValueError("Shared provider selection does not handle " + config.method)
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("K/V provider slots must be nonempty and unique")
    return selected
