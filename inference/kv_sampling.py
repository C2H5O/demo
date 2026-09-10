"""Frame selection for VDA-style inference windows.

Window slots, sequence positions, dataset frame IDs, roles and padding are
separate concepts. Highlight policies accept pixel ratios, never depth or GT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class WindowFrameMetadata:
    """Provenance in input-window order, before DA3's reference permutation."""

    window_id: int
    frame_positions: tuple[int, ...]
    absolute_frame_ids: tuple[int, ...]
    frame_roles: tuple[str, ...]
    is_padding: tuple[bool, ...]
    first_window: bool
    absolute_id_source: str = "sequence_position"

    def __post_init__(self):
        size = len(self.frame_positions)
        if not size or any(len(value) != size for value in (
            self.absolute_frame_ids, self.frame_roles, self.is_padding
        )):
            raise ValueError("Window metadata fields must have the same nonzero length")
        if any(role not in {"key", "overlap", "new"} for role in self.frame_roles):
            raise ValueError("Unknown VDA frame role")
        if self.first_window and any(role != "new" for role in self.frame_roles):
            raise ValueError("First window must not invent previous-frame roles")
        if any(position < 0 for position in self.frame_positions):
            raise ValueError("Negative sequence position")


@dataclass(frozen=True)
class KVSamplingConfig:
    enabled: bool = False
    method: str = "vda_role"
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
    highlight_detection: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value=None):
        """Read one explicit policy; disabled policies leave the dense path intact."""
        if isinstance(value, cls):
            return value
        value = dict(value or {})
        role = dict(value.pop("vda_role", {}))
        first = dict(value.pop("first_window", {}))
        stride = dict(value.pop("spark3r_fixed_stride", {}))
        # Unknown keys fail instead of silently enabling an old selection scheme.
        for section, allowed in ((role, {"key_frames", "overlap_frames", "new_frames"}),
                                 (first, {"method", "num_frames"}),
                                 (stride, {"temporal_stride"})):
            if set(section) - allowed:
                raise ValueError("Unknown KV sampling fields: " + str(set(section) - allowed))
        return cls(**value, **role,
                   **{"first_window_" + key: val for key, val in first.items()},
                   **stride)

    def frame_budget(self, window_length: int) -> int:
        """Resolve the shared F/G budget once from the real inference window size."""
        if not self.enabled:
            return window_length
        if self.method not in {"vda_role", "vda_role_highlight", "spark3r_fixed_stride"}:
            raise ValueError("Unknown kv_sampling.method")
        if not isfinite(self.retention_ratio) or not 0 < self.retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0, 1]")
        counts = (self.key_frames, self.overlap_frames, self.new_frames,
                  self.first_window_num_frames, self.temporal_stride, self.debug_max_windows)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Frame budgets, stride and debug limit must be nonnegative integers")
        target = ceil(window_length * self.retention_ratio)
        first_method = "highlight" if self.method == "vda_role_highlight" else "uniform"
        if self.first_window_method != first_method or self.first_window_num_frames != target:
            raise ValueError("First-window method and budget must match the KV policy")
        if sum((self.key_frames, self.overlap_frames, self.new_frames)) != target:
            raise ValueError("Role quotas must sum to the shared retention budget")
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")
        if self.method == "vda_role_highlight" and (
            window_length != 32 or target != 16
            or (self.key_frames, self.overlap_frames, self.new_frames) != (2, 8, 6)
        ):
            raise ValueError("Highlight policy requires 32 frames and exactly 2+8+6=16 KV")
        if self.method == "spark3r_fixed_stride" and len(range(0, window_length, self.temporal_stride)) != target:
            raise ValueError("Fixed stride must give the same full-window frame budget as G")
        return target

    def as_dict(self):
        return asdict(self)


def resolve_kv_sampling(config: Mapping) -> KVSamplingConfig:
    """Keep legacy unimplemented accelerators fail-closed, never label them dense."""
    if "kv_sampling" not in config and config.get("inference", {}).get("acceleration", "none") != "none":
        raise NotImplementedError("Legacy acceleration needs an explicit kv_sampling configuration")
    return KVSamplingConfig.from_mapping(config.get("kv_sampling"))


def uniform_select(indices: Sequence[int], count: int) -> list[int]:
    """Choose nearest evenly spaced endpoints using integer round-half-up.

    One requested item uses the lower middle. If candidates are insufficient,
    return all of them, once. Candidate order defines the temporal interval.
    """
    candidates = list(dict.fromkeys(indices))
    count = min(max(0, count), len(candidates))
    if not count:
        return []
    if count == 1:
        return [candidates[(len(candidates) - 1) // 2]]
    denominator = count - 1
    return [candidates[(2 * j * (len(candidates) - 1) + denominator) // (2 * denominator)]
            for j in range(count)]


def eligible_frame_slots(metadata: WindowFrameMetadata) -> list[int]:
    """One non-padding slot per source frame; prefer key, then overlap, then new.

    Sequence positions identify duplicate inputs. Dataset IDs are kept for audit
    and are not used as slot indices or to infer roles.
    """
    seen, slots = set(), []
    for role in ("key", "overlap", "new"):
        for slot, (position, frame_role, padding) in enumerate(zip(
            metadata.frame_positions, metadata.frame_roles, metadata.is_padding
        )):
            if frame_role == role and not padding and position not in seen:
                seen.add(position)
                slots.append(slot)
    return sorted(slots)


def select_vda_role_kv_frames(metadata: WindowFrameMetadata, budget: int,
                              role_budgets: Mapping[str, int]) -> list[int]:
    """Sample each role's full interval, then uniformly fill unused budget.

    The first window is sampled as one continuous interval. Short categories
    contribute all available unique frames. Spare quota goes to the union of
    unselected candidates in temporal (sequence-position) order.
    """
    candidates = eligible_frame_slots(metadata)
    target = min(max(0, budget), len(candidates))
    ordered = lambda slots: sorted(slots, key=lambda slot: (metadata.frame_positions[slot], slot))
    if metadata.first_window:
        return sorted(uniform_select(ordered(candidates), target))
    selected = []
    for role in ("key", "overlap", "new"):
        available = ordered([slot for slot in candidates if metadata.frame_roles[slot] == role])
        selected.extend(uniform_select(available, min(role_budgets[role], target - len(selected))))
    remainder = ordered([slot for slot in candidates if slot not in selected])
    selected.extend(uniform_select(remainder, target - len(selected)))
    return sorted(selected)


def select_kv_frames(metadata: WindowFrameMetadata, config: KVSamplingConfig,
                     budget: int, highlight_scores: Mapping[int, float] | None = None) -> list[int]:
    """Dispatch frame selection with an identical unique-frame cap for F and G."""
    if not config.enabled:
        return list(range(len(metadata.frame_positions)))
    if config.method == "vda_role_highlight":
        config.frame_budget(len(metadata.frame_positions))
        if budget != 16:
            raise ValueError("Highlight policy requires a 16-frame budget")
        candidates = eligible_frame_slots(metadata)
        new = [slot for slot in candidates if metadata.frame_roles[slot] == "new"]
        scores = highlight_scores or {}
        if any(slot not in scores or not isfinite(scores[slot]) or not 0 <= scores[slot] <= 1
               for slot in new):
            raise ValueError("Every eligible new frame requires a finite highlight pixel ratio in [0,1]")
        # First window has no historical roles. Tail padding/duplicates never
        # become references; retain all feasible history and up to six new.
        history = [slot for slot in candidates if metadata.frame_roles[slot] != "new"]
        if not metadata.first_window and (
            sum(metadata.frame_roles[slot] == "key" for slot in history) > 2
            or sum(metadata.frame_roles[slot] == "overlap" for slot in history) > 8
        ):
            raise ValueError("Highlight policy expects at most 2 key and 8 overlap frames")
        count = budget if metadata.first_window else config.new_frames
        selected_new = sorted(new, key=lambda slot: (scores[slot], slot))[:count]
        selected = sorted(history + selected_new)
        if len(candidates) == 32 and not metadata.first_window:
            assert len(history) == 10 and len(selected_new) == 6 and len(selected) == 16
    elif config.method == "vda_role":
        selected = select_vda_role_kv_frames(metadata, budget, {
            "key": config.key_frames, "overlap": config.overlap_frames, "new": config.new_frames,
        })
    elif config.method == "spark3r_fixed_stride":
        candidates = eligible_frame_slots(metadata)
        target = min(budget, len(candidates))
        # Temporal stride is over original VDA input slots, never the DA3
        # reference-permuted tensor. Tail fallback does not repeat padding.
        selected = [slot for slot in candidates if slot % config.temporal_stride == 0][:target]
        remaining = [slot for slot in candidates if slot not in selected]
        selected = sorted(selected + uniform_select(remaining, target - len(selected)))
    else:
        raise ValueError("Unknown KV sampling method: " + config.method)
    expected = min(budget, len(eligible_frame_slots(metadata)))
    if config.method == "vda_role_highlight" and not metadata.first_window:
        expected = len(history) + min(config.new_frames, len(new))
    if len(selected) != expected or len(set(selected)) != expected or not selected:
        raise RuntimeError("KV selection must reach the feasible unique-frame budget")
    if any(slot < 0 or slot >= len(metadata.frame_positions) for slot in selected):
        raise RuntimeError("KV frame slot out of bounds")
    if selected != sorted(selected):
        raise RuntimeError("KV frames must remain in original window order")
    return selected
