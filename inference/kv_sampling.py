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
    selected_new_frames: int | None = None
    first_window_method: str = "uniform"
    first_window_num_frames: int = 8
    first_window_num_buckets: int | None = None
    temporal_stride: int = 4
    debug: bool = False
    debug_max_windows: int = 2
    profile_attention: bool = False
    highlight_detection: dict = field(default_factory=dict)
    lightweight_highlight: dict = field(default_factory=dict)
    bucket_highlight: dict = field(default_factory=dict)
    new_frame_selection: dict = field(default_factory=dict)
    spatial_sampling: dict = field(default_factory=dict)
    special_tokens: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

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
        for section, allowed in ((role, {"key_frames", "overlap_frames", "new_frames",
                                        "selected_new_frames"}),
                                 (first, {"method", "num_frames", "num_buckets"}),
                                 (stride, {"temporal_stride"})):
            if set(section) - allowed:
                raise ValueError("Unknown KV sampling fields: " + str(set(section) - allowed))
        return cls(**value, **role,
                   **{"first_window_" + key: val for key, val in first.items()},
                   **stride)

    def frame_budget(self, window_length: int, *, first_window: bool = False) -> int:
        """Validate policy and resolve this window's cap; default is the later-window cap."""
        if not self.enabled:
            return window_length
        if self.method not in {"vda_role", "vda_role_highlight", "vda_role_bucket_highlight",
                               "role_layer_spatial_kv", "spark3r_fixed_stride"}:
            raise ValueError("Unknown kv_sampling.method")
        if not isfinite(self.retention_ratio) or not 0 < self.retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0, 1]")
        counts = (self.key_frames, self.overlap_frames, self.new_frames,
                  self.first_window_num_frames, self.temporal_stride, self.debug_max_windows)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Frame budgets, stride and debug limit must be nonnegative integers")
        if self.selected_new_frames is not None and (
            type(self.selected_new_frames) is not int or self.selected_new_frames < 0
        ):
            raise ValueError("selected_new_frames must be a nonnegative integer")
        target = ceil(window_length * self.retention_ratio)
        first_method = {"vda_role_highlight": "highlight",
                        "vda_role_bucket_highlight": "bucket_highlight",
                        "role_layer_spatial_kv": "bucket_highlight"}.get(self.method, "uniform")
        first_target = 16 if self.method in {"vda_role_bucket_highlight",
                                             "role_layer_spatial_kv"} else target
        if self.first_window_method != first_method or self.first_window_num_frames != first_target:
            raise ValueError("First-window method and budget must match the KV policy")
        if (self.method != "role_layer_spatial_kv"
            and sum((self.key_frames, self.overlap_frames, self.new_frames)) != target):
            raise ValueError("Role quotas must sum to the shared retention budget")
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")
        if self.method == "vda_role_highlight" and (
            window_length != 32 or target != 16
            or (self.key_frames, self.overlap_frames, self.new_frames) != (2, 8, 6)
        ):
            raise ValueError("Highlight policy requires 32 frames and exactly 2+8+6=16 KV")
        if self.method == "vda_role_bucket_highlight":
            resolve_lightweight_highlight_options(self.lightweight_highlight)
            resolve_bucket_highlight_options(self.bucket_highlight)
            if (window_length != 32 or target != 20
                or (self.key_frames, self.overlap_frames, self.new_frames) != (2, 8, 10)
                or type(self.first_window_num_buckets) is not int or self.first_window_num_buckets != 16):
                raise ValueError("Bucket highlight requires window32, first16/16 buckets, later2+8+10=20")
            if first_window:
                return self.first_window_num_frames
        if self.method == "role_layer_spatial_kv":
            resolve_lightweight_highlight_options(self.lightweight_highlight)
            resolve_new_frame_selection_options(self.new_frame_selection)
            resolve_spatial_sampling_options(self.spatial_sampling)
            resolve_special_token_options(self.special_tokens)
            resolve_diagnostics_options(self.diagnostics)
            if (window_length != 32 or target != 24
                or (self.key_frames, self.overlap_frames, self.new_frames,
                    self.selected_new_frames) != (2, 8, 22, 14)
                or type(self.first_window_num_buckets) is not int
                or self.first_window_num_buckets != 16):
                raise ValueError(
                    "Role/layer spatial KV requires window32, first16, and later "
                    "2 key + 8 overlap + 14 selected of 22 new = 24 providers"
                )
            if first_window:
                return self.first_window_num_frames
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


def resolve_lightweight_highlight_options(options: Mapping | None = None) -> dict:
    """Validate inference-only frame-ranking proxy options without touching tensors."""
    defaults = {"brightness_threshold": 0.90, "saturation_threshold": 0.20,
                "downsample_factor": 4}
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("lightweight_highlight must be a mapping")
    values = dict(options or {})
    if set(values) - defaults.keys():
        raise ValueError("Unknown lightweight_highlight fields: " + str(set(values) - defaults.keys()))
    values = {**defaults, **values}
    for name in ("brightness_threshold", "saturation_threshold"):
        value = values[name]
        if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + " must be a finite number in [0,1]")
    factor = values["downsample_factor"]
    if type(factor) is not int or factor < 1:
        raise ValueError("downsample_factor must be a positive integer")
    return values


def _exact_mapping(options: Mapping | None, defaults: Mapping, name: str) -> dict:
    if not isinstance(options, Mapping):
        raise ValueError(name + " must be a mapping")
    values = dict(options)
    if set(values) != set(defaults):
        raise ValueError(name + " fields must be exactly " + str(sorted(defaults)))
    return values


def resolve_new_frame_selection_options(options: Mapping | None = None) -> dict:
    """Validate H's fixed seven-bucket, two-clean-frames-per-bucket policy."""
    expected = {"method": "temporal_bucket_highlight", "num_buckets": 7,
                "keep_per_bucket": 2, "score": "lightweight_highlight",
                "tie_break": "frame_index"}
    values = _exact_mapping(options, expected, "new_frame_selection")
    if any(type(values[key]) is not type(expected[key]) or values[key] != expected[key]
           for key in expected):
        raise ValueError("new_frame_selection must use 7 buckets x 2 lightweight-highlight frames")
    return values


def resolve_spatial_sampling_options(options: Mapping | None = None) -> dict:
    """Validate encoder-block schedule; global blocks use their real block index."""
    if not isinstance(options, Mapping):
        raise ValueError("spatial_sampling must be a mapping")
    values = dict(options)
    if set(values) != {"enabled", "early_layers", "late_layers"} or values["enabled"] is not True:
        raise ValueError("spatial_sampling must enable early_layers and late_layers")
    expected_strides = {
        "early_layers": {"key_stride": 1, "overlap_stride": 2, "new_stride": 2},
        "late_layers": {"key_stride": 1, "overlap_stride": 1, "new_stride": 1},
    }
    required_fields = {"start", "end", "key_stride", "overlap_stride", "new_stride"}
    for name, strides in expected_strides.items():
        section = values[name]
        if not isinstance(section, Mapping) or set(section) != required_fields:
            raise ValueError(name + " must define start, end, and all three role strides")
        if any(type(section[key]) is not int for key in required_fields):
            raise ValueError(name + " fields must be integers")
        if any(section[key] != expected for key, expected in strides.items()):
            raise ValueError(name + " must preserve the configured role strides")
        if not 0 <= section["start"] <= section["end"] < 12:
            raise ValueError(name + " must be a nonempty range within DA3-Small blocks [0, 11]")
        values[name] = dict(section)
    early, late = values["early_layers"], values["late_layers"]
    if early["start"] != 0 or late["end"] != 11 or late["start"] != early["end"] + 1:
        raise ValueError(
            "early/late layer ranges must be contiguous, nonoverlapping, and cover [0, 11]"
        )
    return values


def resolve_special_token_options(options: Mapping | None = None) -> dict:
    values = _exact_mapping(options, {"keep_all": True}, "special_tokens")
    if values["keep_all"] is not True:
        raise ValueError("All special-token K/V must be retained")
    return values


def resolve_diagnostics_options(options: Mapping | None = None) -> dict:
    values = _exact_mapping(options, {"print_once_per_sequence": True}, "diagnostics")
    if values["print_once_per_sequence"] is not True:
        raise ValueError("H requires one bounded KV diagnostic per sequence")
    return values


def spatial_strides_for_layer(layer_index: int, options: Mapping) -> dict[str, int]:
    """Return role strides for a real DA3 encoder block index."""
    if type(layer_index) is not int:
        raise ValueError("layer_index must be an integer")
    values = resolve_spatial_sampling_options(options)
    for name in ("early_layers", "late_layers"):
        section = values[name]
        if section["start"] <= layer_index <= section["end"]:
            return {role: section[role + "_stride"] for role in ("key", "overlap", "new")}
    raise ValueError("DA3 encoder block is outside the configured 0-11 schedule")


def build_spatial_patch_indices(grid_height: int, grid_width: int, stride: int) -> tuple[int, ...]:
    """Row-major indices equivalent to grid[::stride, ::stride]."""
    if any(type(value) is not int or value < 1 for value in (grid_height, grid_width, stride)):
        raise ValueError("Patch grid dimensions and stride must be positive integers")
    return tuple(row * grid_width + column
                 for row in range(0, grid_height, stride)
                 for column in range(0, grid_width, stride))


def build_role_layer_patch_indices(metadata: WindowFrameMetadata, selected_slots: Sequence[int],
                                   layer_index: int, grid_height: int, grid_width: int,
                                   options: Mapping) -> dict[int, tuple[int, ...]]:
    """Build original-grid patch offsets for only the selected provider frames."""
    selected = list(selected_slots)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Selected provider slots must be nonempty and unique")
    if any(type(slot) is not int or not 0 <= slot < len(metadata.frame_roles) for slot in selected):
        raise ValueError("Selected provider slot is outside metadata")
    strides = spatial_strides_for_layer(layer_index, options)
    cache = {stride: build_spatial_patch_indices(grid_height, grid_width, stride)
             for stride in set(strides.values())}
    return {slot: cache[strides[metadata.frame_roles[slot]]] for slot in selected}


def temporal_buckets(candidates: Sequence[int], count: int) -> list[list[int]]:
    """Partition already temporally ordered candidates with floor(b*N/B) bounds."""
    if type(count) is not int or count < 0:
        raise ValueError("Bucket count must be a nonnegative integer")
    candidates = list(candidates)
    if len(candidates) != len(set(candidates)):
        raise ValueError("Temporal bucket candidates must be unique")
    size = len(candidates)
    buckets = min(count, size)
    if buckets == 0:
        return []
    return [candidates[b * size // buckets:(b + 1) * size // buckets]
            for b in range(buckets)]


def bucket_keep_count(bucket_size: int) -> int:
    """Later-window quota: empty=0, sizes1..3=1, sizes4+=ceil(size/2)."""
    if type(bucket_size) is not int:
        raise ValueError("bucket_size must be an integer")
    if bucket_size <= 0:
        return 0
    return 1 if bucket_size <= 3 else (bucket_size + 1) // 2


def resolve_bucket_highlight_options(options: Mapping | None = None) -> dict:
    """Validate later-window grouping separately from its ten-new-frame budget."""
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("bucket_highlight must be a mapping")
    defaults = {"num_buckets": 6, "keep_policy": "size_dependent", "keep_counts": None,
                "prefix_frames": None, "prefix_keep": None, "recent_frames": None}
    values = dict(options or {})
    if set(values) - defaults.keys():
        raise ValueError("Unknown bucket_highlight fields: " + str(set(values) - defaults.keys()))
    values = {**defaults, **values}
    if values["keep_policy"] == "prefix_recent":
        expected = {"num_buckets": 2, "prefix_frames": 14, "prefix_keep": 2, "recent_frames": 8}
        if any(type(values[key]) is not int or values[key] != count for key, count in expected.items()):
            raise ValueError("prefix_recent requires two groups: first14 keep2 and final8 keep all")
        if values["keep_counts"] is not None:
            raise ValueError("prefix_recent uses prefix_keep/recent_frames, not keep_counts")
        return values
    if any(values[key] is not None for key in ("prefix_frames", "prefix_keep", "recent_frames")):
        raise ValueError("Prefix/recent fields require keep_policy=prefix_recent")
    if type(values["num_buckets"]) is not int or values["num_buckets"] != 6:
        raise ValueError("Later bucket highlight requires exactly six temporal buckets")
    if values["keep_policy"] not in {"size_dependent", "fixed"}:
        raise ValueError("Later bucket highlight requires keep_policy=size_dependent or fixed")
    counts = values["keep_counts"]
    if values["keep_policy"] == "fixed":
        capacities = [len(bucket) for bucket in temporal_buckets(range(22), values["num_buckets"])]
        if (not isinstance(counts, (list, tuple)) or len(counts) != 6
            or any(type(n) is not int or not 1 <= n <= capacity for n, capacity in zip(counts, capacities))
            or sum(counts) != 10):
            raise ValueError("Fixed bucket keep_counts must fit six standard buckets and sum to ten")
    elif counts is not None:
        raise ValueError("keep_counts requires keep_policy=fixed")
    return values


def select_prefix_recent_frames(metadata: WindowFrameMetadata, scores: Mapping[int, float],
                                options: Mapping) -> tuple[list[int], list[list[int]], list[int]]:
    """Select two clean prefix frames and retain the original final eight new slots.

    Split BEFORE padding/dedup filtering, so a short tail cannot move early frames
    into the mandatory recent group. Final selected slots retain window order.
    """
    new_slots = [slot for slot, role in enumerate(metadata.frame_roles) if role == "new"]
    total_new = options["prefix_frames"] + options["recent_frames"]
    if metadata.first_window or len(new_slots) > total_new:
        raise ValueError("prefix_recent requires at most the later window's 22 original new slots")
    eligible = set(eligible_frame_slots(metadata))
    split = options["prefix_frames"]
    prefix = [slot for slot in new_slots[:split] if slot in eligible]
    recent = [slot for slot in new_slots[split:] if slot in eligible]
    if any(slot not in scores or not isfinite(scores[slot]) or not 0 <= scores[slot] <= 1 for slot in prefix):
        raise ValueError("Every prefix candidate requires a finite highlight pixel ratio in [0,1]")
    selected_prefix = sorted(prefix, key=lambda slot: (scores[slot], slot))[:options["prefix_keep"]]
    return (sorted(selected_prefix + recent), [prefix, recent], [len(selected_prefix), len(recent)])


def bucket_selection_keep_counts(buckets: Sequence[Sequence[int]], keep_policy: str,
                                 keep_counts: Sequence[int] | None = None) -> list[int]:
    """Actual counts after tail clipping; never redistribute or duplicate frames."""
    if keep_policy == "one":
        return [min(1, len(bucket)) for bucket in buckets]
    if keep_policy == "size_dependent":
        return [bucket_keep_count(len(bucket)) for bucket in buckets]
    if keep_policy != "fixed":
        raise ValueError("Unknown bucket keep policy")
    if (not isinstance(keep_counts, (list, tuple)) or len(keep_counts) < len(buckets)
        or any(type(n) is not int or n < 1 for n in keep_counts)):
        raise ValueError("Fixed bucket selection requires positive integer counts for every bucket")
    return [min(len(bucket), keep_counts[b]) for b, bucket in enumerate(buckets)]


def select_bucket_highlight_frames(candidates: Sequence[int], scores: Mapping[int, float],
                                    count: int, *, keep_policy: str = "one",
                                    keep_counts: Sequence[int] | None = None) -> tuple[list[int], list[list[int]]]:
    """Lowest-score frame(s) per temporal bucket; ties prefer the earlier candidate.

    The first window retains the original one-per-bucket rule. Later windows use
    configured counts. Return selected slots in temporal order, not score order.
    """
    buckets = temporal_buckets(candidates, count)
    actual_counts = bucket_selection_keep_counts(buckets, keep_policy, keep_counts)
    if any(slot not in scores or not isfinite(scores[slot]) or not 0 <= scores[slot] <= 1
           for bucket in buckets for slot in bucket):
        raise ValueError("Every bucket candidate requires a finite highlight pixel ratio in [0,1]")
    selected = []
    for bucket, keep in zip(buckets, actual_counts):
        # Stable sort resolves ties in original candidate temporal order.
        chosen = set(sorted(bucket, key=lambda slot: scores[slot])[:keep])
        selected.extend(slot for slot in bucket if slot in chosen)
    return selected, buckets


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
                     budget: int, highlight_scores: Mapping[int, float] | None = None,
                     selection_audit: dict | None = None) -> list[int]:
    """Dispatch frame selection with an identical unique-frame cap for F and G."""
    if not config.enabled:
        return list(range(len(metadata.frame_positions)))
    if config.method in {"vda_role_highlight", "vda_role_bucket_highlight",
                          "role_layer_spatial_kv"}:
        expected_budget = config.frame_budget(len(metadata.frame_positions), first_window=metadata.first_window)
        if budget != expected_budget:
            raise ValueError("Highlight budget does not match the current first/later window")
        candidates = eligible_frame_slots(metadata)
        new = [slot for slot in candidates if metadata.frame_roles[slot] == "new"]
        scores = highlight_scores or {}
        if any(slot not in scores or not isfinite(scores[slot]) or not 0 <= scores[slot] <= 1
               for slot in new):
            raise ValueError("Every eligible new frame requires a finite highlight pixel ratio in [0,1]")
        # First window has no historical roles. Tail padding/duplicates never
        # become references; retain all feasible history and the policy's new set.
        history = [slot for slot in candidates if metadata.frame_roles[slot] != "new"]
        if not metadata.first_window and (
            sum(metadata.frame_roles[slot] == "key" for slot in history) > 2
            or sum(metadata.frame_roles[slot] == "overlap" for slot in history) > 8
        ):
            raise ValueError("Highlight policy expects at most 2 key and 8 overlap frames")
        count = budget if metadata.first_window else (
            config.selected_new_frames if config.method == "role_layer_spatial_kv"
            else config.new_frames
        )
        if config.method in {"vda_role_bucket_highlight", "role_layer_spatial_kv"}:
            temporal_new = sorted(new, key=lambda slot: (metadata.frame_positions[slot], slot))
            options = (resolve_bucket_highlight_options(config.bucket_highlight)
                       if config.method == "vda_role_bucket_highlight" else None)
            if (config.method == "vda_role_bucket_highlight" and not metadata.first_window
                and options["keep_policy"] == "prefix_recent"):
                selected_new, buckets, keep_counts = select_prefix_recent_frames(metadata, scores, options)
            else:
                if metadata.first_window:
                    num_buckets, keep_policy, configured_counts = (
                        config.first_window_num_buckets, "one", None)
                elif config.method == "role_layer_spatial_kv":
                    selection = resolve_new_frame_selection_options(config.new_frame_selection)
                    num_buckets, keep_policy = selection["num_buckets"], "fixed"
                    configured_counts = [selection["keep_per_bucket"]] * num_buckets
                else:
                    num_buckets, keep_policy = options["num_buckets"], options["keep_policy"]
                    configured_counts = options["keep_counts"]
                selected_new, buckets = select_bucket_highlight_frames(
                    temporal_new, scores, num_buckets, keep_policy=keep_policy,
                    keep_counts=configured_counts)
                keep_counts = bucket_selection_keep_counts(buckets, keep_policy, configured_counts)
            if len(selected_new) > (config.first_window_num_frames if metadata.first_window else config.new_frames):
                raise RuntimeError("Bucket selection exceeded this window's new-frame budget")
            if selection_audit is not None:
                selection_audit.update(new_temporal_buckets=buckets, bucket_selected_slots=selected_new,
                                       bucket_keep_counts=keep_counts)
        else:
            selected_new = sorted(new, key=lambda slot: (scores[slot], slot))[:count]
        selected = sorted(history + selected_new)
        if len(candidates) == 32 and not metadata.first_window:
            expected_new = (config.selected_new_frames
                            if config.method == "role_layer_spatial_kv" else config.new_frames)
            if not (len(history) == 10 and len(selected_new) == expected_new
                    and len(selected) == budget):
                raise RuntimeError("Standard VDA window did not satisfy its exact role/provider budget")
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
    if config.method in {"vda_role_bucket_highlight", "role_layer_spatial_kv"}:
        expected = len(history) + sum(keep_counts)
    elif config.method == "vda_role_highlight" and not metadata.first_window:
        expected = len(history) + min(config.new_frames, len(new))
    if len(selected) != expected or len(set(selected)) != expected or not selected:
        raise RuntimeError("KV selection must reach the feasible unique-frame budget")
    if any(slot < 0 or slot >= len(metadata.frame_positions) for slot in selected):
        raise RuntimeError("KV frame slot out of bounds")
    if selected != sorted(selected):
        raise RuntimeError("KV frames must remain in original window order")
    return selected
