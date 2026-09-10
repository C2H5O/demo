# Training clip selection and attention backend

Training uses 16 consecutive frames. Clip starts are zero-based positions
0, 8, 16, ... within each video (human frame positions 1, 9, 17, ...).
The start grid is independent of source filename IDs and resets for every video.

Teacher caches may still contain every possible start. No cache files are
deleted, rewritten, or regenerated. RGB discovery and cache-metadata fallback
already construct stride-8 candidates; the final DirectTeacherDistillationDataset
also filters candidate metadata before looking up matching cache files.
Missing legal caches are skipped, never replaced by adjacent starts.
Shuffling changes only the order of the legal training sample set.

The training log reports start_stride=8, matched clips, off-stride candidates,
and legal candidates without a cache. When the upstream dataset has already
filtered starts, skipped_off_stride is zero even for a dense cache directory.

When attention_distill.enabled is true, the public training entrypoint disables
the Flash-first model wrappers and allows only efficient SDPA and math SDPA for
the entire training call, including backward/checkpoint recomputation. The
previous backend state is restored on success or exception. A legacy top-level
flash_attention.enabled value is overridden in the effective training config;
inference configuration and query-chunk distillation remain unchanged.

Applied to the actual feature/attention-distillation checkout and training
baseline B/C/D/E worktrees. A/F/G/H are inference-only baselines and do not
require a new training run or checkpoint modification.

Validation uses synthetic dense caches for two videos, different source-ID
origins, missing legal caches, metadata fallback, shuffled sample indices,
and a CPU checkpointed SDPA backward. Real server data and GPU training have
not been run.
