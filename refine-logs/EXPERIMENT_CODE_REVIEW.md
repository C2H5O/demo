# Experiment B code review

Reviewer: independent Codex sub-agent, GPT-5.6-Sol (`xhigh`)

Verdict: **no blocking issues remain**.

## Verified

- VGGT-Omega layers 4/11/17/23 are inter-frame global blocks. The capture
  removes 17 camera/register tokens per frame and records the post-QK-norm Q/K
  that feed the no-RoPE global attention path.
- DA3-Small layers 5/7/9/11 are the odd global blocks after `alt_start=4`.
  Capture applies the real global RoPE, restores original frame order after
  reference-view permutation, and removes the single special token.
- The cache schema validates all four layers, Q/K shape and dtype, frame/grid
  metadata, and finite values. Old caches fail closed.
- Runtime 64x80 to 32x40 alignment uses exact overlap-equivalent average
  pooling; non-integer grids use separable 1-D overlap matrices. No dense
  1280x5120 projection is constructed at runtime.
- The loss computes per-head softmax relations, mean head aggregation,
  directed adjacent-frame pairs, stable JS/KL, query chunking, and an equal
  mean over the four configured layer mappings.
- Disabled mode creates no attention capture hooks or loss object. Resume
  validation includes all objective and numerical settings.
- Dry-run directly differentiates `L_attention` with respect to trainable DA3
  backbone parameters and requires a finite non-zero result.

## Validation observed by reviewer

- Pinned real DA3 source tiny smoke: layers 5/7/9/11 each produced
  `[1,3,6,4,64]` Q/K after norm and RoPE in original frame order; backward was
  non-zero.
- Repository tests: `67 passed, 1 skipped` in the default environment.
- `git diff --check`: clean apart from platform line-ending notices.

## Non-blocking deployment risks

- A native Teacher cache plus batch-16 CUDA dry run was not run on the local
  8 GiB GPU. It must be performed on the target high-memory machine.
- Batch 16 can transiently require about 40 GiB host memory during collation;
  one Teacher layer's Q+K transfer is about 5 GiB, in addition to Student
  activations and the autograd graph.
- Teacher capture was verified against the exact local VGGT-Omega source, but
  the configured checkpoint is not present under this repository's
  `checkpoints/` directory, so the full Teacher runtime hook was not exercised.

## Online Teacher revision — 2026-09-04

Requirement change: depth/confidence/camera supervision remains in the original
baseline NPZ cache, while VGGT-Omega Q/K and the attention relation loss are
computed online without Teacher gradients.

Fresh-agent verdict: **APPROVE; no blocking issues remain.**

Verified in the final live tree:

- The training dataset never opens attention Q/K keys. It reads baseline
  pseudo-labels from `teacher_cache_crossclip_base_raw_448x560` and decodes the
  same `ClipRecord`'s native 1024x1280 Teacher RGB separately.
- The frozen Teacher is in eval mode, all parameters have `requires_grad=False`,
  its forward is wrapped in `torch.no_grad`, and captured Q/K are detached.
- `forward_attention` calls only the aggregator. It clears
  `cached_layer_indices` during the call and restores them in `finally`, so
  depth/camera heads and their large cached input features are skipped. The
  dedicated online instance unloads those heads after strict checkpoint loading
  and before GPU transfer; offline cache generation retains the full model.
- Offline cache generation still stages Q/K on CPU; online training explicitly
  retains detached FP16 Q/K on the source GPU, avoiding a hidden PCIe round trip.
- Teacher chunks are weighted by sample count. This is exactly equivalent to
  the full batch because frame count, grid and directed pair count are fixed.
- Disabled baseline mode loads neither native Teacher RGB nor the Teacher model.
- Legacy checkpoints with disabled/missing attention fields remain resumable;
  enabled cached-attention checkpoints correctly fail the changed contract.

Independent checks:

- Default environment: `70 passed, 1 skipped` at review time.
- Real VGGT-Omega tiny runtime: layers 4/11/17/23 returned detached FP16 Q/K on
  `cuda:0`; aggregator cached-layer settings were restored after the call.
- Real Teacher Q/K through JS loss produced finite positive per-layer losses and
  non-zero Student Q/K gradients.

Operational caveat: non-reentrant checkpointing retains aligned frozen Teacher
Q/K until backward, approximately 320 MiB per clip or 5 GiB at batch 16, in
addition to Student activations and both models' weights. A target-GPU dry run
is required before formal training.
