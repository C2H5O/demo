# Training Check — 2026-08-22

- Data source: server traceback and `outputs/vggtomast3r_v1/numeric_events.jsonl`
- Run: VGG-to-MASt3R V1, epoch 7, global step 42432
- Decision: STOP and restart the corrected experiment from epoch 0
- Evidence: RGB input was finite in `[-1, 1]`; all parameters were finite;
  BF16 and FP32 forward both failed. `pts3d_ref` was only 76.89% finite and
  reached `3.37e38`; `pts3d_other_in_ref` reached `1.29e21`.
- Root cause: teacher point loss independently normalizes student/teacher
  scale, while median-aligned supervised depth also removes scale. The complete
  objective therefore had no absolute-scale anchor. MASt3R's exponential depth
  parameterization eventually overflowed along that unconstrained direction.
- Correction: retain the same two loss terms, but set the existing SCARED
  supervised-depth term to `scale_alignment=none` so metric GT anchors student
  scale. Reject fully scale-invariant V1 configurations. Preserve bounded FP32
  retry and numeric diagnostics as hard safety checks.
- Integrity note: checkpoints trained under the old scale-invariant objective
  must not be combined with the corrected run for final claims.
