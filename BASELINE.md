# Baseline H

Branch: `baseline-h`, created from G's implementation at `18b5640`.

Selected configuration: `configs/baselines/H.yaml` (also `configs/baseline.yaml`).

Training required: False. Implementation complete; runtime validation pending.

H uses the existing attention-distillation checkpoint from the
`feature/attention-distillation` run. It changes inference-time K/V selection
only; training objectives, model weights, Q projection, heads, VDA stitching,
evaluation metrics and output geometry are unchanged.

## Standard later-window policy

`inference/student_video.py` constructs each standard later VDA window as:

```text
32 inputs = 2 key + 8 overlap + 22 new
```

The roles and absolute frame IDs are recorded in `WindowFrameMetadata`; the KV
sampler does not infer them from an attention tensor. All eligible key and
overlap frames remain providers. The 22 temporally ordered new frames are split
into seven contiguous buckets with integer boundaries, producing sizes
`[3,3,3,3,3,3,4]` for a full window. Every bucket contributes its two lowest
`lightweight_highlight` scores. Stable ties prefer the earlier frame and the
final selected list remains in temporal order.

```text
2 key + 8 overlap + 14 selected new = 24 patch-KV provider frames
8 unselected new = query/output plus special-token K/V, but no patch K/V
```

Tail padding and duplicate source positions are filtered by the existing
eligibility rules. A short tail keeps up to two eligible frames from every
nonempty bucket and does not duplicate or redistribute frames.

The lightweight score remains the batched Torch-only proxy in
`inference/lightweight_highlight.py`: FP32 average pooling followed by the ratio
of pixels whose maximum RGB value is at least `0.90` and saturation is at most
`0.20`. The complete eligible-new batch stays on its input device; only the
small score vector is copied to the host once per window. PC-Depth, DINO, an
extra descriptor network and per-frame detector calls are absent from this path.

## First window

The first window has 32 real `new` roles and no fabricated key/overlap history.
It retains H's existing frame selection: split into 16 contiguous buckets and
take one lowest-score frame per bucket. The selected 16 providers use their real
`new` role, so block 5 applies stride 2 and blocks 7/9/11 retain all of their
patches. All 32 frames remain queries and outputs.

## Spatial patch K/V schedule

The checked DA3-Small configuration has 12 encoder blocks. With `alt_start=4`,
the flattened global-attention blocks are the real encoder block indices
`[5,7,9,11]`. The configured `0-5` / `6-11` schedule therefore maps to:

```text
encoder block 5:
  key stride 1; overlap stride 2; selected-new stride 2

encoder blocks 7, 9, 11:
  key stride 1; overlap stride 1; selected-new stride 1
```

Stride 2 uses original row-major patch indices from `grid[::2, ::2]`. Tokens
retain the positions already applied by upstream normalization/RoPE; selected
patches are not renumbered. `frame_slots_to_token_indices` maps original VDA
slots through DA3's reference-view permutation and prepends every frame's full
special-token range. The same final index tensor gathers K and V immediately
before `torch.nn.functional.scaled_dot_product_attention`, while Q is passed
through unchanged.

For the official 448x560 input and patch size 14, `P=32x40=1280`. The checked
checkpoint header contains one CLS token and no register-token tensor, so `S=1`.
The adapter derives `S` from the instantiated encoder at runtime.

| Window/layers | Dense Q tokens | Patch KV tokens | Special KV tokens | Total KV tokens | Retention |
|---|---:|---:|---:|---:|---:|
| standard later, block 5 | 40992 | 9600 | 32 | 9632 | 0.234973 |
| standard later, blocks 7/9/11 | 40992 | 30720 | 32 | 30752 | 0.750195 |
| first, block 5 | 40992 | 5120 | 32 | 5152 | 0.125683 |
| first, blocks 7/9/11 | 40992 | 20480 | 32 | 20512 | 0.500390 |

The actual SDPA layout is
`Q [B,heads,32*(P+S),head_dim]` and
`K/V [B,heads,layer_sparse_tokens,head_dim]`. SDPA returns the full Q shape, so
the depth and camera heads still receive all 32 frame/token outputs.

## Configuration

`configs/baselines/H.yaml` selects:

```yaml
kv_sampling:
  enabled: true
  method: role_layer_spatial_kv
  retention_ratio: 0.75
  vda_role:
    key_frames: 2
    overlap_frames: 8
    new_frames: 22
    selected_new_frames: 14
  new_frame_selection:
    method: temporal_bucket_highlight
    num_buckets: 7
    keep_per_bucket: 2
    score: lightweight_highlight
    tie_break: frame_index
  first_window:
    method: bucket_highlight
    num_frames: 16
    num_buckets: 16
  spatial_sampling:
    enabled: true
    early_layers: {start: 0, end: 5, key_stride: 1, overlap_stride: 2, new_stride: 2}
    late_layers: {start: 6, end: 11, key_stride: 1, overlap_stride: 1, new_stride: 1}
  special_tokens: {keep_all: true}
  diagnostics: {print_once_per_sequence: true}
```

The first completed window prints one bounded `KV sampling diagnostics` JSON
object for the sequence. It includes the configured standard 24-provider policy,
the observed first-window roles/providers, actual per-block sparse token counts,
and both early/late standard token expectations. Debug configuration additionally
retains selection scores, buckets, absolute IDs and observed SDPA shapes.

The existing synchronized `model_forward_seconds` timer starts before
`begin_window`, so lightweight scoring, score-vector transfer, bucket selection,
token-index construction, K/V gather and model execution remain included.
Diagnostics print after the synchronized model timing. Evaluation metric
definitions are unchanged.

## Checkpoint and validation commands

The configured checkpoint remains:

```text
/public/home/2024141520249/Documents/Projects/vggtoda3/outputs/vggtoda3_attention_distill/last.pt
```

Run later from the baseline-H checkout:

```bash
python -m pytest tests/test_prefix_recent_kv.py tests/test_bucket_highlight_kv.py tests/test_vda_role_kv.py tests/test_highlight_kv.py -q

CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/inference/H_debug.yaml --split test --protocol vda --limit-windows 2

CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/baselines/H.yaml --split test --protocol vda

CUDA_VISIBLE_DEVICES=0 python visualize_crossclip_projection.py --config configs/baselines/H.yaml --source student --split test --sequence-index 0
```

This implementation turn ran Python compilation/config/index checks and
`git diff --check`. It did not run pytest, model inference, training, full VDA
evaluation, visualization, profiling, downloads or GPU debugging.
