# Same-clip VGGT-Omega to DA3-Small distillation

The active objective is `direct_teacher_distillation_v1`: one legal
stride-eight 16-frame Student clip is paired with the already-generated raw
VGGT-Omega cache having exactly the same start and absolute frame IDs.

```text
Student RGB C_n [B,16,3,448,560]
  -> official DA3-Small
     -> DINOv2 ViT-S/14 with standard MLP fc1/fc2 LoRA
     -> fully trainable DualDPT depth head
     -> fully trainable CameraDec
     -> depth, K, W2C pose, xyz_local

raw Teacher cache C_n
  -> depth, confidence, valid_mask, K, W2C pose
```

There is no neighboring cache, overlap mapping, reprojection, grid sampling,
or scale alignment. Teacher depth is pixel-aligned pseudo-GT. The loss is:

```text
L = 1.0 L_depth + 0.1 L_camera + 0.01 L_highlight + 0.1 L_smooth
```

`L_depth` is confidence-weighted direct L1 depth with a per-sample uniform
fallback when all valid confidence weights are zero. `L_camera` compares W2C
relative poses `E_i @ inverse(E_0)`, translation direction/log magnitude, and
normalized focal lengths. See
[docs/coordinate_conventions.md](docs/coordinate_conventions.md).

The official DA3 checkpoint is strict-loaded before ray-only modules are
frozen. CameraEnc is retained for strict checkpoint compatibility but excluded
from the optimizer because the real forward uses `cam_token=None`; CameraDec
executes and is supervised by the camera loss.

## Experiment B: cross-frame attention distillation

Experiment B inherits every Experiment A setting from `configs/vggtoda3.yaml`
and adds only frozen-Teacher attention features and their loss.  The executable
config is `configs/vggtoda3_attention_distill.yaml`.

The exact attention paths in the pinned architectures are:

- VGGT-Omega aggregator blocks `4, 11, 17, 23`: all four are inter-frame
  `global` blocks (the register-only blocks are `2, 6, 9, 14, 20`).  Q/K are
  captured after `q_norm`/`k_norm`; this global path receives no RoPE.  Camera
  and register tokens are removed, leaving FP16 tensors with shape
  `[16, 16, 5120, 64]` per Q or K and layer (64x80 patch grid).
- DA3-Small DINOv2 blocks `5, 7, 9, 11`: `alt_start=4` makes these the four
  odd-indexed global blocks.  Q/K are captured after Q/K norm and the actual
  global RoPE, the reference-view permutation is restored, and the camera/CLS
  token is removed.  The runtime shape is `[B, 16, 6, 1280, 64]` per Q or K
  and layer (32x40 patch grid).

Teacher patch Q/K are projected from 64x80 to the Student 32x40 grid using
normalized patch-overlap area on the common resized image extent.  No special
tokens are aligned and no NxN matrix is cached.  For every valid directed
adjacent-frame pair (`t -> t-1`, `t -> t+1`), the loss computes each head's
`softmax(QK^T / sqrt(d) / temperature)`, averages probabilities over heads,
and compares Teacher and Student with Jensen-Shannon divergence.  Query tokens
are chunked and checkpoint-recomputed during backward rather than materializing
the complete clip relation matrix.

The implemented objective is:

```text
L_total = L_baseline + 0.1 * mean(L_4_5, L_11_7, L_17_9, L_23_11)
```

The uncompressed native Teacher Q/K payload is 1.25 GiB per clip in FP16
(4 layers x Q/K x 16 frames x 5120 tokens x 16 heads x 64 dimensions).  The
attention cache therefore uses a separate directory and is never written over
the Experiment A cache.
Experiment B keeps the same scientific batch size (`16`) but sets DataLoader
workers/prefetch to zero so multiple 20 GiB Q/K batches are not queued in host
memory.  Collating one such batch can transiently approach 40 GiB of host
memory because the per-sample tensors and the stacked batch coexist.  One
Teacher layer is transferred to the GPU at a time (5 GiB of Q+K at batch 16),
while all captured Student layers and their autograd graph also remain live;
the native-grid dry run therefore requires a high-memory training GPU and was
not run on the repository-validation machine's 8 GiB GPU.

The NPZ attention extension is flat to remain compatible with the existing
cache format:

```text
attention_schema_version = cross_frame_qk_v1
attention_num_frames = 16
attention_patch_grid_h = 64
attention_patch_grid_w = 80
attention_patch_size = 16
attention_image_height = 1024
attention_image_width = 1280
attention_dtype = float16
attention_qk_stage = post_qk_norm_no_rope
attention_layer_{4,11,17,23}_{q,k} = [16,16,5120,64]
attention_layer_{4,11,17,23}_{layer_index,num_heads,head_dim}
```

Generate and fully audit the independent Experiment B cache, then run the
one-batch gradient sanity check or training:

```bash
python generate_crossclip_teacher_cache.py \
  --config configs/vggtoda3_attention_distill.yaml \
  --split train

python audit_vggtoda3.py \
  --config configs/vggtoda3_attention_distill.yaml \
  --split train \
  --limit 0

python train_direct_teacher_distillation.py \
  --config configs/vggtoda3_attention_distill.yaml \
  --dry-run

python train_direct_teacher_distillation.py \
  --config configs/vggtoda3_attention_distill.yaml
```

The dry run additionally requires finite, non-zero gradients on every captured
Student Q and K and directly checks that `L_attention` alone reaches at least
one trainable DA3 backbone parameter.  A cache without all four Teacher Q/K
pairs fails immediately with an instruction to regenerate it.  Experiment A
remains unchanged and is reproduced with:

```bash
python train_direct_teacher_distillation.py \
  --config configs/vggtoda3.yaml
```

## Required assets

```text
checkpoints/da3-small/config.json
checkpoints/da3-small/model.safetensors
```

The existing Teacher cache root remains configured as
`teacher.raw_cache_root`. Cache format/protocol names remain
`crossclip_local_v1` solely for compatibility with existing `.npz` files.
Training reads only `depth`, `confidence`, `valid_mask`, `intrinsics`,
`extrinsics`, IDs, and scalar identity metadata. Cached XYZ arrays are not
loaded by the training DataLoader.

## Environment

```bash
conda env create --file environment.yml
conda activate vggtomast3r
bash scripts/setup_da3.sh
```

## Prepare, audit, train

```bash
python precompute_highlights.py --config configs/vggtoda3.yaml --split train --workers 4
python audit_vggtoda3.py --config configs/vggtoda3.yaml --split train --limit 5
python audit_vggtoda3.py --config configs/vggtoda3.yaml --split train --limit 0

python train_direct_teacher_distillation.py --config configs/vggtoda3.yaml --dry-run
python train_direct_teacher_distillation.py --config configs/vggtoda3.yaml
python train_direct_teacher_distillation.py \
  --config configs/vggtoda3.yaml \
  --resume outputs/vggtoda3_direct/last.pt
```

An old projection checkpoint cannot resume this objective. New checkpoints
store `objective_protocol: direct_teacher_distillation_v1` and validate the
loss mode, DA3 architecture, LoRA settings, and module trainability before any
optimizer or scheduler state is loaded.

## Evaluate and visualize

```bash
python evaluate_crossclip_projection.py \
  --config configs/vggtoda3.yaml \
  --checkpoint outputs/vggtoda3_direct/last.pt \
  --protocol vda

python evaluate_crossclip_projection.py \
  --config configs/vggtoda3.yaml \
  --checkpoint outputs/vggtoda3_direct/last.pt \
  --protocol endo3r

# Untouched official DA3-Small baseline on raw SCARED datasets 8 and 9.
python evaluate_da3_small_baseline.py \
  --config configs/vggtoda3.yaml \
  --protocol vda

python evaluate_da3_small_baseline.py \
  --config configs/vggtoda3.yaml \
  --protocol endo3r

python visualize_da3_small_baseline.py \
  --config configs/vggtoda3.yaml \
  --clip-index 0

python visualize_crossclip_projection.py \
  --config configs/vggtoda3.yaml \
  --source student \
  --checkpoint outputs/vggtoda3_direct/last.pt \
  --split test \
  --clip-index 0
```

Evaluation names are retained for CLI compatibility; neither evaluation path
implements the removed training projection objective.

## Diagnostics

`metrics.jsonl` records raw and weighted depth/camera/regularization terms,
depth validity and ranges, Teacher confidence, relative camera errors and
focal-length diagnostics. CUDA timing remains available under
`training.timing`; its phases are DataLoader wait, H2D, Student forward, direct
loss, backward, optimizer, and the meaningful DA3 forward sub-phases.

Teacher-cache generation remains available only for explicit maintenance and
is not part of training. It preserves all legacy fields, including XYZ and the
historical `alignment_scale=1` compatibility scalar:

```bash
python generate_crossclip_teacher_cache.py \
  --config configs/vggtoda3.yaml \
  --split train \
  --cache-root ./data/teacher_cache_regenerated_448x560
```
