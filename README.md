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
and adds only online frozen-Teacher attention features and their loss.  The
executable config is `configs/vggtoda3_attention_distill.yaml`.

The exact attention paths in the pinned architectures are:

- VGGT-Omega aggregator blocks `4, 11, 17, 23`: all four are inter-frame
  `global` blocks (the register-only blocks are `2, 6, 9, 14, 20`).  Q/K are
  captured after `q_norm`/`k_norm`; this global path receives no RoPE.  Camera
  and register tokens are removed, leaving detached FP16 tensors with shape
  `[16, 16, 5120, 64]` per Q or K and layer (64x80 patch grid).
- DA3-Small DINOv2 blocks `5, 7, 9, 11`: `alt_start=4` makes these the four
  odd-indexed global blocks.  Q/K are captured after Q/K norm and the actual
  global RoPE, the reference-view permutation is restored, and the camera/CLS
  token is removed.  The runtime shape is `[B, 16, 6, 1280, 64]` per Q or K
  and layer (32x40 patch grid).

Teacher Q/K are generated online from the same clip's native 1024x1280 RGB.
The frozen Teacher runs in `eval` and `torch.no_grad`, and only its aggregator
executes: depth and camera heads are skipped because those labels still come
from the Experiment A cache. Teacher patch Q/K are projected from 64x80 to the
Student 32x40 grid using normalized patch-overlap area on the common resized
image extent. No special tokens are aligned and no Q/K or NxN matrix is read
from or written to cache. For every valid directed
adjacent-frame pair (`t -> t-1`, `t -> t+1`), the loss computes each head's
`softmax(QK^T / sqrt(d) / temperature)`, averages probabilities over heads,
and compares Teacher and Student with Jensen-Shannon divergence.  Query tokens
are chunked and checkpoint-recomputed during backward rather than materializing
the complete clip relation matrix.

The implemented objective is:

```text
L_total = L_baseline + 0.1 * mean(L_4_5, L_11_7, L_17_9, L_23_11)
```

The pseudo-label source is the existing baseline cache:

```text
/public/home/2024141520249/Documents/Projects/vggtofast3r/data/
  teacher_cache_crossclip_base_raw_448x560
```

Training reads `depth`, `confidence`, `valid_mask`, `intrinsics`, `extrinsics`
and clip identity from those files. `online_teacher_batch_size: 1` bounds live
Teacher Q/K to one clip; each chunk is consumed immediately by the relation
loss and released. Online Q/K remain detached FP16 tensors on the Teacher GPU,
so there is no hidden GPU-to-CPU-to-GPU round trip. The aggregator's prediction-
head feature cache is disabled only during this attention-only forward and then
restored; the dedicated online Teacher also unloads its camera/depth/text heads
after strict checkpoint loading and before GPU transfer. DataLoader
workers/prefetch remain disabled because native Teacher RGB is decoded in
addition to Student RGB. At scientific batch size 16, native
float32 Teacher RGB itself is 3.75 GiB; collation can transiently require roughly
twice that host memory while per-sample and stacked tensors coexist. Autograd
checkpointing retains the smaller 32x40-aligned frozen Teacher Q/K until
backward (about 320 MiB per clip, or about 5 GiB at batch 16), in addition to
Student activations and the resident frozen Teacher/Student weights. Run
`--dry-run` on the target training GPU before starting epochs.

Audit the baseline pseudo-label cache, then run the one-batch gradient sanity
check or training. Experiment B does not require attention-cache generation:

```bash
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
one trainable DA3 backbone parameter. It also checks that every online Teacher
Q/K tensor is detached. Experiment A remains unchanged and is reproduced with:

```bash
python train_direct_teacher_distillation.py \
  --config configs/vggtoda3.yaml
```

## Required assets

```text
checkpoints/da3-small/config.json
checkpoints/da3-small/model.safetensors
checkpoints/vggt_omega/vggt_omega_1b_512.pt
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
