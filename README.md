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
