# VGGT-to-MASt3R V1

## Branch and scientific hypothesis

- Branch: `vggtomast3r`
- Base: `origin/feature/distill3r-student` at `b311d13db39ab6f75499449588b3429350e7a309`
- Question: replacing the single-frame Distill3R dense head with the official
  DUNE encoder, binocular MASt3R decoder, and MASt3R point head should isolate
  whether explicit two-view cross-attention improves depth accuracy, DUNE
  patch/grid artifacts, and pair geometry consistency.

V1 changes the student architecture and cache granularity only. It does not add
descriptor, confidence, pose, camera, temporal, smoothness, normal, global,
local/global-consistency, or sparse-global-alignment losses.

## Pinned official implementation

The project keeps the old `external/Distill3R` baseline and adds HTTPS
submodules:

- `external/MASt3R` at `f5209afc300cec36239a7ac992263f36847bbba0`
- `external/MASt3R/dust3r` at `3cc8c88c413bb9e34c41db0e0eef99c2ee010b12`
- `external/DUNE` at `1e1a111c287b674b7af546e7b74db42255e9bcfa`

No official source is modified. `models/student/official_mast3r.py` is the only
external import/load boundary. It invokes official
`load_dune_mast3r_model`, while replacing its runtime
`torch.hub.load("naver/dune", ...)` call with the pinned local DUNE loader and
local `dune_vitsmall14_448.pth`. This is necessary because the joint checkpoint
intentionally omits DUNE backbone tensors and the official loader otherwise
downloads them at runtime.

## Architecture and parameter policy

```text
448x560 RGB pair in [-1,1]
    -> frozen DUNE-S/14 encoder (32x40 token grid)
    -> official MASt3R binocular decoder (trainable)
    -> official MASt3R DPT point/descriptor/confidence heads (trainable)
    -> expose pointmaps only
```

The checkpoint-compatible descriptor/confidence branches remain present, but
V1 does not supervise or consume them. All DUNE parameters have
`requires_grad=False`, the encoder remains in `eval()` during training, and the
optimizer assertion rejects DUNE parameters. Unused MASt3R image encoder
parameters are frozen; only `decoder_embed`, both decoder block stacks,
`dec_norm`, and both downstream heads are trainable.

## Pair definition and input protocol

- Ordered pair: `(I_t, I_{t+2})`; direction is never shuffled.
- `pair_stride: 2`, `pair_step: 1`, exactly two frames.
- RGB, teacher, student, and GT remain `448x560`; no square crop.
- DUNE patch size is 14, so the token grid is `32x40`.

The old 8-frame dataset, cache, student, trainer, evaluation, and configuration
remain intact.

## Coordinate systems and student output

The existing VGGT-Omega convention is camera-from-world:

```text
X_cam = R X_world + t
```

The adapter exposes:

```python
{
    "pts3d_ref": ...,          # [B,448,560,3], frame A in camera A
    "pts3d_other_in_ref": ..., # [B,448,560,3], frame B in camera A
}
```

For B targets, the exporter computes
`X_world_b = R_b.T @ (X_b - t_b)`, then
`X_a = R_a @ X_world_b + t_a`. It never compares B-local teacher points to a
B-in-A student output.

`pts3d_ref[...,2]` is reference-camera depth. In contrast,
`pts3d_other_in_ref[...,2]` is **not** frame B depth. Full-frame evaluation runs
the reverse pair `(I_b,I_a)` and uses the reverse prediction's
`pts3d_ref[...,2]`.

## Pair cache schema

Default root:
`data/teacher_cache_vggtomast3r_pair2_lora_448x560/{train,test}`.
Base-teacher caches require an explicit, separate `--cache-root`.

Each `vggtomast3r-pair-v1` NPZ contains:

```text
frame_id_a, frame_id_b, frame_name_a, frame_name_b, pair_stride
image_shape, teacher_variant, lora_checkpoint
depth_a, depth_b
xyz_local_a, xyz_local_b, xyz_global_a, xyz_global_b
pts3d_a_in_a, pts3d_b_in_a
confidence_a, confidence_b, valid_mask_a, valid_mask_b
intrinsics_a, intrinsics_b, extrinsics_a, extrinsics_b
coordinate_convention, cache_format_version, metadata_json
```

The reader validates version, frame identity, stride, resolution, and pointmap
shape, and explicitly rejects legacy 8-frame caches.

## Loss

```text
L_total = 1.0 * L_teacher_point + 0.1 * L_SCARED_reference_depth
```

`L_teacher_point` applies the existing average-distance scale convention jointly
to both reference-camera pointmaps, then uses confidence-weighted Charbonnier
XYZ distance. VGGT-Omega confidence is a detached weight only.
`L_SCARED_reference_depth` reuses the existing GT mask, mm-to-m conversion,
valid range, and log-L1 loss, only on `pts3d_ref[...,2]`. Unlike the original
scale-aligned depth helper, V1 disables median scale alignment here. The point
term already normalizes student and teacher independently; aligning the depth
term as well makes the complete objective invariant to student output scale.
That unconstrained direction can drive MASt3R's exponential depth
parameterization to FP32 overflow. Direct metric-depth supervision is therefore
the required scale anchor while retaining exactly the declared two loss terms.
Logs include raw and weighted values for both terms.

## Workflow

From a clean server:

```bash
git clone --recursive --branch vggtomast3r https://github.com/C2H5O/demo.git vggtomast3r
cd vggtomast3r
git submodule update --init --recursive

# Reuse the server's working PyTorch/CUDA environment.
pip install -r requirements.txt
pip install -r requirements-vggtomast3r.txt
bash scripts/download_vggtomast3r_checkpoints.sh
python scripts/verify_vggtomast3r_environment.py

# Place existing project assets at the configured relative paths:
# data/SCARED
# checkpoints/vggt_omega/vggt_omega_1b_512.pt
# checkpoints/teacher_lora/last.pt

python generate_teacher_pair_cache.py --config configs/vggtomast3r_v1.yaml --split train
python generate_teacher_pair_cache.py --config configs/vggtomast3r_v1.yaml --split test

python train_vggtomast3r.py --config configs/vggtomast3r_v1.yaml --dry-run
python train_vggtomast3r.py --config configs/vggtomast3r_v1.yaml
python train_vggtomast3r.py --config configs/vggtomast3r_v1.yaml \
  --resume outputs/vggtomast3r_v1/last.pt

python evaluate_vggtomast3r.py --config configs/vggtomast3r_v1.yaml
python evaluate_vggtomast3r.py --config configs/vggtomast3r_v1.yaml \
  --protocol endo3r
python visualize_vggtomast3r.py --config configs/vggtomast3r_v1.yaml \
  --checkpoint outputs/vggtomast3r_v1/last.pt
```

Evaluation defaults to the existing Video-Depth-Anything protocol: pair
predictions are averaged by source frame, converted from reference-view Z depth
to disparity, globally scale/shift aligned per sequence, and scored with
AbsRel, RMSE, and delta1. It reuses the old evaluator's per-sequence memmap and
two streaming metric passes. The retained `--protocol endo3r` path reports
AbsRel, SqRel, RMSE, RMSE-log, delta1, delta2, and delta3, plus the optional
DUNE 14-pixel patch-boundary diagnostic. Both paths infer the second camera by
reversing the input pair; neither treats `pts3d_other_in_ref[...,2]` as
second-camera depth. Visualization uses one fixed configured depth range for
teacher/student/GT panels and labels PLY/NPZ output as `pair-local /
reference-camera coordinates`.

## Known limitations and V2

- Full teacher cache generation and long training are intentionally not run on
  a local development machine.
- Real SCARED, VGGT-Omega, LoRA, joint MASt3R, and DUNE checkpoint loading must
  be smoke-tested on the training server.
- Pair geometry consistency is represented by both pointmaps in one reference
  frame; an additional scalar metric is optional and does not block V1.
- V1 does not run `sparse_global_alignment`.

Planned V2 only:

```text
8-frame clip -> pair graph -> DUNE-MASt3R pair inference -> MASt3R matching
-> sparse global alignment -> camera poses + global reconstruction
```
