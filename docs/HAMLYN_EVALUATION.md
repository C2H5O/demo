# Hamlyn zero-shot evaluation

This branch evaluates the fixed 22-sequence Hamlyn split with one spatial
protocol shared by Ours, official DA3-Small, EndoDAV, and Endo3R. Sequence 9 is
included. No split text file or pre-generated `gt_depths.npz` is required.

## Server workflow

```bash
git clone -b eval_H https://github.com/C2H5O/demo.git
cd demo

git clone https://github.com/Zanue/EndoDAV.git external/EndoDAV
git clone https://github.com/wrld/Endo3R.git external/Endo3R

# Copy the required model weights to the locations below.
bash scripts/eval_all.bash
```

`scripts/eval_all.bash` performs a fail-fast preflight before loading any
model. It verifies the Hamlyn root and all 22 sequences, prints the selected
RGB/GT directories and frame statistics, checks every required checkpoint and
external source file, probes each Python environment, and requires CUDA.

## Default data and checkpoint layout

```text
/public/home/2024141520249/Documents/datasets/vggtodistilldata/Hamlyn/
  rectified01/image01/*.jpg
  rectified01/depth01/*.png
  ...

checkpoints/
  eval_H/ours.pt
  da3-small/model.safetensors
  da3-small/config.json

external/
  EndoDAV/
    ckpts/depth_model.pth
    ckpts/pretrained_model/video_depth_anything_vits.pth
  Endo3R/
    checkpoints/endo3r.pth
    checkpoints/raft-things.pth
```

Camera 01 is the formal monocular evaluation stream; sibling `image02/` and
`depth02/` directories are intentionally not evaluated. Native RGB frames may
be `.jpg`/`.jpeg` (and prepared `.png` is also accepted); GT remains strict
uint16 `.png`. The discovery code
also supports `rectifiedNN/color + depth` and the prepared `croppedNN/` plus
`depth_croppedNN/` layout. It matches RGB and GT by numeric frame ID and
rejects missing sequences, duplicate IDs, unequal RGB/GT ID sets, non-2D GT,
non-`uint16` GT, and sequences without valid GT pixels.

All machine-specific paths can be overridden without editing source:

```bash
export HAMLYN_ROOT=/path/to/Hamlyn
export OURS_CHECKPOINT=/path/to/ours.pt
export DA3_CHECKPOINT_DIR=/path/to/da3-small
export ENDODAV_REPOSITORY=/path/to/EndoDAV
export ENDODAV_CHECKPOINT=/path/to/depth_model.pth
export ENDODAV_PRETRAINED_PATH=/path/to/directory/containing/video_depth_anything_vits.pth
export ENDO3R_REPOSITORY=/path/to/Endo3R
export ENDO3R_CHECKPOINT=/path/to/endo3r.pth
export ENDO3R_RAFT_CHECKPOINT=/path/to/Endo3R/checkpoints/raft-things.pth
export CUDA_VISIBLE_DEVICES=0
```

Endo3R reads RAFT from `./checkpoints/raft-things.pth` relative to its
repository working directory. Therefore `ENDO3R_RAFT_CHECKPOINT` must resolve to
that exact location; changing `ENDO3R_REPOSITORY` is the normal way to relocate both
Endo3R weights together.

The default output root can be changed with `HAMLYN_OUTPUT_ROOT` and the
logical model device with `HAMLYN_DEVICE` (default `cuda:0`).

## Python environments

Ours and DA3 share the repository environment. EndoDAV and Endo3R may use
separate environments:

```bash
export OURS_PYTHON=/path/to/ours/bin/python
export ENDODAV_PYTHON=/path/to/endodav/bin/python
export ENDO3R_PYTHON=/path/to/endo3r/bin/python
bash scripts/eval_all.bash
```

`eval_all.bash` remains the only evaluation command; it dispatches each method
to the configured interpreter.

## Fixed inference protocol

- **Ours:** direct PIL bicubic RGB resize to 224x280, then the baseline-J
  `infer_student_video()` implementation. It keeps the existing 32-frame
  windows, reference/overlap selection, anchor disparity scale-shift,
  blending, and full-sequence stitching.
- **Official DA3-Small:** the identical RGB preprocessing, sequence order, and
  `infer_student_video()` path as Ours. Only model parameters differ.
- **EndoDAV:** direct PIL bicubic RGB resize to 224x280, followed by exactly one
  official `model.infer_video_depth(frames)` call for each complete sequence.
  There is no external VDA window. Official normalized disparity is converted
  with the official 0.1-150 m `disp_to_depth` mapping.
- **Endo3R:** the unmodified official `demo.py` subprocess with
  `--resolution 320 --kf_every 1 --save_result`, which produces 256x320 depth
  for every frame. Predicted depth is bilinearly resized to 224x280 only at the
  common evaluation boundary.

The official checkouts are read-only adapters and are ignored by Git. The
evaluation code never edits either external repository.

## Common evaluation protocol

GT is loaded as a 2D `uint16` PNG in millimeters, converted to meters with
`0.001`, and nearest-neighbor resized to 224x280. Valid pixels satisfy
`0.001 < GT < 0.300` m. Predictions are represented as disparity on the common
grid. Once per complete sequence, all valid pixels are collected and one
float64 least-squares fit solves

```text
GT disparity ~= scale * predicted disparity + shift
```

AbsRel, linear RMSE, and delta1 are computed per valid frame, averaged within
each sequence, then macro-averaged over exactly 22 sequences. There is no
per-image median scaling, no frame-weighted overall mean, and no TAE.

## Outputs and recovery

```text
outputs/hamlyn_eval/
  ours/predictions/sequence_NN/{frame_*.npy,metadata.json}
  ours/evaluation.json
  da3/...
  endodav/...
  endo3r/...
  logs/
  preflight.json
  summary.json
  summary.csv
```

A sequence cache is reused only when method, sequence ID, complete frame-ID
list, representation, inference resolution, prediction resolution, evaluation
resolution, file count, dtype, and shape all match. To rebuild all predictions:

```bash
FORCE_INFERENCE=1 bash scripts/eval_all.bash
```

For debugging only, inference and scoring may be separated:

```bash
python evaluate_hamlyn.py --method ours --stage infer
python evaluate_hamlyn.py --method ours --stage evaluate
```
