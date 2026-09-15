# endodaveval

Independent, adapter-only evaluation of the official
[EndoDAV](https://github.com/Zanue/EndoDAV) model on SCARED datasets 8 and 9.
The official model checkout, checkpoints and dataset stay outside this repo.

## Fixed protocol

EndoDAV keeps its official inference behavior.  Each complete sequence of raw
SCARED RGB frames is passed exactly once to
`model.infer_video_depth(frames)`.  The audited upstream implementation
(`8a9681b43d9b3b1600ba5f389e1ef69120af37cf`) performs its own 32-frame
windowing, overlap, keyframe replacement, alignment and interpolation.  This
adapter never adds an external VDA window.  Its network resolution remains
224 x 280 and official inference upsamples output to the raw input-frame grid.

The model configuration is locked to `vits`, `ssb`, rank 4,
`image_shape=(224, 280)`, no residual blocks, class token enabled, convolution
head disabled, and the official false defaults for inverse sigmoid and output
sigmoid.  Temporal LoRA is enabled exactly when the checkpoint contains the
official `head.motion_modules.*.lora_A/lora_B` weights.  The VDA-S base
checkpoint is loaded by the official constructor.  The documented official
checkpoint metadata (`height`, `width`, and `use_stereo`) is ignored like the
upstream evaluator, while missing model weights or any other unexpected tensor
keys still fail loudly.  No official source file is copied or modified.

For comparison, official normalized disparity is converted with the upstream
`disp_to_depth(..., 0.1, 150.0)` formula.  The resulting depth is converted
back to reciprocal disparity and only then bilinearly resampled to 256 x 320
(H x W).  SCARED `data/depth` ground truth is converted from mm to m and
nearest-neighbor resampled to the same grid.  Valid pixels satisfy
`0.001 < gt < 100.0` m.  One float64 `numpy.linalg.lstsq` disparity scale and
shift is fit over the complete sequence.  AbsRel, linear RMSE and delta1 are
averaged over valid frames, then macro-averaged over sequences.

TAE uses the aligned 256 x 320 depth, adjacent pairs, both directions, VDA
direct-assignment collision behavior, zero for empty directed pairs, and the
fixed `2 * (num_frames - 1)` denominator before multiplying by 100.  SCARED
poses are world-to-camera; translation is scaled mm to m, poses are inverted
to camera-to-world, and relative motion is `inv(T_2_c2w) @ T_1_c2w`.
Intrinsics are scaled from each raw RGB frame directly to 256 x 320.

## Configuration and commands

Copy `configs/scared.example.json` to an ignored local file and fill in the
SCARED root, official EndoDAV checkout, `depth_model.pth`, and the directory
containing `video_depth_anything_vits.pth`.
Keep the example RGB priority (`left`, then `left_finalpass`, then `rgb_data`)
to match Demo and Endo3R discovery; every result records the selected directory
and numeric frame IDs for audit.

```powershell
Copy-Item configs/scared.example.json configs/scared.local.json
$env:PYTHONPATH = "$PWD\src"
python -m endodaveval --config configs/scared.local.json --stage preflight
python -m endodaveval --config configs/scared.local.json --stage all
```

Inference and evaluation can also be separated without changing the protocol:

```powershell
python -m endodaveval --config configs/scared.local.json --stage infer
python -m endodaveval --config configs/scared.local.json --stage evaluate
```

`model_forward_seconds` wraps CUDA-synchronized calls to the official
`model.forward` only.  It excludes RGB disk decode, preprocessing/stitching
outside forward, evaluation resize, GT, metrics, TAE and JSON export.
`sequence_pipeline_seconds` measures the entire official
`infer_video_depth(frames)` call but still excludes disk decode and evaluation.

Result JSON distinguishes the raw model input, internal 224 x 280 network grid,
official upsampled native prediction, and fixed 256 x 320 evaluation grid.

## Lightweight tests

Tests use fake models and deterministic arrays only:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```
