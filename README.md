# VGGT-Omega LoRA Distillation

This independent project implements a two-stage endoscopic reconstruction
pipeline derived selectively from `vggt_omega_distill`, PC-Depth, and EndoDAC.
Those source projects are reference-only and are not imported at runtime.

## Training design

1. Stage one loads pretrained VGGT-Omega, freezes every original backbone and
   output-head parameter, and injects native LoRA only into the MLP/FFN
   projections discovered from the model instance. It then adapts LoRA with
   temporal photometric, geometry, highlight, smoothness, and
   original/inpainted consistency losses.
2. Stage two reloads the pretrained teacher plus its LoRA-only checkpoint,
   freezes the complete teacher, creates or reads offline caches, and trains the
   migrated DUNE ViT-Small student using teacher point maps/confidence,
   SCARED ground-truth depth, geometry consistency, and edge-aware smoothness.
   Cache-backed training never reruns the teacher inside the student loop.

The implemented LoRA branch is
`y = W x + (alpha / rank) B A x`. The pretrained `W` stays frozen, `A` uses
Kaiming initialization, and `B` starts at zero so injection initially preserves
the base output. The supplied configuration follows EndoDAC's ordinary LoRA
branch with `rank=4`, `alpha=1`, no LoRA dropout, and explicitly rejects
DV-LoRA.

## Entry points

```bash
python train_teacher_lora.py --config configs/teacher_lora_finetune.yaml
python train_teacher_lora.py --config configs/teacher_lora_finetune.yaml --dry-run
python generate_teacher_cache.py --config configs/student_distillation.yaml --split train --overwrite
python generate_teacher_cache.py --config configs/student_distillation.yaml --split test --overwrite
python generate_teacher_cache.py --config configs/student_distillation.yaml --split test --base-teacher --cache-root data/teacher_cache_base_448x560 --overwrite
python compare_teacher_caches.py --config configs/student_distillation.yaml --base-cache data/teacher_cache_base_448x560 --finetuned-cache data/teacher_cache_endodac_lora_448x560
python train_student_distillation.py --config configs/student_distillation.yaml
python evaluate.py --config configs/student_distillation.yaml --checkpoint outputs/student_distillation_448x560/last.pt --split test
python evaluate_vda.py --config configs/student_distillation.yaml --checkpoint outputs/student_distillation_448x560/last.pt --split test
```

Install VGGT-Omega itself into the environment as a Python package, for example
with `pip install -e external/vggt-omega`. The project deliberately does not
modify `sys.path` to reach the old repository or an external source checkout.
All dataset, checkpoint, cache, and output paths are local to this project by
default and live in YAML configuration.

`generate_teacher_cache.py --base-teacher` loads only the configured pretrained
VGGT-Omega checkpoint and skips both LoRA injection and LoRA checkpoint loading.
It requires an explicit, separate `--cache-root`; generated NPZ files record
`teacher_variant=base` or `teacher_variant=lora`.

## Main layout

```text
configs/       stage-specific configuration and portable path examples
data/          local datasets and generated teacher caches (contents ignored)
checkpoints/   pretrained and trained weights (contents ignored)
external/      optional local VGGT-Omega source checkout (contents ignored)
datasets/      SCARED discovery, manifests, clips, transforms, and calibration boundary
models/        frozen teacher + LoRA and the migrated DUNE student
losses/        complete teacher self-supervision and student distillation objectives
trainers/      separate teacher-LoRA and student-distillation orchestration
cache/         teacher cache generation and cache-backed dataset API
evaluation/    SCARED depth metrics and sequence evaluation
visualization/ depth, confidence, and point-cloud helpers
utils/         checkpoint, camera, geometry, distributed, logging, seed, and config helpers
scripts/       shell entry points
```

The default local asset layout is:

```text
data/
  SCARED/
  teacher_cache_endodac_lora_448x560/
checkpoints/
  vggt_omega/vggt_omega_1b_512.pt
  teacher_lora/last.pt
  dune/dune_vitsmall14_336.pth
external/
  vggt-omega/                 # optional editable dependency checkout
outputs/                      # generated automatically
```

Only the three README placeholder files are versioned under `data/`,
`checkpoints/`, and `external/`; actual datasets, caches, weights, and external
source trees remain local. Run commands from the project root so the relative
paths in `configs/*.yaml` resolve consistently. This project does not need the
old `vggt_omega_distill`, PC-Depth, or EndoDAC directories at runtime.

## LoRA target discovery

EndoDAC injects ordinary LoRA into every DINO image-encoder block's
`mlp.fc1/fc2`. The corresponding VGGT-Omega runtime paths are:

```text
aggregator.patch_embed.blocks.<index>.mlp.fc1/fc2
```

The temporal `frame_blocks` and `inter_frame_blocks`, Q/K/V projections, and all
camera, dense depth/confidence, point, pose, and alignment heads are excluded.
Initialization raises immediately if any non-LoRA teacher parameter remains
trainable.

This matches EndoDAC's placement on Transformer `mlp.fc1/fc2`, while preserving
VGGT-Omega's already-loaded base linear weights instead of reconstructing them.

## Teacher self-supervision

The SCARED clip dataset optionally runs a robust PC-Depth-inspired highlight
processor and returns:

```text
images               ImageNet-normalized RGB for existing student behavior
highlight_masks      binary [T,1,H,W] masks
inpainted_images     zero-one locally repaired RGB
```

Teacher training converts RGB back to `[0,1]`, predicts depth and camera
parameters for the whole clip, and computes:

```text
L_teacher =
    λ_photo     L_SSIM+L1_reprojection
  + λ_geometry  L_temporal_depth_consistency
  + λ_highlight L_surface_normal_highlight
  + λ_smooth    L_inpainted_edge_aware_smoothness
  + λ_inpaint   L_original/inpainted_prediction_consistency
```

Temporal warping uses VGGT-Omega's camera-from-world convention. It supports
automasking, dynamic-region weighting, minimum reprojection, projected
highlight exclusion, and PC-Depth spatial light-source correction. Frozen
teacher modules remain in evaluation mode during training; only LoRA modules
enter training mode.

Following EndoDAC's ordinary LoRA placement, the supplied configuration adapts
`mlp.fc1/fc2` in every discovered image-encoder block with
`alpha/rank = 1/4`. It uses a `1e-5` learning rate and writes to
`outputs/teacher_lora_endodac_lora`. Its offline cache is written separately
under `data/teacher_cache_endodac_lora_448x560`. Checkpoints created by the removed
anchor-loss version use a different LoRA tensor layout and must not be resumed.

## Student supervision

The configured spatial contract is `H x W = 448 x 560` throughout:

```text
SCARED RGB 1024x1280 -> aspect-preserving VGGT-Omega max_size resize
teacher input/output/cache 448x560
SCARED RGB 1024x1280 -> student resize 448x560
student output, teacher targets, and training GT 448x560
VDA student/teacher/GT evaluation 448x560
```

The teacher cache exporter rejects any input or output that is not exactly
448x560, and cache-backed training rejects stale caches of another resolution
instead of silently resizing them. `student.image_size: 336` remains the
pretrained DUNE positional-embedding base grid; runtime inputs use the dataset
448x560 shape and interpolate that embedding from 24x24 to 32x40 patches.

The cache-backed dataset aligns SCARED ground-truth depth to RGB by numeric
frame ID. PNG, TIFF, and NPY depth are supported, including configurable channel
and scale. SCARED `depthmap_rectified` values are converted from millimetres to
metres with `dataset.ground_truth.scale: 0.001` before supervised loss. The
student objective combines:

```text
teacher local/global point-map distillation
teacher confidence distillation
point-map distance and surface-normal geometry
edge-aware point-map smoothness
median-scale-aligned SCARED ground-truth depth
```

## Checkpoints

Teacher checkpoints contain only `lora_state_dict` plus epoch, global step,
optimizer, scheduler, AMP scaler, and configuration state. Loading order is:

```text
pretrained VGGT-Omega -> MLP LoRA injection -> LoRA checkpoint -> freeze check
```

Student checkpoints retain the complete migrated training state.
Student training starts at `1e-5` and applies cosine decay after every optimizer
update down to `1e-6`. The scheduler state is included in `last.pt` and numbered
epoch checkpoints, so resumed training continues from the saved decay position.

Both stages refresh `last.pt` and preserve numbered checkpoints after every
completed epoch by default:

```text
outputs/teacher_lora_endodac_lora/epoch_0001.pt
outputs/teacher_lora_endodac_lora/epoch_0002.pt
outputs/student_distillation_448x560/epoch_0001.pt
outputs/student_distillation_448x560/epoch_0002.pt
```

The interval is controlled by `training.save_every`; both supplied
configurations set it to `1`.

## Current status

- Complete project structure and stage-specific entry points are implemented.
- VGGT-Omega base parameters and all output heads remain frozen.
- EndoDAC-style MLP LoRA injection, validation, saving, and loading are implemented.
- PC-Depth-inspired highlight detection, inpainting, light alignment, and loss are implemented.
- Teacher temporal self-supervision and formal AMP/checkpoint training loop are implemented.
- Frozen adapted-teacher cache generation is implemented.
- DUNE student cache distillation plus SCARED ground-truth depth supervision is implemented.
- Depth evaluation and reconstruction visualization are implemented.

No dataset, pretrained weight, teacher cache, checkpoint, or experiment output
is included. Populate the documented local directories before running
non-mock training.

## Endo3R depth evaluation

`evaluate.py` follows Endo3R's SCARED depth protocol: it pairs prediction and
`data/depthmap_rectified` frames by numeric frame ID, converts ground truth
from millimetres to metres, resizes both maps to 320x256 with nearest-neighbour
interpolation, applies one median scale ratio to the entire scene, and reports
AbsRel, SqRel, RMSE, RMSE-log, delta1, delta2, and delta3. Scene scores are
weighted by ground-truth sequence length for the final mean.

The project-specific adapter only runs the fixed eight-frame student over
overlapping clips, averages repeated predictions of the same frame, and
selects the RGB directory available in this project's SCARED copy. Official
Endo3R uses `left_rectified`; this project defaults to `auto` so a dataset
containing `left`, `left_finalpass`, or `rgb_data` remains evaluable. The
selected directory names are printed before inference and saved in the output
JSON. This input-path adaptation does not change Endo3R's depth scoring. The
separate `compare_teacher_caches.py` command is a collapse diagnostic and is
not used by `evaluate.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config configs/student_distillation.yaml \
  --checkpoint outputs/student_distillation_448x560/last.pt \
  --split test \
  --output outputs/student_distillation_448x560/evaluation_test_endo3r.json
```

## Video-Depth-Anything depth evaluation

`evaluate_vda.py` is a separate evaluation path that reads the existing
`configs/student_distillation.yaml`; no additional evaluation config is used.
Project adaptation is limited to SCARED discovery, numeric RGB/GT pairing,
overlapping DUNE inference, and conversion of `xyz_local[..., 2]` depth to the
disparity input expected by the upstream evaluator. The scale-and-shift
alignment and AbsRel/RMSE/delta1 scoring core remain unchanged.
Every evaluable clip in the selected split is traversed (1474 clips for the
current test split). Overlapping predictions are accumulated in a temporary
per-sequence disk-backed array, then the unchanged global scale-and-shift and
metric calculations are performed in two streaming passes. This keeps host
memory bounded without truncating a sequence. The VDA adapter uses the
configured 448x560 student/cache resolution for prediction and GT; the
separate official Endo3R entry above intentionally retains its own 320x256
protocol.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_vda.py \
  --config configs/student_distillation.yaml \
  --checkpoint outputs/student_distillation_448x560/last.pt \
  --split test \
  --output outputs/student_distillation_448x560/evaluation_test_vda.json
```

The same entry point evaluates the cached VGGT-Omega teacher without loading
the model, checkpoint, CUDA, or RGB frames. Teacher-cache mode follows the
existing YAML temporal clip settings so its paths match the already generated
cache, and traverses every configured cached test clip. Run it once for each
teacher:

```bash
python evaluate_vda.py \
  --config configs/student_distillation.yaml \
  --teacher-cache data/teacher_cache_base_448x560 \
  --split test \
  --output outputs/evaluation_teacher_base_vda.json

python evaluate_vda.py \
  --config configs/student_distillation.yaml \
  --teacher-cache data/teacher_cache_endodac_lora_448x560 \
  --split test \
  --output outputs/evaluation_teacher_finetuned_vda.json
```
