# VGGT-Omega LoRA to Distill3R

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
   official Distill3R `CompressedFast3R` student using teacher point maps/confidence,
   SCARED ground-truth depth, geometry consistency, and edge-aware smoothness.
   Cache-backed training never reruns the teacher inside the student loop.

The implemented LoRA branch is
`y = W x + (alpha / rank) B A x`. The pretrained `W` stays frozen, `A` uses
Kaiming initialization, and `B` starts at zero so injection initially preserves
the base output. The supplied configuration follows EndoDAC's ordinary LoRA
branch with `rank=4`, `alpha=1`, no LoRA dropout, and explicitly rejects
DV-LoRA.

## Reproducible HTTPS setup

The project is Git-managed and pins the official
[Distill3R](https://github.com/TheFourthKaramazov/Distill3R) repository as a
submodule. All repository/submodule URLs use HTTPS. On the training server:

```bash
git clone --recursive https://github.com/C2H5O/demo.git vggtodistill3r
cd vggtodistill3r
bash scripts/setup_environment.sh
conda activate vggtodistill3r
python scripts/verify_environment.py
```

`environment.yml` creates a new `vggtodistill3r` environment with Python
3.10.20, PyTorch 2.3.1, torchvision/torchaudio 0.18.1/2.3.1, and the CUDA 12.1
PyTorch runtime. Download Distill3R's official DUNE 448 checkpoint over HTTPS
to the exact path configured by `student.pretrained_checkpoint`:

```bash
mkdir -p checkpoints/dune
curl --fail --location --retry 5 --continue-at - \
  https://download.europe.naverlabs.com/dune/dune_vitsmall14_448.pth \
  --output checkpoints/dune/dune_vitsmall14_448.pth.part
mv checkpoints/dune/dune_vitsmall14_448.pth.part \
  checkpoints/dune/dune_vitsmall14_448.pth
```

Model construction reads this configured file directly through DUNE's official
checkpoint loader. It never downloads weights implicitly and does not depend on
`TORCH_HOME` or a symbolic link.

For an existing clone, initialize all pinned sources with:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Entry points

The `scripts/*.sh` wrappers always execute through the named
`vggtodistill3r` Conda environment. Direct Python commands below assume that
environment is already activated.

```bash
python train_teacher_lora.py --config configs/teacher_lora_finetune.yaml
python train_teacher_lora.py --config configs/teacher_lora_finetune.yaml --dry-run
python generate_teacher_cache.py --config configs/student_distillation.yaml --split train --overwrite
python generate_teacher_cache.py --config configs/student_distillation.yaml --split test --overwrite
python generate_teacher_cache.py --config configs/student_distillation.yaml --split test --base-teacher --cache-root data/teacher_cache_base_448x560 --overwrite
python compare_teacher_caches.py --config configs/student_distillation.yaml --base-cache data/teacher_cache_base_448x560 --finetuned-cache data/teacher_cache_endodac_lora_448x560
python train_student_distillation.py --config configs/student_distillation.yaml
python evaluate.py --config configs/student_distillation.yaml --checkpoint outputs/student_distill3r_448x560/last.pt --split test
python evaluate_vda.py --config configs/student_distillation.yaml --checkpoint outputs/student_distill3r_448x560/last.pt --split test
python -m pytest -q
```

Install VGGT-Omega itself into the environment as a Python package, for example
with `pip install -e external/vggt-omega`. The project deliberately does not
modify `sys.path` to reach the old repository or an external source checkout.
The student adapter only adds the pinned Distill3R/Fast3R submodule roots to its
import path. All dataset, checkpoint, cache, and output paths are local to this
project by default and live in YAML configuration.

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
models/        frozen teacher + LoRA and the official Distill3R adapter
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
  dune/dune_vitsmall14_448.pth
external/
  Distill3R/                  # pinned Git submodule (with recursive submodules)
  vggt-omega/                 # optional editable dependency checkout
outputs/                      # generated automatically
```

Datasets, caches, weights, and outputs remain ignored; the official Distill3R
source is the exception and is represented by a pinned Git submodule commit.
Run commands from the project root so relative paths resolve consistently. This
project does not need the old `vggt_omega_distill`, PC-Depth, or EndoDAC
directories at runtime.

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
images               RGB normalized according to the active stage
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
instead of silently resizing them. Stage-two RGB uses Distill3R's `[0,1]`
input convention. The official DUNE-S/14 encoder receives a 32x40 patch grid,
and the official Distill3R DPT heads return both local and global point maps at
448x560. The adapter rejects any different input or output shape.

Multi-view training keeps Distill3R's official `flash_attention` decoder
backend. The CUDA-enabled Linux PyTorch build must provide Flash SDPA; for a
small single-view compatibility smoke test on a build without that kernel, set
`student.decoder_attention_implementation: pytorch_naive`. The naive backend is
not recommended for normal multi-view training because attention memory grows
quadratically with the total patch count.

The cache-backed dataset aligns SCARED ground-truth depth to RGB by numeric
frame ID. PNG, TIFF, and NPY depth are supported under `data/depth` and
`data/scene_points`, including configurable channel and scale. Values are
converted from millimetres to metres with `dataset.ground_truth.scale: 0.001`
before supervised loss. The
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
outputs/student_distill3r_448x560/epoch_0001.pt
outputs/student_distill3r_448x560/epoch_0002.pt
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
- Official Distill3R student cache distillation plus SCARED ground-truth depth supervision is implemented.
- Depth evaluation and reconstruction visualization are implemented.

No dataset, pretrained weight, teacher cache, checkpoint, or experiment output
is included. Populate the documented local directories before running
non-mock training.

## Endo3R depth evaluation

`evaluate.py` follows Endo3R's SCARED depth protocol: it pairs prediction and
`data/depth` frames by numeric frame ID, converts ground truth
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
  --checkpoint outputs/student_distill3r_448x560/last.pt \
  --split test \
  --output outputs/student_distill3r_448x560/evaluation_test_endo3r.json
```

## Student inference visualization

The command-line visualization workflow is retained from the earlier
`vggt_omega_distill` project. The exporter now loads the configured official
Distill3R student checkpoint and keeps the same depth-image and RGB point-cloud
outputs.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_depth.py \
  --checkpoint outputs/student_distill3r_448x560/last.pt \
  --split test \
  --clip-offset 0

CUDA_VISIBLE_DEVICES=0 python tools/visualize_cloud.py \
  --checkpoint outputs/student_distill3r_448x560/last.pt \
  --split test \
  --clip-offset 0 \
  --point-stride 2
```

Add `--sequence-id dataset_8/keyframe_0` to select a sequence explicitly. For
the optional interactive point-cloud viewer, install
`requirements-visualization.txt`, add `--serve --host 0.0.0.0 --port 8080`, and
forward the server port over SSH.

## Video-Depth-Anything depth evaluation

`evaluate_vda.py` is a separate evaluation path that reads the existing
`configs/student_distillation.yaml`; no additional evaluation config is used.
Project adaptation is limited to SCARED discovery, numeric RGB/GT pairing,
overlapping Distill3R inference, and conversion of `xyz_local[..., 2]` depth to the
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
  --checkpoint outputs/student_distill3r_448x560/last.pt \
  --split test \
  --output outputs/student_distill3r_448x560/evaluation_test_vda.json
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
