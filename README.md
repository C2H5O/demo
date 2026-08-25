# Cross-clip teacher projection

This repository contains one experiment only: frozen 16-frame VGGT-Omega
teacher caches jointly supervise a trainable DUNE ViT-S/14 encoder and Fast3R
DPT point head.

```text
16 RGB frames
  -> trainable DUNE blocks [2, 5, 8, 11]
  -> Fast3R DPT point head
  -> camera-local XYZ
  -> projection + highlight + smoothness losses
```

There is no Fast3R/MASt3R decoder, random image-ID embedding, photometric loss,
SSIM, ground-truth training loss, extra point loss, confidence loss, or online
teacher forward during student training.

## Setup

```bash
git submodule sync --recursive
git submodule update --init --recursive
conda env create --file environment.yml
conda activate vggtofast3r
python scripts/verify_environment.py
```

Required local assets (ignored by Git):

```text
data/SCARED/
checkpoints/vggt_omega/vggt_omega_1b_512.pt
checkpoints/dune/dune_vitsmall14_448.pth
```

VGGT-Omega must be installed as an importable Python package. The pinned DUNE
source and Fast3R head source are provided by the Git submodules.

## Full workflow

Generate immutable raw teacher caches:

```bash
python generate_crossclip_teacher_cache.py --config configs/crossclip_teacher_projection.yaml --split train
python generate_crossclip_teacher_cache.py --config configs/crossclip_teacher_projection.yaml --split test
```

Teacher cache generation uses its own batched input pipeline. The default
`teacher.inference_batch_size: 4` sends `[B,16,3,448,560]` tensors to
VGGT-Omega. BF16 is used only for the frozen model forward; camera decoding and
all saved cache arrays remain FP32. `teacher_dataloader` controls CPU loading
independently of the training `dataloader`. Compressed NPZ writes run in a
bounded background pool so compression can overlap the next GPU batch. On an
80 GB GPU, test batch sizes 4 and 8 with `--limit 16` before a full run; reduce
the value if CUDA reports out of memory.

To resume without validating every earlier cache, select the first unfinished
clip by global index or by its SCARED location:

```bash
python generate_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml --split train \
  --start-dataset-id 5 --start-keyframe-id keyframe_3 --start-clip-start 0
```

`--limit` counts clips from the selected start. Existing caches at or after the
start are still validated and skipped; caches before it are not opened.

To convert an existing **complete** compressed raw cache to complete
uncompressed NPZ without rerunning the teacher, first stop every process that
can read or write that cache root. Inspect a small selection, convert one file,
then run the resumable full conversion:

```bash
python convert_crossclip_teacher_cache.py \
  --root data/teacher_cache_crossclip_base_raw_448x560 --dry-run --limit 5
python convert_crossclip_teacher_cache.py \
  --root data/teacher_cache_crossclip_base_raw_448x560 \
  --confirm-no-readers --limit 1
python convert_crossclip_teacher_cache.py \
  --root data/teacher_cache_crossclip_base_raw_448x560 \
  --confirm-no-readers
```

The converter is sequential and preserves every key, shape, dtype, and value.
For each source it writes a UUID-named temporary in the same directory, fsyncs
it, checks ZIP CRC and exact NumPy equality, confirms that the source did not
change, and only then uses an atomic `os.replace`. It refuses symlinks,
incomplete/non-cross-clip NPZ files, insufficient temporary disk space,
concurrent converters, and writes without `--confirm-no-readers`. A completed
uncompressed file is detected from its ZIP metadata and skipped, so rerunning
the same command resumes safely. A hard-killed job may leave the root lock;
check the hostname/PID recorded in the lock and remove it only after confirming
that process is gone. Do not run training, evaluation, visualization, cache
generation, or alignment against the same root during conversion.

Audit and write the separate offline-aligned cache roots:

```bash
python align_crossclip_teacher_cache.py --config configs/crossclip_teacher_projection.yaml --split train --audit-only
python align_crossclip_teacher_cache.py --config configs/crossclip_teacher_projection.yaml --split train
python align_crossclip_teacher_cache.py --config configs/crossclip_teacher_projection.yaml --split test
```

Inspect teacher cache, dry-run, train, and resume:

```bash
python visualize_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml --source teacher --split train --clip-index 0
python train_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml --dry-run
python train_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml
python train_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml --resume outputs/crossclip_teacher_projection/last.pt
```

The randomly initialized DPT output layer starts with a small weight scale and
a positive raw-Z bias. This is required because Fast3R's exponential point-map
postprocess scales vector magnitude but does not force its Z direction to face
the camera. Training aborts after five consecutive batches without a valid
student-to-teacher projection instead of silently advancing with zero loss.

Joint training uses separate AdamW parameter groups: the DPT head follows
`training.learning_rate`, while DUNE follows the smaller
`training.encoder_learning_rate`. Setting `student.freeze_encoder: true`
retains the supported head-only ablation and removes DUNE from the optimizer.
Because all 16 frame encodings retain activations in joint mode, use a physical
batch size of 1 and gradient accumulation for a larger effective batch.

Evaluate and visualize the trained student:

```bash
python evaluate_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml
python evaluate_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml --protocol endo3r
python visualize_crossclip_projection.py --config configs/crossclip_teacher_projection.yaml --source student --checkpoint outputs/crossclip_teacher_projection/last.pt --split test --clip-index 0
```

Run CPU-safe unit tests:

```bash
python -m pytest -q
```

See [the experiment protocol](docs/crossclip_teacher_projection.md) and
[coordinate conventions](docs/coordinate_conventions.md) for the exact
neighbor mapping, projection direction, cache metadata, pose convention, and
offline scale alignment contract.
