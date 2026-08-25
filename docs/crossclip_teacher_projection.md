# Frozen-teacher cross-clip projection experiment

The experiment uses 16 independently encoded frames. DUNE ViT-S/14 returns
0-based Transformer blocks `[2,5,8,11]` and is optimized jointly with Fast3R's
DPT point head. No Fast3R/MASt3R/LLaMA multi-view decoder or random image
identifier is used. A supported `freeze_encoder: true` ablation trains the DPT
head alone.

The total objective is exactly:

```text
L = 1.0 L_projection + 0.01 L_highlight + 0.1 L_smooth
```

`L_projection` averages valid left/right adjacent-clip teacher projection
residuals. `L_highlight` is a camera-facing surface-normal term on detected
highlights. `L_smooth` is mean-normalized inverse-depth smoothness using
inpainted RGB and excluding highlight-crossing edges. There is no photometric,
SSIM, ground-truth, extra point-map, or other training loss.

## Full workflow

Generate immutable FP32 raw caches from the frozen base teacher for both
training and evaluation splits:

```bash
python generate_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml \
  --split train
python generate_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml \
  --split test
```

Existing caches that predate the strict camera/validity integrity metadata are
rejected. Regenerate those explicitly with `--overwrite`; do not mix them with
the aligned root.

Raw-cache inference is genuinely batched: `teacher.inference_batch_size`
controls the leading dimension of `[B,16,3,H,W]`, while `teacher_dataloader`
controls CPU workers, pinned memory, persistence, and prefetching. Teacher AMP
defaults to BF16, but geometry decoding and saved arrays remain FP32. NPZ
compression is overlapped with later GPU batches using
`teacher.cache_write_workers`; queued writes are bounded to avoid uncontrolled
host-memory growth. For an initial throughput and memory check:

```bash
python generate_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml \
  --split train --limit 16
```

Monitor compute utilization with `watch -n 1 nvidia-smi`. The training
`dataloader.batch_size` does not control this offline teacher-cache command.

For an exact resume point, use either `--start-index N` or the location tuple
`--start-dataset-id`, `--start-keyframe-id`, and `--start-clip-start`. Start
selection occurs before existing-cache validation, so earlier compressed NPZ
files are not opened. For example:

```bash
python generate_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml --split train \
  --start-dataset-id 5 --start-keyframe-id keyframe_3 --start-clip-start 217
```

The location must identify an exact clip. `--limit`, when present, is the
number of clips processed from that selected start.

Audit scale drift without writing aligned caches:

```bash
python align_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml \
  --split train --audit-only
```

Write the separate offline-aligned roots used by the default training config:

```bash
python align_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml \
  --split train
python align_crossclip_teacher_cache.py \
  --config configs/crossclip_teacher_projection.yaml \
  --split test
```

Inspect an aligned teacher cache before training:

```bash
python visualize_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml \
  --source teacher --split train --clip-index 0
```

Run one guarded dry run, train, or resume:

```bash
python train_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml --dry-run
python train_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml
python train_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml \
  --resume outputs/crossclip_teacher_projection/last.pt
```

VDA remains the default evaluation; Endo3R is retained explicitly:

```bash
python evaluate_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml
python evaluate_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml --protocol endo3r
```

Visualize a trained student clip with fixed/adaptive depth maps and one
camera-local PLY per frame:

```bash
python visualize_crossclip_projection.py \
  --config configs/crossclip_teacher_projection.yaml \
  --source student \
  --checkpoint outputs/crossclip_teacher_projection/last.pt \
  --split test --clip-index 0
```

See [coordinate_conventions.md](coordinate_conventions.md) for the complete
geometry, pose, intrinsic, scale, and adjacent-window contracts.
