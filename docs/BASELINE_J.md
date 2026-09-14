# Baseline J

> **32-frame ordinary video training clips，VGGT-Ω frozen full-online teacher，DA3 student；teacher/student 输入同一组32帧；所有 teacher supervision 来自同一次 online forward；不使用任何 teacher cache；loss 与原训练完全一致。**

Baseline J inherits Baseline E's Student and Teacher architectures, attention
layer mapping, attention objective, angular soft-highlight behavior, direct
depth/camera distillation, smoothness term, all loss weights, optimizer, data
split, and VDA evaluation protocol. It changes only:

1. the training clip length from 16 to 32 frames; and
2. cached depth/camera/confidence plus online attention into one full-online
   frozen VGGT-Ω forward.

## Training data flow

```text
one sequence, start t
  -> [I_t, I_{t+1}, ..., I_{t+31}]
  -> the same ordered frame IDs feed both branches
     -> frozen VGGT-Ω, one forward: depth/confidence/camera/attention
     -> trainable DA3 Student
  -> unchanged Baseline E losses and weights
```

The sampler keeps Baseline E's `sample_stride=1` and `window_stride=8`. It does
not assign key/overlap/new roles and does not implement a `2+8+22` inference
window. Teacher cache paths are disabled in the J configuration and are never
consulted by the J dataset or training path.

The Teacher reads the corresponding native `teacher_rgb` frames, applies the
same CPU bicubic resize used by the latest Baseline E path (`1024x1280` to
`512x640`), and therefore keeps the Teacher and Student attention grids at
`32x40`. Online dense Teacher supervision is valid-aware resized to the
unchanged Student supervision resolution (`448x560`); intrinsics are adjusted
only for this spatial resize. No metric-depth scale alignment, median scaling,
per-clip normalization, or Teacher-to-GT alignment is added.

J also retains E's mathematically equivalent relation-loss execution settings:
directed frame pairs are processed two at a time and detached Teacher
probabilities are computed outside Student activation checkpoint recomputation.
The frame offsets, temperatures, divergence, query chunk, reduction, and weight
are unchanged.

`dataloader.batch_size` is user-configurable. The shipped value of `1` is only
a conservative memory default for the full-online Teacher plus Student workload;
the trainer does not impose an exact batch size. Likewise, unused cache-path
settings do not block full-online startup, and different Teacher/Student patch
grids are handled by the existing patch-overlap spatial aligner.

## Commands

Minimal backward dry run:

```bash
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py \
  --config configs/baselines/J.yaml \
  --dry-run
```

Formal training (run manually only after the dry run passes):

```bash
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py \
  --config configs/baselines/J.yaml
```

Evaluation remains the existing VDA protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py \
  --config configs/baselines/J.yaml \
  --checkpoint outputs/baseline_J/last.pt \
  --split test \
  --protocol vda \
  --output outputs/baseline_J/evaluation_test.json
```
