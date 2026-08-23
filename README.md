# Cross-clip teacher projection

This repository contains one experiment only: frozen 16-frame VGGT-Omega
teacher caches supervise a frozen DUNE ViT-S/14 encoder plus a trainable
Fast3R DPT point head.

```text
16 RGB frames
  -> frozen DUNE blocks [2, 5, 8, 11]
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
