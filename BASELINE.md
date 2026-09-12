# Baseline I

Selected configuration: `configs/baselines/I.yaml`.

See [experiment plan](docs/BASELINE_EXPERIMENTS_4090.md).

Training required: True. Implemented: True.

Baseline I inherits Baseline E and changes only depth distillation to one
detached, clip-shared affine disparity fit per 16-frame sample, followed by a
frame-balanced confidence-weighted Smooth-L1 residual.
