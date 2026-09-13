# Baseline I

Selected configuration: `configs/baselines/I.yaml`.

See [experiment plan](docs/BASELINE_EXPERIMENTS_4090.md).

Training required: True. Implemented: True.

Baseline I inherits Baseline E's complete training and loss behavior. Its only
experimental variable is fixed offline Teacher-to-Teacher multiplicative scale
alignment: adjacent raw 16-frame clips use their eight shared absolute frame IDs,
and all Teacher geometry is accumulated into the first clip's sequence gauge.
Raw cache NPZ files remain unchanged; training reads the alignment JSON and scales
Teacher depth plus world-to-camera translation.
