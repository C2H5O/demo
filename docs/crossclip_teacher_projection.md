# VGGT-Omega to DA3-Small experiment protocol

The active configuration is `configs/vggtoda3.yaml`. It retains the established
projection, highlight and smoothness losses, frozen teacher cache validation,
VDA/Endo3R evaluation, AMP, resume and visualization machinery. The controlled
changes are the official pretrained DA3-Small student and stride-eight clip
starts/neighbors.

See [coordinate_conventions.md](coordinate_conventions.md) for the normative
depth/camera geometry and exact eight-frame overlap mapping. See the repository
README for complete audit, dry-run, training, resume, evaluation, visualization
and optional teacher-cache regeneration commands.
