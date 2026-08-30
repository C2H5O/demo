# VGGT-Omega to DA3-Small experiment protocol

The active configuration is `configs/vggtoda3.yaml`. It retains the established
projection, highlight and smoothness losses, frozen teacher cache validation,
VDA/Endo3R evaluation, AMP, resume and visualization machinery. The controlled
changes are the official pretrained DA3-Small student and stride-eight clip
starts/neighbors.

Backbone adaptation uses standard LoRA only on the two MLP linear projections
(`fc1` and `fc2`) in each of the 12 DINOv2 ViT-S blocks. The frozen backbone
base, LoRA adapters, fully trainable DualDPT/CameraEnc/CameraDec heads, and
disabled ray-only DualDPT branch are audited independently at startup.

Highlight detection and inpainting are materialized once per unique student
frame by `precompute_highlights.py`. Training strictly reads the resulting mask
and clean-RGB PNGs, validated against a per-keyframe configuration marker; no
OpenCV highlight processing executes in DataLoader workers.

See [coordinate_conventions.md](coordinate_conventions.md) for the normative
depth/camera geometry and exact eight-frame overlap mapping. See the repository
README for complete audit, dry-run, training, resume, evaluation, visualization
and optional teacher-cache regeneration commands.
