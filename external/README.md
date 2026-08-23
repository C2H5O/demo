# Pinned upstream sources

The current experiment requires only:

```text
external/DUNE/
external/Distill3R/          # supplies external/fast3r and its DPT head
```

Initialize them with:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

VGGT-Omega is an installed Python package rather than a tracked submodule.
