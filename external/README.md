# External source dependencies

The official Distill3R repository is pinned as a Git submodule at:

```text
external/Distill3R/
```

Initialize Distill3R and its DUNE/Fast3R/VGGT dependencies with HTTPS URLs:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

If VGGT-Omega is not installed globally, place a source checkout at:

```text
external/vggt-omega/
```

Then install it into the active environment from the project root:

```bash
pip install -e external/vggt-omega
```

The optional VGGT-Omega checkout is intentionally excluded from Git. Teacher
runtime code imports the installed package and never reaches into an earlier
project by hard-coded path. Student runtime code imports the pinned official
Distill3R submodule through the project adapter.
