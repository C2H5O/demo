# Optional external source

If VGGT-Omega is not installed globally, place a source checkout at:

```text
external/vggt-omega/
```

Then install it into the active environment from the project root:

```bash
pip install -e external/vggt-omega
```

The checkout is intentionally excluded from Git. Runtime code imports the
installed package and never reaches into any earlier project by hard-coded
path.
