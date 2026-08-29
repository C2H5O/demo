# VGGT-DA3 coordinate and overlap conventions

The student consumes one joint tensor `[B,16,3,448,560]`. DA3 predicts depth,
pixel-space intrinsics and camera poses. All extrinsics exposed by this project
are world-to-camera matrices:

```text
X_camera = R @ X_world + t
```

The official camera decoder first produces C2W pose encodings. The student
wrapper follows DA3's own conversion, then inverts C2W exactly once and exposes
`extrinsics_w2c` with shape `[B,16,3,4]`.

## Deterministic geometry

For each pixel `(u,v)` and predicted depth `Z`:

```text
X = (u-cx)/fx * Z
Y = (v-cy)/fy * Z
xyz_local = [X,Y,Z]
xyz_global = inverse(T_w2c) @ homogeneous(xyz_local)
```

Runtime checks require `depth == xyz_local[...,2]`, positive depth, finite
camera matrices, and exact output shapes. There is no learned point-map or ray
geometry head in the reconstruction path.

## Stride-eight neighbors

Clip frames remain consecutive, while legal starts are `s = 0,8,16,...`:

```text
C_0  = [0,...,15]
C_8  = [8,...,23]
C_16 = [16,...,31]
```

For current `C_s`, previous supervision is `C_(s-8)[8:16]` matched to
`C_s[0:8]`; next supervision is `C_(s+8)[0:8]` matched to `C_s[8:16]`.
Both intersections must contain exactly eight identical absolute frame IDs.
Neighbors are keyed by dataset, sequence and `clip_start`; no stride-one or
cross-sequence fallback is allowed.

## Projection frame

DA3 local points depend on predicted depth and K. They are transformed by the
predicted C2W to `xyz_global`, then by the frozen matching teacher W2C to the
teacher camera frame. The existing relative depth projection residual and
teacher confidence weighting are applied there. This preserves the original
projection/highlight/smoothness objective while allowing projection gradients
to reach the DA3 backbone, depth head and camera decoder.

Teacher arrays remain detached, lazy-loaded pseudo-labels. Dense supervision
is always 448x560. A cache may additionally retain native high-resolution
teacher input metadata; this is audited but never used to resize student input.
