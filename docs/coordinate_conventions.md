# Cross-clip coordinate conventions

This document is normative for `configs/crossclip_teacher_projection.yaml`.
Every point, depth, pose and intrinsic matrix used by the experiment must obey
the contracts below.

## Student and teacher point maps

For frame `i`, the student predicts `P_s[i,v,u] = (X,Y,Z)` in that frame's
own camera-local coordinate system. The Fast3R DPT head is applied to DUNE
features from each frame independently; no decoder establishes a common
multi-view frame. Consequently, point maps from different frames must not be
concatenated as one reconstruction without an explicitly supplied pose.

The teacher cache stores both:

- `xyz_local[i]`: camera-local points for teacher frame `i`;
- `xyz_global[i]`: the teacher's world/gauge points, retained for diagnostics
  but not consumed by the training loss.

The pseudo-label used by training is teacher `depth`, sampled in the matching
teacher frame. `depth_adaptive` and `depth_fixed` are visualization color maps
only and are never pseudo-labels.

## Pose convention

Cached extrinsics are world-to-camera matrices:

```text
X_camera = R @ X_world + t
```

Therefore `X_world = R^T @ (X_camera - t)`. If an entire teacher gauge is
multiplied by a positive scale `s`, camera-local/global XYZ, depth, and the
extrinsic translation `t` are multiplied by `s`; `R` and `K` do not change.

## Intrinsics and image transforms

`K[i]` is defined on the exact post-transform `448 x 560` teacher depth grid.
The cache generator passes the same deterministic RGB tensor used by the
student dataset to VGGT-Omega after only converting `[-1,1]` to `[0,1]`.
There is no independent teacher resize, random crop, flip, or augmentation.

The configured `resize_mode: resize` directly maps the source image to
`448 x 560`. For a general resize/crop, `resize_crop_intrinsics` applies:

```text
fx' = sx fx       fy' = sy fy
cx' = sx cx - crop_x
cy' = sy cy - crop_y
```

Projection and cached depth must always use the same grid and transformed
intrinsics.

## Student-to-teacher projection

For a student camera-local point `(X,Y,Z)` and the matching teacher frame's
intrinsics `K`, projection is:

```text
u = fx X/Z + cx
v = fy Y/Z + cy
```

The student `(u,v)` drives `grid_sample` over the teacher depth map. A sample is
valid only when student XYZ is finite, `Z > eps`, `(u,v)` is inside the teacher
grid, and the sampled teacher-valid mask is true. Teacher depth, confidence and
intrinsics are detached pseudo-label tensors. Gradients flow through the
student XYZ and its sampling grid, never into teacher data.

The relative residual at a valid projected pixel is:

```text
|Z_student - D_teacher(u,v)| /
    (Z_student + D_teacher(u,v) + eps)
```

Teacher confidence optionally weights this residual. Highlight pixels are
included in projection by default (`projection_ignore_highlight: false`).
If an otherwise valid overlap has zero total normalized teacher confidence,
the loss falls back to uniform weights instead of silently disabling
projection supervision; the effective left/right weight sums are logged.

## Adjacent-clip mapping

For a sequence of `n` frames, legal clips are `C_t = [t, ..., t+15]` for
`t = 0, ..., n-16`. Windows use stride one and never cross sequence boundaries.

For student clip `C_t`:

- left supervision uses student local indices `[0:15]` and teacher
  `C_(t-1)[1:16]`;
- right supervision uses student local indices `[1:16]` and teacher
  `C_(t+1)[0:15]`.

Each side therefore contributes 15 matching absolute frames. `C_0` has no
left side; `C_(n-16)` has no right side. For example, `C_9=[9,...,24]` is
matched to `C_8[1:16]=[9,...,23]` on the left and
`C_10[0:15]=[10,...,24]` on the right. Runtime assertions compare absolute
frame IDs before computing either loss.

## Scale policy

Teacher clip-scale drift is audited and optionally corrected offline. Adjacent
teacher clips estimate a robust median depth ratio over their 15 shared frames;
aligned caches are written to a different root. Raw caches remain unchanged.

No per-batch or per-sample alignment is applied to the student, because that
would remove the scale signal from the pseudo-label objective. VDA evaluation
retains its protocol-defined sequence scale/shift alignment; that evaluation
operation is not part of training.

Cache validation is fail-closed: every frame must meet the declared valid-pixel
fraction, valid depth must be positive and agree with local-point `Z`, and
intrinsics/extrinsics must be finite and non-degenerate. Only dense values at
pixels already marked invalid may be stored as zero; camera matrices are never
repaired with `nan_to_num`.
