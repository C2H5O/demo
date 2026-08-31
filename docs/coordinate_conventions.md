# VGGT-DA3 direct-distillation coordinate conventions

The Student and Teacher consume the same consecutive 16-frame clip at one
cache-compatible start `n`:

```text
C_n^S = [n, n+1, ..., n+15]
C_n^T = [n, n+1, ..., n+15]
```

The dataset requires identical `clip_start`, sequence identity, spatial
resolution, and all 16 `absolute_frame_ids` before returning a sample. The
cache sampling stride is eight only because those are the legal starts used by
the existing cache. No previous or next clip is read.

## Depth

Student and Teacher depth are compared at the same frame and pixel. No point
projection, resampling, or scale alignment is applied. Student local geometry
is deterministically reconstructed only for the unchanged highlight and
smoothness regularizers:

```text
X = (u-cx)/fx * Z
Y = (v-cy)/fy * Z
xyz_local = [X,Y,Z]
```

## Camera poses

Both cache and Student expose world-to-camera matrices:

```text
X_camera = R @ X_world + t
```

The official DA3 camera conversion first returns C2W. The Student wrapper
inverts it exactly once to expose W2C. Camera distillation never compares
absolute poses. With frame zero as the clip reference, it compares:

```text
T(i <- 0) = E_i @ inverse(E_0),  i = 1,...,15
```

This maps coordinates from reference camera zero into camera `i` and cancels a
common absolute world gauge. Rotation is supervised with the SO(3) geodesic
angle; translation direction and log magnitude are supervised independently.
Intrinsics supervision uses `fx/W` and `fy/H`; fixed principal-point terms are
diagnostic only and do not enter the loss.
