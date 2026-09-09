# Full-sequence DA3 evaluation

## Inference and comparison scope

The former evaluator forwarded independent 16-view clips at stride 8 and
averaged overlapping inverse-depth predictions. The current evaluator forwards
each full RGB sequence through `inference/student_video.py`, using the window
layout in [VDA's inference implementation](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/video_depth.py).
VDA itself does not put the entire long video into one attention operation.

- Window length 32; stride 22; reference positions `[0,12,24,25,26,27,28,29,30,31]`.
- The first two reference images act as global/rolling anchors. The last eight
  provide a transition to the following window. Real frame order and coverage
  are retained; repeated padding is never counted as output.
- DA3 camera-local Z is converted to disparity. A positive scale and shift fit
  on the two predicted anchor maps brings the next window into the existing
  disparity gauge. No GT enters inference alignment.
- The eight transition disparities are linearly blended, then emitted once.
  Constant anchors or nonpositive scale estimates trigger a positive scale-only
  fallback, reported as `alignment_fallback_count`.
- RGB is decoded by window; finalized predictions use temporary disk-backed
  storage for sequence-level GT alignment and scoring. GPU storage does not
  grow with video duration. Disk usage is about `T*448*560*4` bytes per sequence.

This is an adaptation of VDA inference to a different network and depth
representation, not use of VDA weights or a claim of identical VDA behavior.
The Student wrapper accepts arbitrary positive view counts in evaluation mode;
training and attention-capture execution still require exactly 16 views.
The official DA3-Small, pseudo-label Student and attention Student must all be
reevaluated using this same pipeline for a controlled comparison.

## Spatial metrics

The VDA spatial core retains a single sequence-level least-squares scale/shift
fit in disparity against valid GT, followed by AbsRel, RMSE and delta1.
SCARED depth files are loaded in millimetres and converted to metres. The valid
range remains `(0.001,100)` metres. Predictions/GT are evaluated at 448x560;
GT resizing uses nearest neighbor. Frames are averaged within a sequence and
sequences are macro-averaged. Missing GT sequences and missing predicted GT
frames are recorded; incomplete or debug runs never claim a full test set.

## TAE definition and SCARED adaptation

The sole operational reference is
[Video-Depth-Anything's `benchmark/eval/eval_tae.py`](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/benchmark/eval/eval_tae.py).
The shared implementation in `evaluation/temporal_alignment.py` intentionally
preserves its numerical behavior and adds no alternative TAE definition.

Before reprojection, all predicted disparities in one sequence receive one
float64 least-squares scale/shift fit against GT-valid pixels. SCARED GT depth
is converted from millimetres to metres first, and the existing valid range
remains `(0.001, 100)` metres. The fit is never per-frame, per-pair, or
per-inference-window. VDA-style anchor alignment inside 32-frame inference is a
separate, GT-free stitching step and remains unchanged.

For each adjacent pair, `tae_torch` uses integer pixel coordinates beginning at
zero, backprojects camera Z-depth, applies `R_2_1` and `t_2_1`, rounds projected
coordinates, and performs `depth_proj[valid_Y, valid_X] = valid_Z`. Duplicate
indices therefore follow VDA's direct-assignment behavior: there is no minimum-Z
buffer, scatter reduction, splatting, or fusion. The valid comparison mask is
exactly `(depth_proj > 0) & (depth2 > 0) & mask`; because SCARED has no matching
extra benchmark mask, `mask` is all true and is not replaced by the GT-valid
alignment mask. AbsRel uses target `depth2` as its denominator. An empty
projection contributes zero, as in the reference.

Every pair is evaluated in both directions. The sum is divided by exactly
`2 * (num_frames - 1)` and multiplied by 100, so TAE is percent and lower is
better. Empty directions remain in this fixed denominator; they are not skipped.

VDA's dataset extractor copies the
[ScanNet pose files](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/benchmark/dataset_extract/dataset_extract_scannet.py)
unchanged, and ScanNet's
[official exporter writes camera-to-world poses](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py).
Therefore VDA's `T_2_1 = inverse(T_2) @ T_1` uses c2w inputs. SCARED
`camera-pose` is world-to-camera, as also shown by
[EndoSurf's SCARED reader](https://github.com/Ruyi-Zha/endosurf/blob/master/data/scared2019/preprocess.py),
which inverts that matrix to obtain `c2w`. The evaluator first scales the raw
SCARED w2c translation by `0.001`, then inverts it to VDA-compatible c2w and
uses the same `inverse(T_2) @ T_1` formula. This maps camera-1 coordinates to
camera 2; identity poses produce an identity relative transform.

Like the official evaluator, both directions of a pair use the first frame's
single `K`; there is no source-K/target-K variant. Per-sequence JSON records
whether adjacent resized SCARED intrinsics were exactly equal and their maximum
absolute difference, rather than silently changing the metric when they differ.
The camera matrices must correspond to full-FOV RGB; the data adapter only
rescales K for direct image resizing and does not infer crop, rectification, or
distortion corrections.

## Timing and visualization

`mean_frame_inference_seconds = sum(synchronized forward time) / unique frames`.
`mean_frame_inference_ms` multiplies this by 1000; `inference_fps` is its inverse.
Forward time includes the native depth head, camera decoder and local geometry
currently computed by the wrapper, including repeated anchors and padding.
It excludes checkpoint loading, RGB preprocessing, transfers, fusion, GT
alignment/metrics, and file export. The first invocation is included and
`warmup_excluded` is false. Sequence pipeline time also includes decode,
transfers, fusion and callbacks. Hardware, precision, window/input-frame counts,
resolution and coverage accompany the timings. This is amortized video
throughput, not single-frame latency or online/causal latency.

DA3 native camera poses are independent window coordinate systems. Affine
inverse-depth fusion does not induce a consistent Sim(3) transform for them.
The sequence visualizer therefore reconstructs camera-local PLYs from fused
depth and blended predicted intrinsics, and saves native poses separately per
window. It does not fabricate a merged global cloud. Camera-window files also
contain original frame IDs, including repeated anchors and padding.

The Teacher visualization remains an explicit single cached 16-frame clip;
training, losses, optimizers and teacher-cache protocols are unchanged.
