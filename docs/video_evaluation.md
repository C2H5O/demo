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

[Depth Any Video, Eq. 7](https://arxiv.org/html/2410.10815v2) defines temporal
alignment error using bidirectional reprojection of predicted depths with
dataset cameras. Its repository does not expose the benchmark evaluator in
the inspected public root. We use the executable interpretation in
[VDA's `eval_tae.py`](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/benchmark/eval/eval_tae.py)
as an additional implementation reference.

For each consecutive source-frame pair, let `E_i` be dataset world-to-camera
extrinsics and `K_i` its resized intrinsics. Transform backprojected depth from
frame i to j using `E_j @ inverse(E_i)`, project to the target image, and compute
`mean(abs(projected_Z - predicted_target_Z) / predicted_target_Z)` on valid
projected pixels. Repeat in the reverse direction. `tae` is 100 times the mean
of these directed pair errors. Lower is better; its unit is percent. All frames
use the same GT disparity alignment as the spatial metrics. No per-frame or
per-pair rescaling is allowed.

The paper's printed sum bounds and `T-2` denominator are inconsistent with the
stated adjacent-frame definition. The executable VDA reference uses `2*(T-1)`;
we use the actual number of valid directed comparisons. Two explicit numerical
differences from that script are necessary for deterministic and auditable
evaluation: collisions use a nearest-surface z-buffer instead of unordered
last-write assignment; empty projections are failures/skips, not zero errors.
Points behind the target camera are rejected. Source and target intrinsics may
differ. These choices are recorded as a SCARED adaptation, so resulting scores
must not be presented as a bit-identical ScanNet/ScanNet++ reproduction.

TAE considers neighboring RGB frames whose numerical IDs differ by
`tae.frame_id_step` (default 1). It does not bridge missing frames. Sparse GT
can still supply the one sequence-wide depth alignment; cameras must exist for
each scored RGB frame. The metric uses valid positive predicted depths and
in-bounds positive-Z projections, without an additional GT visibility mask,
optical-flow network, or filtering based on the prediction's error.

SCARED `camera-calibration.KL` and `camera-pose` are read from
`data/frame_data/*.json`; translations are multiplied by 0.001, like depth GT.
The world-to-camera convention is also used in
[EndoSurf's SCARED reader](https://github.com/Ruyi-Zha/endosurf/blob/master/data/scared2019/preprocess.py).
The camera matrices must correspond to the RGB field of view. This reader
supports full-FOV direct resizing; it does not infer crop, rectification or
distortion corrections. Cropped/preprocessed RGB or alternative calibration
formats need an explicit matching camera transformation before evaluation.
Rigid camera reprojection does not compensate for independently moving or
deforming tissue; interpret TAE together with spatial accuracy and pair
coverage rather than as a complete measure of depth quality.

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
