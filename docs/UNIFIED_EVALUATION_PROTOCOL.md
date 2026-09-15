# Unified paper evaluation protocol

DA3 (Baseline A) and Ours (Baseline J) retain the native 448 x 560 model
input and the existing VDA-style full-sequence inference procedure.  The paper
evaluation grid is configured independently as 256 x 320 (H x W).

For every emitted native depth prediction, inference first converts depth to
reciprocal disparity.  The disk-backed sequence spool then bilinearly resamples
disparity to 256 x 320.  SCARED `data/depth` ground truth is converted from mm
to m and nearest-neighbor resampled to the same grid.  One float64 disparity
scale and shift is fit with `numpy.linalg.lstsq` over all valid pixels in the
complete sequence (`0.001 < depth < 100.0` m).  AbsRel, linear RMSE and delta1
are averaged over valid frames and then macro-averaged over sequences.

TAE consumes the same aligned 256 x 320 depths and intrinsics scaled from each
raw RGB frame to that grid.  It preserves the operational Video-Depth-Anything
adjacent-pair, bidirectional, direct-assignment, empty-pair-zero, fixed-
denominator and percent-output behavior.  Evaluation resampling is performed
outside the synchronized model-forward timing scope.

Result JSON distinguishes `model_input_resolution_hw`,
`native_prediction_resolution_hw`, and `evaluation_resolution_hw`.

