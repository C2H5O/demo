# Teacher self-supervised loss boundary

Implemented components:

- differentiable temporal reprojection from VGGT-Omega depth, intrinsics, and
  camera-from-world extrinsics;
- SSIM/L1 photometric loss, automasking, dynamic weighting, and minimum
  reprojection;
- PC-Depth-inspired highlight masking, light alignment, surface-normal
  highlight loss, and inpainted-image edge-aware smoothness;
- depth, pose, and confidence consistency between original and inpainted
  teacher predictions.

The implementation is framework-native and does not depend on PC-Depth at
runtime.
