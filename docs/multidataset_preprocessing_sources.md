# Multi-dataset preprocessing sources

This document separates verified publication/repository facts from local-release
details. The scripts deliberately stop or require an explicit attestation when
the public sources do not establish a safe camera geometry. They never infer a
stereo split from an image shape.

## Shared output contract

Every successful sequence has a complete marker, `teacher_rgb` PNGs at
1280×1024 (W×H), `student_rgb` PNGs at 560×448 (W×H), and a per-frame source
mapping in `metadata.json`. Both RGB branches are made from the exact same
canonical image using proportional contain-and-pad to 4:5. This avoids an
anisotropic 16:9→4:5 stretch. It preserves the complete source FOV (including
black/circular borders); an entrypoint may use a *published* rectification or
reprojection before that common final step.

## C3VD

- Official dataset URL: <https://durrlab.github.io/C3VD/>
- Official preprocessing URL: <https://github.com/DurrLab/C3VD>
- Original paper: <https://doi.org/10.1016/j.media.2023.102956>
- Endo3R implementation found: the public Endo3R repository identifies C3VD as
  a training dataset, but its training preprocessing is not publicly released:
  <https://github.com/wrld/Endo3R>
- Fallback source: none used.
- Original resolution: **UNVERIFIED for the local download**. The C3VD dataset
  webpage documents the releases; inspect the actual image files before a run.
- Camera model: verified **omnidirectional**, with official `config.ini`
  parameters `width,height,cx,cy,ao,a1,a2,a3,a4,c,d,e` in the registration
  repository README.
- Stereo/mono: mono endoscope video.
- Stereo layout: not applicable.
- Rectification: the official repository supplies registration/rendering, not a
  generic pinhole conversion in Python. The script requires
  `--official-canonical-rgb` and only accepts directories named `rgb`,
  `registered_rgb`, `reprojected_rgb`, or `perspective_rgb`; it will not process
  raw omnidirectional frames as pinhole images.
- Canonical FOV construction: official registered/reprojected perspective RGB,
  attested by the operator, then proportional 4:5 contain+pad.
- Teacher/student transform: same canonical frame → 1280×1024 / 560×448.
- Implementation source: official C3VD README plus the shared implementation.
- UNVERIFIED items: local release directory, exact raw resolution, FPS,
  calibration/registered-RGB provenance until inspected.

## StereoMIS

- Official dataset URL: **UNVERIFIED**. Public papers identify StereoMIS as
  da Vinci Xi stereo in-vivo porcine sequences, but the author download page
  and exact release documentation were not retrievable in this research pass.
- Official preprocessing URL: **UNVERIFIED**.
- Original paper/source found: Endo3R cites StereoMIS as a stereo dataset and
  reports rectification before pseudo-depth: <https://arxiv.org/abs/2504.03198>.
- Endo3R implementation found: <https://github.com/wrld/Endo3R>; the public
  training preprocessing is not released.
- Fallback source: none used (the script intentionally does not borrow SCARED
  calibration or assume a `StereoCalibration.ini` schema).
- Original resolution/FPS/camera model/calibration format: **UNVERIFIED** for
  this release.
- Stereo/mono: stereo (verified by the dataset name/use in Endo3R literature).
- Stereo layout: **UNVERIFIED**. The script accepts only an attested
  `rectified_left`/`left_rectified` directory, or a caller-selected
  `side-by-side`/`top-bottom` packed layout after official rectification. It
  never chooses horizontal or vertical split automatically.
- Rectification: must be performed by author/official workflow before input;
  unsupported calibration files fail safely rather than being guessed.
- Canonical FOV construction: official rectified left → proportional 4:5
  contain+pad.
- Teacher/student transform: same canonical left frame → standard sizes.
- UNVERIFIED items: all local calibration details and raw container layout.

## AutoLaparo

- Official dataset URL/preprocessing: <https://autolaparo.github.io/>
- Official code URL: linked from the project page; local code/archive layout is
  **UNVERIFIED** until inspected.
- Original paper: <https://doi.org/10.1007/978-3-031-16452-1_57>
- Endo3R implementation found: <https://github.com/wrld/Endo3R>; public
  training preprocessing is not released.
- Fallback source: none used.
- Original resolution: verified 1920×1080; FPS: verified 25, from the official
  project page.
- Camera model/intrinsics/distortion: **UNVERIFIED**.
- Stereo/mono: public description supplies surgical videos; stereo status and
  local encoding are **UNVERIFIED**, so this entrypoint processes only ordinary
  RGB image sequences or a single video stream, never a packed stereo frame.
- Rectification / official crop: **UNVERIFIED**; no crop is invented.
- Canonical FOV construction: complete input frame → proportional 4:5
  contain+pad, preserving any black border and same FOV for both branches.
- Teacher/student transform: same canonical frame → standard sizes.
- Implementation source: official resolution/FPS and streaming OpenCV decode.
- UNVERIFIED items: local video filename/layout, valid-FOV crop, calibration.

## EndoVis18 (Robotic Scene Segmentation)

- Official dataset URL: <https://opencas.dkfz.de/endovis/datasetspublications/>
- Official challenge data URL: <https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Data/>
- Original paper: <https://arxiv.org/abs/2001.11190>
- Endo3R implementation found: <https://github.com/wrld/Endo3R>; public
  training preprocessing is not released.
- Fallback source: none used.
- Original resolution: verified SXGA 1280×1024 per eye in the challenge paper.
- Stereo/mono: verified stereo pairs; left and right are separate directories
  (`left_frames`, `right_frames`) in common challenge releases.
- Camera model: calibrated da Vinci Xi stereo; detailed model/distortion field
  names are **UNVERIFIED** until the local `camera_calibration.txt` is parsed.
- Stereo layout: separate left/right files; script reads left only and uses
  natural numeric filename ordering.
- Rectification: **UNVERIFIED** for a particular archive. The script requires
  acknowledgement and retains published left geometry without inventing a
  calibration transform. It records calibration-file presence in metadata.
- Canonical FOV construction: published left image → proportional 4:5
  contain+pad. Labels are never read.
- Teacher/student transform: same canonical left frame → standard sizes.
- UNVERIFIED items: local archive calibration semantics, whether files are
  pre-rectified, actual timestamps/FPS.

## Hamlyn (Endo-Depth-and-Motion rectified evaluation release)

- Official Hamlyn dataset URL: <https://hamlyn.doc.ic.ac.uk/vision/>
- Evaluation/preprocessing release: <https://davidrecasens.github.io/EndoDepthAndMotion/>
- Official code: <https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion>
- Original paper: <https://arxiv.org/abs/2005.07557>
- Endo3R implementation found: <https://github.com/wrld/Endo3R> (evaluation
  support; no public training-preprocessing implementation).
- Fallback source: none used.
- RGB/depth facts: the evaluation release states that RGB images are rectified
  uint8 JPG, GT is Libelas-generated uint16 PNG depth in **mm**.
- Original resolution/FPS/camera calibration schema: **UNVERIFIED** per local
  sequence; the script reads dimensions rather than presuming one.
- Stereo/mono: rectified stereo release; the processed contract uses left RGB.
- Stereo layout: separate rectified image files in the documented release.
- Rectification: supplied release is rectified; `--rectified-rgb` is required.
- Canonical FOV construction: rectified left RGB and its corresponding depth
  receive the same proportional contain+pad transform. Depth uses nearest and a
  separate valid mask, so invalid zero values cannot contaminate valid depth.
- Teacher/student transform: same canonical RGB → standard sizes; GT depth is
  saved as float32 `.npy` at 448×560 in mm, zero invalids retained.
- Implementation source: Endo-Depth project description + shared valid-aware
  transform.
- UNVERIFIED items: exact local folder names/frame IDs (the script checks them
  and rejects silent RGB/GT mismatches).
