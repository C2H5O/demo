"""Immutable constants for the formal Hamlyn zero-shot protocol."""

from __future__ import annotations


HAMLYN_SEQUENCE_IDS = (
    1,
    4,
    5,
    6,
    8,
    9,
    11,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
)

HAMLYN_GT_SCALE = 0.001
HAMLYN_MIN_DEPTH = 0.001
HAMLYN_MAX_DEPTH = 0.300
EVALUATION_RESOLUTION_HW = (224, 280)

METHOD_ORDER = ("ours", "da3", "endodav", "endo3r")
METHOD_LABELS = {
    "ours": "Ours",
    "da3": "DA3",
    "endodav": "EndoDAV",
    "endo3r": "Endo3R",
}
INFERENCE_RESOLUTIONS_HW = {
    "ours": (224, 280),
    "da3": (224, 280),
    "endodav": (224, 280),
    "endo3r": (256, 320),
}
PREDICTION_REPRESENTATIONS = {
    "ours": "disparity",
    "da3": "disparity",
    "endodav": "depth",
    "endo3r": "depth",
}

SCALE_ALIGNMENT = (
    "one float64 numpy.linalg.lstsq disparity scale and shift over all "
    "GT-valid pixels in each complete sequence"
)
METRIC_AGGREGATION = (
    "per-frame within sequence, then macro mean over 22 sequences"
)


def validate_protocol_constants() -> None:
    if len(HAMLYN_SEQUENCE_IDS) != 22 or 9 not in HAMLYN_SEQUENCE_IDS:
        raise RuntimeError("Hamlyn protocol must contain exactly 22 sequences including 9")
    if len(set(HAMLYN_SEQUENCE_IDS)) != len(HAMLYN_SEQUENCE_IDS):
        raise RuntimeError("Hamlyn sequence IDs must be unique")
    if EVALUATION_RESOLUTION_HW != (224, 280):
        raise RuntimeError("Hamlyn evaluation grid must remain 224x280 HxW")
    if INFERENCE_RESOLUTIONS_HW["endo3r"] != (256, 320):
        raise RuntimeError("Endo3R inference must remain 256x320 HxW")


validate_protocol_constants()
