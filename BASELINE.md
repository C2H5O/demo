# Baseline H

Branch: `baseline-h`, created from G's implementation at `18b5640`.

Selected configuration: `configs/baselines/H.yaml` (also `configs/baseline.yaml`).

Training required: False. Implementation complete; runtime validation pending.

H = existing baseline C checkpoint + 16-frame highlight-aware KV selection.
Here C means the already-trained `feature/attention-distillation` run, whose
server project directory is `vggtoda3`; it does not mean a separate baseline_C run.
G remains E checkpoint + its original 8-frame VDA role KV sampling. C/G configs,
model weights and training objectives are unchanged.

The old H inherited `select_vda_role_kv_frames` in `inference/kv_sampling.py`:
2 key + 2 uniformly selected overlap + 4 uniformly selected new frames.
For a standard window it selected slots `[0,1,2,9,10,17,24,31]`.
There was no highlight selector, token-level highlight pruning, temporal binning
or minimum-spacing rule. `DA3KVAttention.attend` already kept full Q and gathered
only K/V patch tokens, retaining every frame's special tokens.

H now uses `kv_sampling.method: vda_role_highlight`, retention 0.5 and fixed
quotas 2 key + 8 overlap + 6 new. All eligible key/overlap frames are retained.
New frames are ranked by `(highlight_pixel_ratio, original_window_slot)`;
the lowest six are then sorted back into original window order. Equal scores
prefer earlier slots; near-equal but distinct scores keep their numeric order.
There is no uniform/random/stride fallback, temporal binning or forced last frame.
Concentrated selections are intentionally allowed in this clean ablation.

The first window actually has 32 new frames and no history: select the lowest
16 highlight scores. Tail windows exclude padding and duplicate source positions,
keep all eligible history and at most six eligible new frames (possibly fewer
than 16 total). Input windows and all their queries remain length 32.

`SpecularHighlightProcessor.detect_mask_numpy` in `datasets/highlight.py`
exposes the original detector without producing an inpainted output. It preserves
the absolute/candidate/relative thresholds, connected components, median filtering,
and dilation. Its internal candidate-region fill remains part of relative detection;
no filled image is sent to the model. The original `process_numpy` training API
still returns the same mask and inpainted RGB.

The RGB-only sequence evaluator does not load precomputed highlight masks.
`DA3KVAttention.begin_window` therefore computes masks online only for eligible
new candidates: 22 in a normal later window, 32 in the first, fewer in a tail.
Score = binary mask mean over all model-input RGB pixels, not tokens and not
GT-valid pixels. H's `highlight_detection` thresholds match the existing dataset
detector. No neural-network forward or persistent mask cache is added.
Detection uses CPU OpenCV and currently includes a device-to-host image copy
for CUDA inputs; its real cost at 448x560 has not been benchmarked.

The unchanged `infer_student_video` timer starts before `begin_window` and stops
after synchronized model inference. Detection, image copies for detection, mask
averaging, ranking and K/V gather are included in `model_forward_seconds` and
`inference_fps`, as well as the broader `sequence_pipeline_seconds`/`pipeline_fps`.
The optional `global_sdpa_seconds` remains kernel-only and excludes selection;
it must not be presented as total inference time. First-use OpenCV import is also
inside the first timed selection. Debug printing is outside model-forward timing
but inside pipeline timing; warm-up is not excluded.

Synthetic example (window slots, not absolute sequence IDs):

```text
10:.18 11:.27 12:.05 13:.31 14:.22 15:.08 16:.14 17:.29
18:.06 19:.16 20:.24 21:.04 22:.19 23:.25 24:.13 25:.35
26:.21 27:.03 28:.17 29:.28 30:.07 31:.20

Score-ranked new: [27,21,12,18,30,15]
Temporal new:     [12,15,18,21,27,30]
Key:             [0,1]
Overlap:         [2,3,4,5,6,7,8,9]
Final KV:        [0,1,2,3,4,5,6,7,8,9,12,15,18,21,27,30]
```

`configs/inference/H_debug.yaml` logs only the first two windows per sequence,
including `new_highlight_scores`, `new_candidate_count`, selected slots/roles,
absolute IDs and actual SDPA token counts. Normal H does not print per-window
scores; result JSON retains the first two audit examples.

The actual global attention contract is 32Q x 16KV **for patch frames**, plus
all 32 frames' special tokens in K/V. With P patch tokens and S special tokens
per frame: Q `[B,heads,32*(P+S),head_dim]`, K/V `[B,heads,16*P+32*S,head_dim]`.
At the checked local DA3-Small configuration (448x560, patch14, six heads,
head_dim64, S=1), these are Q `[1,6,40992,64]`, K=V `[1,6,20512,64]`.
These full-resolution sizes are source-derived, not a checkpoint runtime result.
Small real upstream CPU Transformer tests observed Q `[1,2,160,12]` and
K=V `[1,2,96,12]`, with unchanged Q, identical K/V gather and all 32 outputs.
Local attention, full QKV projections and heads are unchanged.

`frame_slots_to_token_indices` preserves original slot identity through DA3's
reference-view permutation. Gathering occurs after Q/K normalization and RoPE.
The checked DA3 upstream uses 2D local RoPE and common global patch positions;
it does not encode real inter-frame time distance. No temporal embedding or
renumbering to 0..15 is introduced. Sequence positions and dataset IDs remain
available in the audit.

VDA window traversal, anchor alignment, blending, output aggregation, GT,
valid masks, spatial metrics, exact VDA TAE and SCARED split handling are unchanged.

The source branch's `configs/vggtoda3_attention_distill.yaml` specifies
`outputs/vggtoda3_attention_distill/last.pt`. Combined with the server project
root already used by this repository, H's default checkpoint is:

```text
/public/home/2024141520249/Documents/Projects/vggtoda3/outputs/vggtoda3_attention_distill/last.pt
```

This path is derived from source configuration and the specified server project;
its existence has not been checked on the server. Outputs are isolated under
`outputs/baseline_H/`. This branch does not contain or copy checkpoint files;
use `--checkpoint` if selecting a different saved checkpoint from that run.
The checkpoint loader restores C's actual student architecture/LoRA configuration.
Historical C loss settings are provenance only and are not executed by inference.

From this worktree root, using the same configured environment/data paths as G:

```bash
C_CKPT='/public/home/2024141520249/Documents/Projects/vggtoda3/outputs/vggtoda3_attention_distill/last.pt'

# Two-window audit; not a full evaluation or a speed/accuracy result.
python evaluate_crossclip_projection.py --config configs/inference/H_debug.yaml --checkpoint "$C_CKPT" --limit-windows 2

# Complete C-weight inference with sparse KV.
python evaluate_crossclip_projection.py --config configs/baselines/H.yaml --checkpoint "$C_CKPT"

# Dense comparison using the same C checkpoint and existing C config.
python evaluate_crossclip_projection.py --config configs/baselines/C.yaml --checkpoint "$C_CKPT"
```

H evaluation writes `outputs/baseline_H/evaluation_test.json`; its debug variant
writes `outputs/baseline_H/evaluation_debug.json`. Visualization output is
`outputs/baseline_H/visualization` when using the configured output directory.
Existing entrypoints default to their historical configs; pass `--config`
explicitly to select H (or the branch alias `configs/baseline.yaml`).

CPU validation: `tests/test_highlight_kv.py`, `tests/test_vda_role_kv.py` and
`tests/test_student_video.py`: **46 passed**. This includes detector equivalence,
selector examples/ties/tails, F/G budget preservation and small real upstream
attention under first/middle/saddle_balanced reference strategies. No checkpoint,
full evaluation, GPU benchmark, training or download was run. Real speed, memory
and accuracy remain unvalidated. Changes are not automatically committed/pushed.

Review from the baseline-h checkout before server evaluation:

```bash
git status
git diff
```

See [implementation, shape audit and test commands](docs/VDA_ROLE_KV_SAMPLING.md).
