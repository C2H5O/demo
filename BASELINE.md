# Baseline H

Branch: `baseline-h`, created from G's implementation at `18b5640`.

Selected configuration: `configs/baselines/H.yaml` (also `configs/baseline.yaml`).

Training required: False. Implementation complete; runtime validation pending.

H = existing baseline C checkpoint + lightweight-highlight selection over the new
frames: first window 16 KV; standard later windows 20 KV.
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

The previous `vda_role_highlight` method used per-frame PC-Depth detection and
global lowest-score selection. That method and detector remain available unchanged
for legacy experiment reproduction; the current H configuration no longer uses them.

H uses `kv_sampling.method: vda_role_bucket_highlight`, later-window retention
0.625 (20/32) and quotas 2 key + 8 overlap + 10 new. All eligible key/overlap
frames are retained. The 22 new frames keep their original window order and are
split before padding or duplicate filtering into a first group of 14 and a final
group of 8. The first group contributes the two lowest lightweight-highlight
scores, with ties resolved by the earlier slot; all eligible frames in the final
group are retained regardless of score. The final history+new set is sorted by
original window slot. No spare quota is redistributed and no source frame is
repeated.

In a short tail, padding and duplicate source positions are removed after the
14/8 split. The first group therefore contributes at most two eligible frames,
while the final group contributes every eligible frame present in its original
eight-slot range. The actual later-window KV count can consequently be below 20.
The previous six-bucket `size_dependent` and `fixed` policies remain available
for regression reproduction, but are not selected by the H configuration.

The first window has 32 new frames and no history: use 16 temporal buckets and
select one lowest-score frame from each. Tail windows exclude padding and duplicate
source positions, keep all eligible history and apply the fixed 14/8 split
(possibly fewer than 20 total). The first window always uses the original
one-per-bucket rule, including short first windows. Input windows and all queries
remain length 32.

`DA3KVAttention.begin_window` gathers eligible new RGB tensors once and calls
`compute_lightweight_highlight_scores` in `inference/lightweight_highlight.py`
on the complete [N,3,H,W] batch. Input RGB is [0,1], before student ImageNet
normalization. All pooling and scoring stay on the input device (GPU in normal
sequence inference). No image/mask host transfer, NumPy, OpenCV, PC-Depth detector
or per-frame detection loop is used by this method.

The lightweight proxy converts to FP32 and average-pools with kernel=stride=4
(448x560 -> 112x140), then computes:

```python
value = rgb.amax(dim=1)
min_rgb = rgb.amin(dim=1)
saturation = (value - min_rgb) / value.clamp_min(1e-6)
score = ((value >= 0.90) & (saturation <= 0.20)).float().mean(dim=(-2, -1))
```

Only the [N] score vector is copied to a Python list, once per window, for simple
deterministic prefix selection and audit. N is 22 for standard later windows,
32 for the first, and smaller for tails. No eligible new frames means no scoring
or transfer. This inference-only frame-ranking proxy is not segmentation GT or a
replacement for the PC-Depth detector used by legacy/training paths.

New options under `kv_sampling.lightweight_highlight` are
`brightness_threshold: 0.90`, `saturation_threshold: 0.20`, `downsample_factor: 4`.
Invalid keys, nonfinite/out-of-range thresholds and nonpositive/noninteger factors
fail validation. `first_window.method` is `bucket_highlight`, with `num_frames: 16`
and `num_buckets: 16`. Later `bucket_highlight` options use
`keep_policy: prefix_recent`, `num_buckets: 2`, `prefix_frames: 14`,
`prefix_keep: 2` and `recent_frames: 8`. Only this H method permits different
first/later budgets: `frame_budget(32, first_window=True)` returns 16; the default
later-window budget is 20. H validates window32, the fixed 14/8 split, 2/8/10
later quotas and the unchanged 16-frame/16-bucket first window. Legacy methods
keep their original budgets.

`DA3KVAttention.begin_window` resolves the current window's budget before selection.
The old unconditional selector budget16 and normal-window 2/8/6 assertions have
been replaced with per-window budget checks. Tail count checks sum the actual
prefix/recent keep counts. Gather and token counts already use `len(selected)` and
need no fixed shape.

The unchanged `infer_student_video` timer starts before `begin_window` and stops
after synchronized model inference. Batch scoring, the score-vector transfer,
prefix/recent selection and K/V gather are included in `model_forward_seconds` and
`inference_fps`, as well as the broader `sequence_pipeline_seconds`/`pipeline_fps`.
The optional `global_sdpa_seconds` remains kernel-only and excludes selection;
it must not be presented as total inference time. Debug printing is outside model-forward timing
but inside pipeline timing; warm-up is not excluded.

Synthetic example (window slots, not absolute sequence IDs):

```text
10:.18 11:.27 12:.05 13:.31 14:.22 15:.08 16:.14 17:.29
18:.06 19:.16 20:.24 21:.04 22:.19 23:.25 24:.13 25:.35
26:.21 27:.03 28:.17 29:.28 30:.07 31:.20

Prefix (first 14): [10,11,12,13,14,15,16,17,18,19,20,21,22,23]
Recent (final 8):  [24,25,26,27,28,29,30,31]
Prefix winners:    [21,12] by score, emitted in window order as [12,21]
Temporal new:      [12,21,24,25,26,27,28,29,30,31]
Key:             [0,1]
Overlap:         [2,3,4,5,6,7,8,9]
Final KV:        [0,1,2,3,4,5,6,7,8,9,12,21,24,25,26,27,28,29,30,31]
```

`configs/inference/H_debug.yaml` logs and retains the first three windows per sequence,
including `new_highlight_scores`, `new_candidate_count`, selected slots/roles,
absolute IDs and actual SDPA token counts. `new_temporal_buckets` records the two
original groups and the flat, window-ordered `bucket_selected_slots` records their
selected slots. The new `bucket_keep_counts` field is [1]*16 in the first full
window and [2,8] in a standard later window (clipped for tails).
`highlight_score_type` remains
`gpu_brightness_low_saturation_ratio`. No pixel masks are serialized.
Normal H does not print per-window scores; its JSON retains two audit examples.

The actual global attention contract is 32Q x 16KV first and 32Q x 20KV later
**for patch frames**, plus
all 32 frames' special tokens in K/V. With P patch tokens and S special tokens
per frame: Q `[B,heads,32*(P+S),head_dim]`, K/V `[B,heads,len(selected)*P+32*S,head_dim]`.
At the checked local DA3-Small configuration (448x560, patch14, six heads,
head_dim64, S=1), Q is `[1,6,40992,64]` in all windows. First-window K=V is
`[1,6,20512,64]` (20480 patch tokens +32 specials), later K=V is
`[1,6,25632,64]` (25600 patch tokens +32 specials).
These full-resolution sizes are source-derived, not a checkpoint runtime result.
The new test code checks tiny-upstream Q `[1,2,160,12]` and first/later K=V
`[1,2,96,12]` / `[1,2,112,12]`,
unchanged Q, identical K/V gather and all 32 outputs; it has not been run.
Local attention, global layers 5/7/9/11, full QKV projections and heads are unchanged.

`frame_slots_to_token_indices` preserves original slot identity through DA3's
reference-view permutation. Gathering occurs after Q/K normalization and RoPE.
The checked DA3 upstream uses 2D local RoPE and common global patch positions;
it does not encode real inter-frame time distance. No temporal embedding or
renumbering of selected frames is introduced. Sequence positions and dataset IDs remain
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

# Three-window profiling, including the first and later window policies.
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/inference/H_debug.yaml --checkpoint "$C_CKPT" --split test --protocol vda --limit-windows 3 --output outputs/baseline_H/evaluation_prefix_recent_3windows.json

# Complete C-weight inference with sparse KV.
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/baselines/H.yaml --checkpoint "$C_CKPT" --split test --protocol vda --output outputs/baseline_H/evaluation_prefix_recent_test.json
```

H evaluation writes `outputs/baseline_H/evaluation_test.json`; its debug variant
writes `outputs/baseline_H/evaluation_debug.json`. Visualization output is
`outputs/baseline_H/visualization` when using the configured output directory.
Existing entrypoints default to their historical configs; pass `--config`
explicitly to select H (or the branch alias `configs/baseline.yaml`).

Tests are supplied but NOT RUN for this change, as requested.
`tests/test_prefix_recent_kv.py` covers the current first14/keep2/final8 policy,
ties, clipped tails, duplicate sources, invalid configuration and the adapter's
dynamic gather contract. `tests/test_bucket_highlight_kv.py` retains the previous
six-bucket 1/1/1/1/2/4 policy as an independent regression fixture, along with
legacy methods and the unchanged batched proxy. The adapter test runs a single
instance through first16 -> later20, checking dynamic gather, full Q, role/token
counts, retention ratios and bucket_keep_counts.
`tests/test_highlight_kv.py` keeps its legacy PC-Depth/global-top-k checks unchanged.
No test execution, model inference, evaluation, profiling, runtime debugging,
or training was performed for this change. Server validation is pending.

Review from the baseline-h checkout before server evaluation:

```bash
git status
git diff
```

See [implementation, shape audit and test commands](docs/VDA_ROLE_KV_SAMPLING.md).
