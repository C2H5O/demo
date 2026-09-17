# Baseline H = QG-K20

Baseline H is a training-free Query-Grouped K/V inference experiment using the
same final Student weights as Baseline J. It does not use highlight detection,
content descriptors, Teacher inference, spatial token pruning, or TAE.

```text
32-frame VDA window
    ↓
all frames retained as queries
    ↓
temporal query groups (default: 8 frames/group)
    ↓
each group attends to its own K=20 provider frames
    ↓
all patch tokens retained for every provider
    ↓
one batched grouped SDPA for the standard 32/8 layout
    ↓
full 32-frame depth and camera output
```

VDA window construction, anchors, overlap, disparity alignment, blending,
emission order, and output frame count are unchanged.

## Provider selection

For every query group, the provider set first contains every unique real frame
represented by that group. Remaining capacity is filled without RGB or model
features:

1. when `preserve_vda_history: true`, available `key` and `overlap` frames are
   selected first;
2. remaining slots are filled by deterministic temporal-uniform sampling over
   the rest of the complete window;
3. providers are unique and returned in temporal order.

The standard window has four query groups of eight frames and 20 providers per
group. A tail with fewer than 20 unique real frames uses all unique real frames;
padding is never duplicated into K/V. Query and output frames are never removed.

## Attention path

`inference/query_group_kv.py` maps original VDA frame slots through DA3's
reference-first internal permutation. For the standard configuration it forms:

```text
Q: [B*4, heads, 8*tokens_per_frame, head_dim]
K: [B*4, heads, (20*patch_tokens + special_tokens), head_dim]
V: [B*4, heads, (20*patch_tokens + special_tokens), head_dim]
```

and invokes `scaled_dot_product_attention` once per configured global layer.
The grouped output is scattered back to the exact original DA3 token layout.
All provider patch tokens remain at native spatial resolution. With
`keep_all_special_tokens: true`, special-token K/V from every frame is retained;
setting it to false restricts special tokens to provider frames.

The adapter derives global layers from the instantiated encoder and `alt_start`.
For the current 12-block DA3-Small (`alt_start=4`) they are `[5, 7, 9, 11]`.
`apply_layers: null` applies QG to all of them; a list such as `[7, 9]` enables a
layer ablation.

## Configuration

`configs/baselines/H.yaml` exposes the experiment controls directly:

```yaml
kv_sampling:
  enabled: true
  method: query_group
  query_group_size: 8
  kv_frames: 20
  provider_selection: temporal_uniform
  preserve_vda_history: true
  keep_all_special_tokens: true
  apply_layers: null
  batched_sdpa: true
  diagnostics: false
```

`query_group_size`, `kv_frames`, history preference, special-token policy,
target layers, and batched SDPA can be changed in YAML without code changes.
The implementation does not require a 32-frame window or a group size of eight;
only logically impossible settings such as nonpositive sizes, K larger than the
input window, or K too small for a group's mandatory unique frames are rejected.

## Weights, metrics, and timing

H selects Baseline J's final immutable training checkpoint:

```text
/public/home/2024141520249/Documents/Projects/vggtoda3/outputs/baseline_J/last.pt
```

The shared evaluator reuses its matching sibling `ours.pt` or performs the
existing strict auto-merge when absent. H does not train or modify Baseline J.
The Dense comparison comes from the existing Baseline-J evaluation and is not
rerun by H.

H reports AbsRel, RMSE, delta1, model inference time/FPS, and peak CUDA allocated
and reserved memory. `tae.enabled: false` prevents temporal alignment and TAE
from running or appearing in the output. As in the existing Baseline-J Dense
result, metrics are evaluated at 256x320 while the Student still receives and
predicts at its native 448x560 grid. The synchronized model timer begins
before provider construction and includes grouping, gathers, reshapes, grouped
SDPA, output restoration, and all other model-forward work. RGB decoding, GT
loading, metric computation, and serialization remain outside model FPS.

## Commands

Lightweight local checks only:

```bash
python -m pytest tests/test_query_group_kv.py tests/test_qg_evaluation.py -q
```

Formal QG-K20 SCARED VDA evaluation from the Baseline-H checkout:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py \
  --config configs/baselines/H.yaml \
  --split test \
  --protocol vda
```

This command runs only QG-K20. It does not rerun Dense and does not compute TAE.
