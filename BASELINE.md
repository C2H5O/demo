# Baseline H: staggered spatial K/V

Baseline H runs the trained Baseline-J epoch-15 Student without training. Its
inference config is `configs/baselines/H.yaml`. The evaluator strictly merges
that training checkpoint to the matching `ours.pt` when needed and uses the
existing 32-frame VDA stitching. Training, model input, and native depth are
448x560. Predictions and ground truth are evaluated at 224x280. TAE is disabled.

## Inference policy

All 32 frames and all tokens remain queries in every window. All 32 frames'
special tokens remain K/V. The first window scores its 32 new frames with the
existing batched lightweight highlight proxy, divides them into 16 temporal
buckets, and selects one provider per bucket. Later windows contain two key,
eight overlap, and 22 new frames. Key and overlap frames are kept; the 22 new
frames form seven temporal buckets and the two lowest-score frames per bucket
are selected. Thus a standard later window has 24 provider frames. Tail windows
may have fewer unique real providers because padding is excluded.

The same original-window provider set is used in all global blocks. In blocks 5
and 7, key providers keep every patch and overlap/new providers keep one
stride-2 lattice (320 of 1280 patches at the 32x40 inference patch grid). The
lattice phase is assigned by the selected provider's rank in original VDA window
order, including key ranks:

| Global block | Key | Overlap/new | Phase by provider rank |
| --- | --- | --- | --- |
| 5 | full | stride 2 | rank modulo 4 |
| 7 | full | stride 2 | (rank + 2) modulo 4 |
| 9, 11 | full | full | none |

Phases 0, 1, 2, 3 are grid offsets (0,0), (0,1), (1,0), (1,1).
Only patch K/V are sampled. The existing reference-first DA3 permutation maps
these original-window slot and patch choices to internal token indices before
K/V gathering. No query tokens or output frames are removed.

The bounded sequence diagnostic prints one JSON sample with layer, provider
slot, provider rank, role, stride, phase, offsets, and patch count. The result
JSON also retains bounded selection examples.

## Checkpoint and command

Both `vda_evaluation.checkpoint` and `inference.source_checkpoint` point to:

```text
/public/home/2024141520249/Documents/Projects/vggtoda3-baseline-J/outputs/baseline_J/epoch_0015.pt
```

Run from the Baseline-H checkout on the server:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py \
  --config configs/baselines/H.yaml \
  --split test \
  --protocol vda
```

This command runs the full evaluation only when invoked explicitly.
