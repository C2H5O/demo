# Baseline H

Branch: `baseline-h`, created from G's implementation at `18b5640`.

Selected configuration: `configs/baselines/H.yaml` (also `configs/baseline.yaml`).

Training required: False. Implementation complete; runtime validation pending.

H = existing baseline C checkpoint + the exact VDA role KV sampling used by G.
Here C means the already-trained `feature/attention-distillation` run, whose
server project directory is `vggtoda3`; it does not mean a separate baseline_C run.
G remains E checkpoint + VDA role KV sampling. C/G configs and all model,
training and inference algorithm code are unchanged on this branch.

H inherits G's policy: all 32 query frames, 2 key + 2 overlap + 4 new patch-KV
frames, all special tokens, uniform first-window sampling and the same tail
fallback. No highlight/texture/ring-anchor selection, retraining or new weights.

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

Only configuration inheritance, checkpoint/output routing, JSON and diff checks
were performed for H. No model forward, training or benchmark was run.

See [implementation, shape audit and test commands](docs/VDA_ROLE_KV_SAMPLING.md).
