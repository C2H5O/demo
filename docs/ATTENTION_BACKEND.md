# Baseline DA3 attention backend

The seven baseline branches were inspected individually. Their starting trees
only differ in BASELINE.md and configs/baseline.yaml; all model, capture, loss,
trainer, entrypoint, and named baseline config files are identical.

| Baseline | Online model / supervision | Q/K capture | Runtime status |
|---|---|---|---|
| A | Official DA3-Small inference; no Teacher | None | Evaluation only |
| B | DA3 Student; cached Teacher geometry | None | Training supported |
| C | DA3 Student; cached geometry plus online frozen VGGT-Omega Q/K | Existing Student and Teacher hooks | Training supported |
| D | DA3 Student; cached Teacher geometry | None | Training supported |
| E | DA3 Student; cached geometry plus online frozen VGGT-Omega Q/K | Existing Student and Teacher hooks | Training supported |
| F | Planned Spark3R reference using E weights | No running path | Still unimplemented; existing guards preserved |
| G | Planned Spark3R proposed using E weights | No running path | Still unimplemented; existing guards preserved |

F/G inherit attention-distillation settings from their planned E-based definition,
but neither training nor Spark3R inference is implemented/enabled by this patch.
No Spark3R algorithm or KV retention is added.

## Configuration and actual execution

All baseline configs inherit:

```yaml
student:
  attention_backend: auto
```

- auto: prefer Flash SDPA for the actual Q/K/V, mask, dtype and device; otherwise
  efficient SDPA, then math SDPA. If PyTorch has no SDPA API, use eager.
- flash: require Flash SDPA. An unsupported build/device/input raises an error.
- sdpa: select a supported PyTorch SDPA kernel, preferring Flash, with no eager
  fallback when the SDPA API is missing.
- eager: explicit DA3 pre-scaled Q / softmax / V implementation for comparison.

Put the override under student in configs/baseline.yaml or the selected letter
config. Training, official evaluation and trained evaluation/visualization all
construct DA3SmallStudent. Evaluation preserves the saved checkpoint's architecture
and weights but takes attention_backend from the runtime config, including for old
checkpoints. No checkpoint conversion is needed.

The pinned official DA3 source (3d835ec1a5802d64a8b8b15f817a1ab54809bfe4)
already uses PyTorch SDPA by default (fused_attn=True). This patch makes Flash
preference explicit, adds a reversible backend switch and verifies the selected
kernel. It does not establish a speedup over a server that already dispatched Flash.
There is no HuggingFace attn_implementation argument on this construction path.

Only official DA3 Attention instances are adapted, without modifying installed
third-party source or globally patching PyTorch. Each SDPA call enables exactly
one selected kernel. The initial backend report prints after successful execution
using actual tensors, once per Student. A later change such as an FP32 retry emits
one warning per newly observed backend. No Q/K or probability tensors are retained
by the backend policy. OOM and arbitrary execution errors are not silently caught.

No additional flash-attn installation required.
Only PyTorch SDPA is used; no flash-attn or xFormers attention dependency is added.
Online Teacher execution and its kernel policy are untouched.

## Mathematical contract

- QKV projection and reshape remain [3,B,H,N,D]; kernel inputs remain [B,H,N,D].
- q_norm/k_norm, positional RoPE, output projection and projection dropout keep
  their order and modules. Residuals and block normalization remain upstream.
- scale is explicitly the original head_dim**-0.5; is_causal=False.
- Dropout is attn_drop.p in training and exactly zero in evaluation.
- Official [B,N,N] masks are broadcast to [B,1,N,N], equivalent to copying across
  heads. Boolean True means allowed; additive masks keep their original values.
  No separate padding mask is introduced. Baseline forwards do not add masks.
- Eager preserves the original no-mask equation and supports masks for comparison
  with the original fused path. (Upstream's unused eager path ignores masks.)
  Empty rows are handled safely in eager; native SDPA retains the installed
  PyTorch version's mask behavior.
- Input/autocast dtype is preserved. No forced BF16 conversion is added.
- Existing DA3 Q/K normalization hooks, RoPE reconstruction, view-order restoration
  and JS/KL loss are unchanged. C/E still extract Q/K before kernel execution;
  Flash does not need to return an N x N probability matrix.
- Checkpoint keys, parameter identities and trainability are unchanged. All data,
  loss weights, seed, optimizer, schedule, LoRA and freezing settings are unchanged.
  Batch size remains 1 and gradient accumulation remains 4 in the baseline configs.

## Lightweight verification

Run in the corresponding baseline checkout with the existing environment and
pinned DA3 source installed. No checkpoint/data loading or download occurs:

```bash
python scripts/check_attention_backend.py --config configs/baseline.yaml
python scripts/check_attention_backend.py --config configs/baseline.yaml --backend flash
python scripts/check_attention_backend.py --config configs/baseline.yaml --backend eager
python -m pytest tests/test_attention_backend.py -q
```

The script compares eager against the requested backend for no mask, boolean mask
and additive mask; checks finite forward/backward, actual official DA3 attention,
Q/K hooks, parameter/state identity; and prints profiler operators. For Flash it
asserts aten::_scaled_dot_product_flash_attention appears. A successful default
auto test with EFFICIENT_ATTENTION is explicitly not proof of Flash execution.
If a Flash version rejects explicit masks, only those masked comparison cases use
a labeled SDPA fallback; the unmasked probe and official DA3 probe remain strict.
Use --da3-root to inspect an existing source checkout; it is read without writing
Python bytecode there. The test suite accepts DA3_SOURCE_ROOT for the same purpose.

Local evidence: Windows RTX 4060 Laptop, BF16, PyTorch 2.3.1 and 2.7.1+cu128.
Both builds lack a usable Flash kernel and execute EFFICIENT_ATTENTION.
Profiler contains aten::_scaled_dot_product_efficient_attention and its backward.
No-mask max/mean absolute error: 0.0078125 / 0.0006808718.
Boolean and additive mask max/mean: 0.0078125 / 0.0007042475.
No NaN/Inf. Adapted official forward versus original fused forward max error: 0.
No full DA3 checkpoint forward, long training, cache generation or full evaluation
was run. Actual Flash execution still needs the server's compatible Linux build.

The repository-pinned PyTorch 2.3.1 environment passes:

```bash
python -m pytest tests/test_attention_backend.py tests/test_crossclip_student.py tests/test_attention_distillation.py tests/test_crossclip_config_and_entrypoints.py tests/test_sequence_evaluation.py -q
# 61 passed
```

## Server commands and timing

A requires no training:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_da3_small_baseline.py --config configs/baselines/A.yaml
```

Run each command in its matching baseline branch checkout; these are instructions
for the user, not jobs launched as part of this change:

```bash
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/B.yaml
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/C.yaml
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/D.yaml
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/E.yaml
```

Add --dry-run for the existing single-batch forward/backward audit first.
F/G have no valid training command and remain blocked pending their implementation.
Their future method still uses E's checkpoint.

Existing training timing remains enabled with its original frequency. timing.jsonl
and TIMING console records include iteration_wall_ms, forward/backward/optimizer
milliseconds, clips_per_second, peak_gpu_allocated_bytes and peak_gpu_reserved_bytes.
Throughput is per micro-batch including its data wait, not per optimizer update;
memory peaks are the process peaks since CUDA statistics were last reset.
Existing evaluation/visualization speed reporting is preserved.
