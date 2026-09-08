# Flash SDPA audit (before implementation)

Baseline C source: 2307d27; E source: 1cac491 (same implementation, different
baseline selector/description). Training entrypoint delegates
to trainers/direct_teacher_distillation_trainer.py. C/E inherit _4090_common.yaml.

## Existing computation

- Student: strict official DA3-Small safetensors load, then ray execution disabled.
  Local external/Depth-Anything-3 DINOv2 Attention.forward uses PyTorch SDPA
  with fused_attn=True by default, otherwise explicit QK/softmax/V. Q/K norm
  precedes RoPE. Dropout and any attention mask belong to the upstream forward.
  The training wrapper passes no attention/frame mask. Frames are restored after
  DA3 reference-view permutation in the capture path.
- Teacher: strict VGGT-Omega load, frozen/eval online instance; prediction heads
  and aggregator output caches are unused in this existing attention-only path.
  Local source inspected at D:/Projects/vggt_omega_distill/external/vggt-omega.
  SelfAttention.compute_attention already calls PyTorch SDPA, no dropout or bias.
  Frame blocks use RoPE, selected inter-frame global blocks do not.
- Capture: forward hooks on q_norm/k_norm. Teacher detaches patch Q/K and casts
  to configured float16; Student replays RoPE and restores original frame order
  with autograd intact. Neither hook requests attention weights or disables SDPA.
- Shapes for B=1, F=16, 448x560: Student patch Q/K [1,16,6,1280,64]
  (ViT-S/14); Teacher patch Q/K [1,16,16,980,64] (patch size 16).
  Teacher patch-overlap alignment produces [1,16,16,1280,64]. Special tokens
  remain in backbone attention but are excluded from distillation as before.
- Mapping: Teacher [4,11,17,23] -> Student [5,7,9,11]. Adjacent directed frame
  offsets [-1,1], temperatures 1/1, per-head softmax followed by head mean,
  additive epsilon 1e-6 and renormalization, JS divergence (KL option retained),
  mean over batch/query/valid frame pairs and then equal mean over layers.
  Teacher/Student scaling uses each tensor's own head dimension.
- Loss already query-chunks at 128 with complete per-target-frame keys and
  non-reentrant checkpoint per chunk. It does NOT form a full clip [B,H,N,N].
  Backward recomputes chunk probabilities, avoiding retention of all probability
  matrices. Unequal final chunks are weighted by actual query count.
- Teacher forward uses no_grad plus BF16 autocast; capture output remains FP16.
  Student uses BF16 autocast; relation logits/probabilities/divergence use FP32.
  The local DA3 source has no explicit block checkpoint call; the loss does.
  Preserve production backward gradient auditing for installations using
  checkpointed backbones.

## Risks and implementation decision

SDPA API use alone does not establish the selected CUDA kernel. Local runtime is
PyTorch 2.3.1 on RTX 4060 Laptop, not the target 4090. Server checkpoint/data paths
in the configs are not local. Full 16-frame backbone activations, captured Q/K,
Teacher weights, and math fallback remain possible OOM sources.

Keep upstream attention computation intact. Add an opt-in, instance-local SDPA
call policy using TorchFunctionMode inside the existing attention method. Select
Flash, then efficient, then optional math using actual Q/K/V eligibility; enforce
one backend per call and log it after execution. No global functional patch,
third-party dependency, projection replacement, or model source copy. Wrapping
the attention method also covers checkpoint recomputation. C/E opt in only.

Extract the existing checkpointed query-chunk sum into a named helper without
changing its math. Remove avoidable contiguous copies before capture casts or
frame indexing. Add independent full-matrix loss/gradient references and small
upstream module SDPA equivalence tests. Preserve all experiment/data settings.

## Implemented and validated

- C/E opt in; common config defaults off. Existing query chunk size remains 128.
  Runtime CUDA eligibility sees the same autocast input dtype as SDPA (FP32
  LayerNorm Q/K can become BF16 at the SDPA boundary). Captured Q/K are unchanged.
  Backend policy lives on model instances, leaves functional SDPA and upstream
  files untouched, and preserves state_dict keys. First successful call for
  each module/layout logs its enforced backend; fallbacks warn with shape,
  dtype, layout and PyTorch eligibility diagnostics.
- Named query_chunk_divergence_sum retains JS/KL, per-model scaling, head mean,
  epsilon, temperatures, offsets, mapping and reductions. All target-frame keys
  participate in each softmax. Checkpoint discards probabilities and recomputes
  them in backward. A saved-tensor test checks probabilities are not retained.
- Removed contiguous copies immediately before Teacher dtype conversion and
  Student frame-order indexing. No full attention maps or Q/K disk writes added.
- Successful dry runs now print peak CUDA allocated/reserved bytes, measured
  since before model loading. No optimizer/scheduler/training behavior changed.

Validation environment: Python 3.10, PyTorch 2.3.1, RTX 4060 Laptop GPU, Windows.
The local PyTorch build reports "Torch was not compiled with flash attention".
Both actual upstream attention modules execute efficient_sdpa in BF16 CUDA tests;
CPU tests execute math_sdpa. CUDA Flash output equivalence is NOT verified on
this build. Eligible Linux CUDA builds select Flash first using the same tested
single-backend context. The 4090 training backend still requires server dry run.
PyTorch backend controls: https://docs.pytorch.org/docs/stable/backends

67 targeted tests passed (7.91 seconds). Tests include original attention loss,
online frozen-Teacher flow, production backward audit, Student contract, baseline
loss/config regressions, and 20 new kernel/chunk checks. Exact command:

```bash
python -m pytest tests/test_flash_attention.py tests/test_attention_distillation.py tests/test_crossclip_student.py tests/test_crossclip_config_and_entrypoints.py tests/test_direct_teacher_distillation_loss.py -q
```

Set DA3_ATTENTION_SOURCE and VGGT_ATTENTION_SOURCE to the installed upstream
attention.py paths to run the four real-module tests (otherwise explicitly
skipped). These tests load only the attention module source, not checkpoints.

Independent dense-reference comparison uses B=1, F=3, N=129 (unequal final
chunk), Teacher H=4/D=64 and Student H=2/D=32, temperatures 0.8/1.3, chunks
16/32/64. FP32 arithmetic is retained even for BF16 Q/K, as in production.
Maximum absolute errors across tested chunks:

| Input dtype / divergence | Loss | Student Q/K gradient |
| --- | ---: | ---: |
| FP32 / JS | 7.4505806e-9 | 1.4551915e-11 |
| FP32 / KL | 5.9604645e-8 | 4.3655746e-11 |
| BF16 / JS | 1.4901161e-8 | 2.3841858e-7 |
| BF16 / KL | 2.9802322e-8 | 1.4305115e-6 |

Fixed-weight eval attention output comparisons:

| Module / reference -> selected | Max absolute error | Mean absolute error |
| --- | ---: | ---: |
| DA3 CPU explicit -> math SDPA | 1.1920929e-7 | 1.8201304e-8 |
| Teacher CPU math -> math SDPA | 0 | 0 |
| DA3 CUDA BF16 explicit -> efficient SDPA | 0.001953125 | 0.000311937707 |
| Teacher CUDA BF16 math -> efficient SDPA | 0.001953125 | 0.0003195912 |
| Synthetic CUDA BF16 math -> efficient SDPA | 0.0078125 | 0.000493524421 |

Small module tests also run checkpointed backward with Q-norm hooks active.
Synthetic CUDA Q/K/V gradients meet atol=3e-6, rtol=0.05. CPU dropout/mask/scale
comparison is exact with fixed seed, and strict no-math fallback is tested.

Both requested C/E --dry-run commands were attempted. They stop during dataset
construction: configured SCARED and Teacher cache paths do not exist locally.
Server checkpoints are also absent from the worktree. No full model forward,
complete dry-run success, 4090 backend, or dry-run peak memory is claimed.

Remaining OOM risks: 16-frame activations and captured Q/K coexist with frozen
Teacher weights; math fallback can still allocate quadratic backbone attention;
Teacher alignment and checkpoint metadata are not free. Flash/query chunking
does not prove 24GB fit. On server OOM, change only query_chunk_size 128 -> 64
-> 32; never reduce frames, resolution, layers, or weights. No automatic OOM
retry changes experiment settings. No full training or downloads were performed.

Experimental definitions are unchanged: C remains B + attention distillation;
E remains C + soft-highlight. This is not an F/G inference method component.
