# VGGT-to-MASt3R V1 实验代码审查

**日期**：2026-08-20  
**审查方式**：fresh-agent GPT-5.6-Sol，xhigh；修复后复读当前工作树。

## 最终结论

**BLOCKING：无。** 当前实现可以进入受用户边界限制的 sanity/server smoke；真实 checkpoint、SCARED 与完整 cache/训练仍须服务器验证。

## 首轮 blocking findings 与修复

1. **无 GT sequence 会中断 evaluation，且全数据深度缓存会消耗数 GB RAM。**
   - 修复：先按现有 Endo3R 逻辑 preflight/跳过无 `data/depth` 的 sequence；按 sequence 推理、评分并释放 depth accumulator。
2. **pair cache 只检查字段存在，可能把 base 或错误 LoRA cache 用于主实验。**
   - 修复：强制检查 cache version、camera convention、`teacher_variant=lora`、LoRA checkpoint provenance、frame identity、stride、resolution 与 pointmap shape。

## 复审中修复的 non-blocking findings

- 保留旧 `losses.__all__` / Distill3R API。
- `training.log_every` 生效。
- checkpoint 保存/恢复 Python、NumPy、Torch、CUDA 与 DataLoader generator RNG 状态。
- patch artifact 基于每帧聚合后的 depth，并使用 SCARED GT-valid mask；不再重复加权内部帧。
- environment verify 同时检查实际 DUNE patch size 14、448x560 output shape 与 finite output。
- 当 `lambda_supervised_depth>0` 且整 epoch 无有效 GT pixel 时 fail fast。
- partial gradient-accumulation window 使用实际 window size，不再 under-scale。
- evaluator 强制 `evaluation.protocol=endo3r`。

## 剩余低风险事项

- 可选 `--max-steps` debug 提前返回时不写 checkpoint；正式训练、dry-run 与 epoch checkpoint 不受影响。
- 真实联合 checkpoint/DUNE checkpoint、CUDA kernel、SCARED 数据和 VGGT-Omega+LoRA sample cache 未在本机执行，符合用户明确执行边界。

## 审查确认正确的核心语义

- ordered `(t,t+2)`、448x560、32x40 DUNE token grid。
- camera-from-world 变换与 `xyz_global_b -> camera A` target。
- `pts3d_ref` / `pts3d_other_in_ref` 均在 reference-camera 坐标。
- second-view depth 只由 reverse-pair `pts3d_ref[...,2]` 得到。
- evaluation 使用真实 SCARED GT 与现有 Endo3R scene aggregation。
- DUNE frozen/eval/optimizer-excluded；仅 MASt3R decoder/head trainable。
- objective 仅为 confidence-weighted pair point loss 与 reference GT depth。

