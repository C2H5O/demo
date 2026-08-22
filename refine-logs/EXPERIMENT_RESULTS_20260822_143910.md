# VGGT-to-MASt3R V1 当前结果

**日期**：2026-08-22
**计划**：`refine-logs/EXPERIMENT_PLAN.md`

## M0：实现 — COMPLETE

- 新增 frozen-base single-frame teacher cache generator；明确禁用 LoRA 与微调权重。
- 每个 source frame 独立推理并保存一个 FP32、版本化、带 provenance 的 NPZ。
- reader 可按有序 metadata 组合严格的 2 或 8 个 frame cache。
- cache 路径包含 dataset/keyframe/sequence/frame-index/frame-id，避免 manifest 碰撞。
- student、loss、trainer、默认 VDA、保留的 Endo3R 和 visualization 全部迁移到
  A-local/B-local 语义；旧 `B-in-A` target 不再使用。
- 旧 cache schema 与旧 checkpoint protocol 均 fail closed。

## M1：Sanity 与审查 — PASSED

| 检查 | 结果 |
|---|---|
| 全仓库 pytest | 95 passed, 2 skipped |
| focused frame-cache/V1 pytest | 31 passed, 1 skipped |
| fresh-agent 修复后复审 | 0 blocking, 0 non-blocking；reviewer 27 passed, 1 skipped |
| compileall | 通过 |
| frame/compatibility/visualization CLI `--help` | 通过 |
| git diff whitespace check | 通过（仅现有 LF→CRLF 提示） |

两条 pytest warning 来自现有 `torch.cuda.amp.autocast` deprecation，不影响本轮协议。

## M2：真实 teacher frame cache smoke — PENDING SERVER

本机缺少真实 SCARED、VGGT-Omega 权重与 CUDA 运行资产，尚未验证：

- official preprocessing 后 `[1,3,448,560]` 单帧 forward；
- 真实 depth/xyz/confidence 数值范围与可视化；
- 2/8 frame composition 对真实 cache 的读取。

## M3/M4：Dry-run、训练与评估 — NOT STARTED

- 尚无本协议下的训练 checkpoint 或 VDA/Endo3R 指标。
- 旧 pair-cache 协议 checkpoint 不兼容，不能 resume 或用于本轮评估。
- 当前科学结论仍为 **inconclusive**；现阶段只完成实现与本地验证。

## 下一步

在 GPU 服务器依次生成少量 frame cache、检查可视化、运行真实 `--dry-run`；通过后
再生成完整 train/test cache，并从本协议兼容的初始化开始训练。
