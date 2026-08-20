# 初始实验结果

**日期**：2026-08-20  
**计划**：`refine-logs/EXPERIMENT_PLAN.md`

## Results by Milestone

### M0：实现 — COMPLETE

- 新增并固定官方 MASt3R、DUSt3R/CroCo 与 DUNE submodule。
- 实现 strict pair dataset/cache、camera-from-world transform、官方 student adapter、双项 loss、trainer、Endo3R evaluator、patch metric 与 pair visualization。
- fresh-agent code review 首轮发现 2 个 blocker；修复后复审无 blocker。

### M1：Sanity — PASSED

| Run | 结果 | 状态 |
|---|---|---|
| 全仓库 pytest | 78 passed, 1 skipped | DONE |
| V1 focused pytest（项目 Conda env） | 18 passed | DONE |
| pinned source import | PyTorch 2.3.1, CUDA 12.1；MASt3R/DUNE import 成功 | DONE |
| compileall / git diff --check | 通过 | DONE |

### M2：真实 checkpoint / sample cache smoke — PENDING SERVER

- 本机没有 `dunemast3r_cvpr25_vitsmall.pth`、`dune_vitsmall14_448.pth`、SCARED、VGGT-Omega 或 LoRA checkpoint。
- 未执行完整 teacher cache、真实 `--dry-run` 或长期训练，符合用户执行边界。

### M3：正式训练与主结果 — NOT STARTED

- Depth accuracy、真实 patch artifact ratio 与 pair geometry 结论尚无数据。
- 当前科学结论：**inconclusive（尚未运行真实实验）**。

## Summary

- Must-run：2/6 完成；4/6 等待服务器资产/训练。
- Nice-to-have：2/2 已实现代码，0/2 有真实结果。
- Main result：尚不能判断 two-view cross-attention 是否改善目标指标。
- Ready for `/auto-review-loop`：NO；需先完成 R003-R006。

## Next Step

在服务器依次运行 full environment verify、少量 pair cache smoke、真实 `--dry-run`，确认后再启动完整 cache/训练/evaluation。

