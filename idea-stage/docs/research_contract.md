# Research Contract: VGGT-to-MASt3R V1

## Selected Idea

- **Description**: 使用冻结 VGGT-Omega base 对每个 SCARED frame 独立生成可复用 cache，再用 DUNE+MASt3R binocular architecture 组合两帧训练。
- **Source**: 用户于 2026-08-22 更新的 teacher cache 协议。
- **Selection rationale**: 消除 LoRA 和 teacher sequence-length conditioning，令 2/8 帧样本复用同一批 frame-local pseudo labels。

## Core Claims

1. 显式 two-view cross-attention 可能改善 SCARED reference-view depth accuracy。
2. 官方 MASt3R binocular decoder/point head 可能降低 14-pixel ViT grid artifact。
3. 两个 teacher pointmap 分别位于各自 camera-local 坐标系；本协议不声称提供跨帧 pose、pair-local 融合或全局重建监督。

## Method Summary

Teacher 使用冻结 pretrained VGGT-Omega base，每次只处理一个 frame，并保存 FP32 depth/local points/confidence。2 帧或 8 帧样本在 reader 中组合独立 frame cache。Student 从官方 joint checkpoint 初始化，DUNE encoder frozen，MASt3R decoder/head trainable；双向 2B 解码后暴露 `pts3d_ref` 和 `pts3d_other_local`。

Loss 仅包含 confidence-weighted pair point distillation 与 reference-frame SCARED GT depth。第二帧自身 depth 必须通过 reverse-pair reference output 获得。

## Experiment Design

- **Datasets**: SCARED train/test，448x560，pair stride 2。
- **Baselines**: 保留的 DUNE/Distill3R baseline。
- **Metrics**: AbsRel、SqRel、RMSE、RMSE-log、delta1/2/3；patch artifact ratio。
- **Key hyperparameters**: point 1.0，GT depth 0.1，AdamW 1e-4，batch 4，DUNE frozen。
- **Compute budget**: 未指定；本地只运行 unit/import/single-batch smoke。

## Baselines

| Method | Dataset | Metric | Score | Source |
|---|---|---|---|---|
| DUNE/Distill3R baseline | SCARED | Endo3R depth metrics | 待现有实验结果 | project baseline |

## Current Results

| Method | Dataset | Metric | Score | Notes |
|---|---|---|---|---|
| V1 implementation | synthetic/mock | contract tests | 17/17 passed | 非科学结果 |

## Key Decisions

- 不从 `dune_vitsmall14_336.pth` 随机初始化 decoder；使用官方 joint checkpoint 加固定 DUNE-S/14 448 backbone。
- 不从独立 frame cache 构造虚假的 `pts3d_other_in_ref` 或共享 global 坐标。
- 不加载 teacher LoRA；旧 pair-cache checkpoint 不跨协议 resume。
- V1 不加入 descriptor/confidence/global alignment 等变量。

## Status

- [x] Idea selected
- [ ] Baseline reproduced
- [x] Main method implemented
- [ ] Representative dataset results
- [ ] Full dataset results
- [ ] Ablation studies
- [ ] Paper draft
