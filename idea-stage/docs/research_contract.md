# Research Contract: VGGT-to-MASt3R V1

## Selected Idea

- **Description**: 在保持 VGGT-Omega+LoRA teacher、SCARED protocol、分辨率和 supervision 不变时，以官方 DUNE+MASt3R binocular architecture 替换单帧 Distill3R head。
- **Source**: 用户提供的 V1 实验规范。
- **Selection rationale**: 单变量隔离 student architecture，直接检验 two-view cross-attention。

## Core Claims

1. 显式 two-view cross-attention 可能改善 SCARED reference-view depth accuracy。
2. 官方 MASt3R binocular decoder/point head 可能降低 14-pixel ViT grid artifact。
3. 两个 pointmap 在同一 reference-camera 坐标系中，可用于检验 pair-local geometry consistency；V1 不声称完成全局重建。

## Method Summary

严格输入 ordered `(I_t,I_{t+2})`。VGGT-Omega+LoRA teacher 同时处理两帧，缓存 A-local、B-local、world points，并把 B world points转换到 A camera。Student 从官方 joint checkpoint 初始化，DUNE encoder frozen，MASt3R decoder/head trainable，训练接口只暴露 `pts3d_ref` 和 `pts3d_other_in_ref`。

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
- 不使用 `pts3d_other_in_ref[...,2]` 作为 second-view depth。
- V1 不加入 descriptor/confidence/global alignment 等变量。

## Status

- [x] Idea selected
- [ ] Baseline reproduced
- [x] Main method implemented
- [ ] Representative dataset results
- [ ] Full dataset results
- [ ] Ablation studies
- [ ] Paper draft
