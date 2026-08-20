# VGGT-to-MASt3R V1 实验计划

**分支**：`vggtomast3r`  
**基线**：`origin/feature/distill3r-student@b311d13db39ab6f75499449588b3429350e7a309`

## 科学问题

在 teacher、SCARED 数据协议、448x560 分辨率与 loss 尺度约定不变时，
以官方 DUNE-S/14 encoder + MASt3R binocular decoder/point head 替换单帧
Distill3R dense head，显式 two-view cross-attention 是否改善 depth accuracy、
14-pixel patch/grid artifact 与两帧三维一致性。

## 里程碑与运行顺序

1. M0 实现：固定官方 submodule，实现 pair dataset/cache、坐标转换、student、loss、trainer/evaluator。
2. M1 Sanity：synthetic/mock unit tests 与官方 source import。
3. M2 服务器 smoke：加载两个官方 checkpoint，运行 448x560 forward；生成最多数个 pair cache。
4. M3 Dry-run：真实 SCARED/cache 单 batch forward、loss、backward、optimizer。
5. M4 正式实验：完整 pair cache、训练、Endo3R depth 与 patch artifact evaluation。

## Runs

| Run | 优先级 | 内容 | 成功标准 |
|---|---|---|---|
| R001 | MUST-RUN | 13 类 synthetic/mock contract tests | 全部通过，无真实数据/checkpoint 依赖 |
| R002 | MUST-RUN | pinned MASt3R/DUNE import | 只从 `external/` 固定源码导入 |
| R003 | MUST-RUN | official checkpoint 448x560 forward | 两个输出均为 `[1,448,560,3]`，DUNE frozen |
| R004 | MUST-RUN | 少量 LoRA teacher pair cache | schema/version/坐标断言通过 |
| R005 | MUST-RUN | `train_vggtomast3r.py --dry-run` | dataset/cache/model/forward/loss/backward/optimizer 全通过 |
| R006 | MUST-RUN | 完整训练与 Endo3R evaluation | 报告 AbsRel/SqRel/RMSE/RMSE-log/delta1/2/3 |
| R007 | NICE-TO-HAVE | patch artifact measurement | 输出 boundary/non-boundary gradient 与 ratio |
| R008 | NICE-TO-HAVE | pair visualization | 固定深度范围 panel 与 reference-camera PLY |

## 固定设置

- SCARED train/test，ordered `(t,t+2)`，pair step 1，不随机换向。
- 448x560，patch 14，token grid 32x40。
- teacher：相同 VGGT-Omega base + 当前 rank-4 LoRA，完全 frozen。
- student：官方 `dunemast3r_cvpr25_vitsmall.pth`；DUNE-S/14 frozen；decoder/head trainable。
- loss：`1.0 * teacher_pair_point + 0.1 * reference_SCARED_depth`；其他 loss 禁用。
- optimizer：AdamW，lr 1e-4，weight decay 0.05，cosine，5% warmup，grad clip 1.0。
- batch 4 pairs/GPU，AMP auto，YAML 可调。

## 预算与执行边界

用户未指定正式 GPU-hours。当前机器只允许运行 static inspection、unit/import tests
和至多单批 smoke；不运行完整 teacher cache 或长期训练。R003-R008 中涉及真实权重、
数据和训练的部分须在服务器验证与排期。
