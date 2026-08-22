# Teacher cache 推理审查

**日期**：2026-08-22
**对象**：VGGT-Omega + rank-4 LoRA 的 VGG-to-MASt3R pair cache
**审查方式**：本地代码/数值审计 + fresh `gpt-5.6-sol` ultra reviewer
**review_independence**：`same-family`
**acceptance_status**：`provisional`

## 结论

附件不能证明 teacher 已损坏。`depth_a_local.npy` 数值完整、非 NaN、非
常数；附带 PNG 近乎全黑的直接原因是把实际 `0.696–1.153` 的值固定映射到
`0.1–10.0` 色域，全部结构只占色条约 4.6%。

但当前实验存在两个已经确认的 blocking 问题：

1. LoRA teacher 在连续 8 帧上训练，legacy cache 也按 8 帧推理；V1 pair
   cache 却只输入 `[t,t+2]` 两帧。这会改变 inter-frame attention 的条件分布，
   并违反实验计划中“teacher 设置不变、只替换 student”的约束。
2. cache 只记录可变路径 `./checkpoints/teacher_lora/last.pt`，不记录 checkpoint
   内容摘要和完整推理指纹；文件被覆盖后，旧 cache 仍会被 `skip` 并静默复用。

因此，当前 cache 的准确状态是：**数值可读，但 teacher 协议与来源未验证；只可
用于调试，不可用于最终主结果或“只改变 student”的结论。**

现有证据还不能确认“2 帧导致平滑”或“LoRA 已坍缩”。这两项必须由同帧
`base/LoRA × 2/8-frame` FP32 A/B 决定。

## 附件数值审计

| 项目 | 结果 |
|---|---:|
| shape / dtype | `448×560 / float32`（由 cache fp16 读回） |
| finite fraction | `1.0` |
| min / max | `0.695801 / 1.153320` |
| mean / std | `0.982278 / 0.106195` |
| p1 / p50 / p99 | `0.725098 / 0.993652 / 1.145508` |
| gradient mean / p95 / max | `0.001346 / 0.003020 / 0.012278` |
| 平面拟合 R² / residual RMSE | `0.487 / 0.0761` |
| 高频能量，FFT radius > 0.05 | `0.98%` |
| unique depth values | `781 / 250,880` |
| 水平相邻值完全相同 | `53.2%` |
| median abs(dx) / abs(dy) | `0 / 0.0009765625` |

按 p1–p99 自适应拉伸后能看到连续的低频几何轮廓，说明 PNG 的黑暗主要是显示
问题；但深度本身确实偏平滑。没有同帧 RGB、GT、base-teacher 和 8 帧输出时，
不能区分合理的组织表面、模型域偏移和 teacher 推理退化。

绝对尺度约 1.0 也不是单独的错误证据：LoRA 自监督目标没有 metric scale anchor，
当前 student point loss 又会做 pair scale normalization，SCARED 监督负责学生的
metric scale。

### fp16 判定

cache 在 FP32 teacher forward 后将 depth/xyz 转成 fp16。深度约 1 时 fp16 ULP
正好是 `0.0009765625`，与本图典型像素梯度同量级，因此亚 ULP 的细边缘会被抹掉。
这是已确认的**次级细节保真风险**，但不能解释约 0.46 的全局深度范围，也不是
全图暗或全局坍缩的主因。legacy cache 同样用 fp16，因此它也不是 V1 特有回归
的充分解释。

若最终声称改善 fine structure，应统一使用 FP32 cache，或证明 fp16 相对 FP32
的 depth-boundary F1 下降 `<0.02` 且下游主指标下降 `<1%`。

## teacher 推理链路审计

已确认正确：

- base checkpoint 使用 `strict=True` 加载；LoRA expected/provided key 集合严格
  相等。
- `[2,3,448,560]` 会由官方 `VGGTOmega.forward` 正确扩展为
  `[1,2,3,448,560]`，不存在 batch/frame 维错位。
- 官方预处理输出 `[0,1]` RGB，aggregator 内部再做 ImageNet normalization。
- dense head 在 FP32 中输出 `depth=exp(depth_logits)`。
- 附件 `depth_a` 是 unprojection 前的 raw depth，因此坐标反投影不能造成该图
  的平滑。

仍未验证：

- 严格 key 加载不能证明 LoRA 有效：`B` 可全零、`BA` 可极小、checkpoint 可来自
  错误 run，或 adapter 对最终输出的影响可小于 fp16 量化。
- V1 的两帧输入将真实 `[t,t+2]` 当成两个连续槽位，并删除了 `t+1` 及其余上下文。
  VGGT-Omega 明确执行跨帧 attention，因此输出理论上依赖序列长度和内容。
- cache 没有 checkpoint SHA、上下文帧、目标 slot、代码版本、preprocess 与
  autocast/dtype 的完整 fingerprint。

## 最小决定性远端实验

至少选择 20 个连续 8 帧窗口、40 个具有明显几何边界的目标帧。对每个窗口只
预处理一次得到 `X8`，再令 `X2=X8[[i,i+2]]`，避免把预处理差异混入上下文 A/B。

| 条件 | 权重 | 输入 | 取值 |
|---|---|---|---|
| A | base | `X2` | 两帧全部输出 |
| B | LoRA | `X2` | 两帧全部输出 |
| C | base | `X8` | 提取与 A 相同物理帧 |
| D | LoRA | `X8` | 提取与 B 相同物理帧 |

所有条件必须保存 cast 前 FP32 raw depth/pose/confidence/xyz 和 cast 后 fp16
round-trip 结果。同时记录：

- min/max/mean/std、p1/p50/p99、gradient mean/p95/max、total variation；
- `z(d)=log(d)-median(log(d))` 后的相关性、relative L1、SSIM；
- RGB Sobel 与 depth edge 的 correlation、top-q precision/recall；
- 有 GT 时的 median-scaled AbsRel、delta1、depth-boundary F1；
- confidence 分布及 FP32→FP16 的边缘/数值损失；
- base/LoRA resolved path、SHA-256、step/epoch、代码 commit；
- 每层 `||A||F`、`||B||F`、finite/nonzero 比例，以及关键的
  `||(alpha/r)BA||F / ||W||F`；
- adapter on/off 输出差异及目标 LoRA module 的 forward-hook call count。

再做两个控制：同一 `X8` 经过 legacy 和当前 wrapper 的 8 帧路径应数值一致；同一
物理目标帧应做 slot/order control，避免把位置敏感性误判为上下文长度效应。

预计诊断远低于 1 GPU-hour（具体取决于 GPU）；不需要训练。

## 建议预注册的 go/no-go gate

以下阈值是 reviewer 给出的工程建议，不是当前附件推出的事实。应按 clip 做 paired
bootstrap 95% CI。

定义：

```text
E_ctx = median_frames mean_pixels |z(depth_S2) - z(depth_S8)|
```

只有“结构改变”和“方向性变差”同时成立，才声称 S=2 materially harmful。

结构改变满足任一：

- `E_ctx >= 0.03` 且 CI 下界 `>0.02`；
- median correlation `<0.95`；
- `TV(S2)/TV(S8) <=0.85` 且至少 70% 帧同方向。

方向性质量变差满足任一：

- `BF1(S8)-BF1(S2) >=0.05`，且 CI 下界 `>0.02`；
- `AbsRel(S2)-AbsRel(S8) >=0.01` absolute、且至少 5% relative，CI 下界 `>0`；
- 无 GT 时，预注册 edge-alignment 至少下降 `0.05`，且至少 70% 帧变差。

接受 S=2 为无实质差异需同时满足：`E_ctx` CI 上界 `<0.01`、correlation
`>=0.98`、BF1 差异绝对值 `<0.02`、AbsRel 相对差异 `<2%`，若做 student
短程 matched run 则主指标差异 `<1%`。两组条件之间属于灰区。

LoRA “数值未激活”硬失败满足任一即可：

- 所有 `(alpha/r)BA` 精确为零；
- adapter on/off 的全部相关 FP32 输出均在 `rtol=1e-5, atol=1e-6` 内；
- 至少 40 帧中，on/off 对最终 cache 量的差异 p99 小于 `0.25 fp16 ULP`，且
  cast 后至少 99.9% 元素 bit-identical。

若 LoRA 确实改变输出，但 held-out 主指标改善 `<1% relative` 且 paired CI
包含 0，只能表述为 “no demonstrated benefit”，不能称为加载 bug。

## 结果到结论矩阵

| A/B 结果 | 允许的结论 | 不允许的结论 |
|---|---|---|
| A/B 平滑，C/D 恢复边缘并过 harmful gate | 2 帧上下文造成结构/质量回归 | LoRA 必然坍缩 |
| A/C 有结构，B/D 均明显变差 | LoRA adaptation 退化；检查 checkpoint/training | cache writer 是唯一原因 |
| LoRA `BA=0` 或 on/off 无数值差 | LoRA 对当前 cache 未激活 | LoRA 训练有效 |
| 四组尺度对齐后结构近似 | 当前主要是可视化误判；S=2 影响小 | 可继续声称 teacher 协议未变 |
| 四组均平 | base teacher/域适配/场景需进一步检查 | 仅凭一张图宣布 cache 损坏 |
| S=2 与 S=8 落在灰区 | 扩大样本或做 matched student run | 强结论 |

即使 S=2 最终质量无实质差异，它仍是新 teacher protocol。最终实验只能二选一：

1. 恢复 canonical 连续 8 帧 teacher forward，再提取 `t,t+2` 的所有关联输出；或
2. 明确把 S=2 声明为新协议，并对所有 baseline 使用同一协议重跑。

## 优先级行动项

| 优先级 | 行动 | 需要 GPU | 计算量 |
|---|---|---|---|
| P0 | 实现 fingerprint、严格 cache miss、固定+自适应可视化、FP32/FP16 统计 | 否 | 很低 |
| P0 | 实现 `base/LoRA × S2/S8` 单次诊断与 LoRA `BA`/hook 检查 | 代码否，运行是 | <1 GPU-hour |
| P0 | 运行至少 20 窗口 A/B 并按上述 gate 判定 | 是 | <1 GPU-hour 预期 |
| P1 | 根据结果选择 canonical S8 或正式 S2 protocol | 否 | 决策 |
| P1 | 用新 schema 重建 cache，并从相同干净初始化重训 student | 是 | 完整 cache+训练 |
| P2 | FP32 vs FP16 cache fidelity ablation | 是 | 小型推理/训练对照 |

不应在 A/B 前静默把生产默认从 S=2 改成 S=8；可以先实现两条路径。现有 cache
及其训练出的 student checkpoint 只可用于调试，不能作为 corrected run 的最终
模型或 warm start。

## 审查轮次与共识

- Round 1：reviewer 反驳“黑图=teacher 坏”，确认 8→2 协议变化和 cache provenance
  缺失为 blocking；要求完整 2×2 FP32 A/B。
- 本地回应：补充自适应图、低频/平面拟合和 fp16 ULP 证据；接受显示问题与细节
  保真风险，但不接受在无 RGB/GT 对照时宣布 teacher 坍缩。
- Round 2：reviewer 给出实施 gate 和预注册阈值；共识为先落地非语义诊断，生产
  teacher context 的默认变更等待 A/B。

完整 trace：`.aris/traces/research-review/2026-08-22_run01/`。由于 reviewer 为
同家族 Codex/GPT，本结论是 provisional review，不是 cross-family acquittal。
