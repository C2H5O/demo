# DA3 内窥镜蒸馏实验清单（2026-09-08）

## 实验定义与当前证据

所有字母是本轮统一命名，历史“Experiment B”指 attention 实验，对应下表 C。用户报告：伪标签蒸馏相对原始模型提升，attention 相对伪标签蒸馏可视化提升；attention 当时训练约 10–11 轮，完整定量评估待完成。不能把这些陈述写成已核查的最终论文数值。

| Baseline | 方法 | 训练 | 对照目的 | 分支 |
|---|---|---|---|---|
| A | 官方 DA3-Small，SCARED 零样本推理 | 无 | 域适应前起点 | codex/baseline-a-official-da3 |
| B | 伪标签深度 + 相机 + 原高光 + smoothness | 20 轮 | 基础蒸馏；不是仅 depth loss | codex/baseline-b-pseudo-distill |
| C | B + attention 蒸馏 | 20 轮 | C−B：attention 贡献；当前在跑的方法 | codex/baseline-c-attention-distill |
| D | B 的高光项替换为总开角 10° 平滑容忍 | 20 轮 | D−B：单独高光贡献 | codex/baseline-d-pseudo-soft-highlight |
| E | C 的高光项替换为总开角 10° 平滑容忍 | 20 轮 | E−C：联合方法的高光贡献；E−D：attention 贡献 | codex/baseline-e-attention-soft-highlight |
| F | E 权重 + Spark3R 论文参考算法 | 不重新训练；待实现 | 原始加速参考 | codex/baseline-f-spark3r-reference |
| G | E 权重 + 高光可靠性/窗口锚点感知 KV 策略 | 不重新训练；待实现 | G−F：修改的价值；G−E：速度精度取舍 | codex/baseline-g-spark3r-proposed |

F/G 是规划分支，配置会主动拒绝训练、评估和学生可视化，直到真正实现加速。不能将其计入已完成实验。A 也拒绝训练。B–E 配置可用于服务器 dry-run，本轮没有启动正式训练。

## 10° 的准确含义与原公式

设 θ 为表面法线与相机观察方向的无向夹角。原式为 `(1−|cos θ|)^2`，是法线趋向平行观察方向的软先验，不是法线垂直观察方向的硬限制。

新配置 `highlight_cone_full_angle_degrees: 10` 表示整个容忍锥的开角为 10°，计算半角为 5°，绝非每侧 10°。新损失为

`[s · softplus((cos(5°) − |cos θ|) / s)]²`，`s = 0.0005`。

锥内有很小但非严格为零的惩罚，锥外连续增强；这是原目标的替代形式，不是另叠加一个高光损失。B/C 保留原公式，D/E 只改变公式，权重均为 0.01，smoothness 权重仍为 0.1。新公式通常进一步降低高光项数值，不能声称它自动解决低占比问题。必须同时看有效像素、梯度和高光区域误差。既有 x3 权重配置保留为独立辅助实验，不混入主表。

旧日志审计：去重后 7843 个 step，555 个重复冲突记录；高光加权占比约 0.2464%→0.3769%，smoothness 约 0.0510%→0.1248%，不是精确零。低标量占比不能直接推导没有训练作用，详见 loss_audit_20260908.md。

## 路径与 RTX4090 24GB 设置

共用配置 `configs/baselines/_4090_common.yaml`：

- 官方 Student：`/public/home/2024141520249/Documents/Projects/vggtoda3/checkpoints/da3-small/model.safetensors`，同目录 `config.json`。
- Teacher：`/public/home/2024141520249/Documents/Projects/vggtoda3/checkpoints/vggt_omega/vggt_omega_1b_512.pt`。
- 缓存：`/public/home/2024141520249/Documents/Projects/vggtofast3r/data/teacher_cache_crossclip_base_raw_448x560`。
- 训练 processed 根：`/public/home/2024141520249/Documents/datasets/vggtodistilldata/processed/SCARED`。沿用数据集元数据中的 teacher_rgb、student_rgb 路由，不对二者误用同一张预处理图。
- 原始测试 RGB/GT/相机根仍是 `/public/home/2024141520249/Documents/datasets/vggtodistilldata/scared`。这是测试几何路径，与 processed 训练根分开；服务器需核查真实目录。

缓存元数据的 checkpoint identity 保留旧配置字符串 `./checkpoints/vggt_omega/vggt_omega_1b_512.pt`；实际加载权重使用上面的绝对路径。此字段用于严格身份匹配，不是第二个加载路径。尚未读取服务器缓存元数据，若 dry-run 报身份不匹配，先核查缓存真实来源，不能简单关闭验证。

统一 micro-batch=1、梯度累积=4（每卡名义有效 batch=4）、BF16、Student head_chunk_size=1、在线 Teacher batch=1、num_workers=0。训练保持 16 帧、既有输入尺寸和学习率/优化器/种子，B–E 都从官方初始化重新训练以保证一致性。不要把旧实验 checkpoint 直接续到不同目标或不同 batch 配置里。梯度累积与一次 batch=4 的损失归约并不保证数值完全一致。

这些是保守设置，尚未在 4090 上实测峰值显存，不保证不会 OOM。attention 仍需在线 Teacher 提取 Q/K，缓存仅省去伪标签几何推理；先跑下面的 backward dry-run。若 OOM，先减 attention query_chunk_size 128→64 并检查结果等价，不擅自降低帧数/分辨率或改变 Teacher。dry-run 通过也不是长跑无 OOM 保证，要观察首个 epoch 的峰值和长序列评估峰值。

## 可复制命令（Linux，单卡）

先进入 `/public/home/2024141520249/Documents/Projects/vggtoda3` 并激活已有环境。不要同时在同一个 checkout 切分支并运行多个实验；并发运行应使用独立 clone/worktree。

```bash
git fetch origin
git switch codex/baseline-a-official-da3
CUDA_VISIBLE_DEVICES=0 python evaluate_da3_small_baseline.py --config configs/baselines/A.yaml
```

B：
```bash
git switch codex/baseline-b-pseudo-distill
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/B.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/B.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/baselines/B.yaml --checkpoint outputs/baseline_B/last.pt --split test --protocol vda
```

C：
```bash
git switch codex/baseline-c-attention-distill
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/C.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/C.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/baselines/C.yaml --checkpoint outputs/baseline_C/last.pt --split test --protocol vda
```

D：
```bash
git switch codex/baseline-d-pseudo-soft-highlight
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/D.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/D.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/baselines/D.yaml --checkpoint outputs/baseline_D/last.pt --split test --protocol vda
```

E：
```bash
git switch codex/baseline-e-attention-soft-highlight
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/E.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 python train_direct_teacher_distillation.py --config configs/baselines/E.yaml
CUDA_VISIBLE_DEVICES=0 python evaluate_crossclip_projection.py --config configs/baselines/E.yaml --checkpoint outputs/baseline_E/last.pt --split test --protocol vda
CUDA_VISIBLE_DEVICES=0 python visualize_crossclip_projection.py --config configs/baselines/E.yaml --source student --checkpoint outputs/baseline_E/last.pt --split test --sequence-index 0
```

F/G 不提供伪造的加速运行命令；实现后使用同一个 E checkpoint，无须新训练。恢复同配置的中断任务时，可在对应训练命令加 `--resume outputs/baseline_E/last.pt`（其他字母相应替换）。不同 loss 配置禁止混合恢复。

## 需要补齐的证据与执行顺序

1. 先完成当前 attention 训练及完整评估；原始 A、旧伪标签、旧 attention 都用同一新版完整序列协议重评，历史 clip 结果不能直接与新结果并表。
2. 跑匹配资源/训练预算的 B–E；报告完整测试集 SCARED 8/9 的逐序列 AbsRel、RMSE、δ1、TAE、有效帧/帧对覆盖率和平均每帧推理耗时。学生完整序列接口内部使用 32 帧窗口，训练仍为 16 帧。
3. 在训练数据中预先划分验证序列以选择角度、权重、KV 预算；不使用 8/9 测试集调参。至少 3 个随机种子复跑关键 B/C/E 对照，报告均值、标准差和序列级配对差值；不要将相邻帧当独立样本制造显著性。
4. 高光必须有区域证据：有效 GT ∩ 高光区域、邻域、非高光区域误差及覆盖率；若饱和区没有可靠 GT，明确缺测，增加可核查几何参考或独立标注，不能用 Teacher 伪标签当真实精度证据。
5. 辅助消融：去除高光、原式、总开角 0/5/10/20°（当前默认 10°）、固定公式的权重扫描；所有调参在验证集。记录 raw/weighted loss、各层梯度范数与方向、有效法线比例。必要时将 smoothness on/off 独立做对照。
6. 公平外部 baseline：实际评估官方 VDA、DA3-Small；加入一个能取得权重且协议兼容的内窥镜方法（例如 EndoDAC）后再定最终列表，不能凭文献数值与本地 SCARED 协议直接比较。报告 Teacher 参数量与缓存/在线训练成本。
7. 图：同帧同色标 RGB/GT/A/B/C/E 深度及误差；高光局部放大、mask 与法线角度；运动/遮挡/强高光失败案例；跨窗口边界时间曲线和视频；各层注意力对齐示例；E/F/G 延迟-精度/TAE Pareto 图及 KV 选择图。无可靠全局位姿时不把独立局部点云拼接称为全局重建。

## Spark3R 改造方案与风险

见 SPARK3R_PROPOSAL.md。三项目前是拟议贡献，不等同于已经证明的独立创新。最容易受到质疑的是：通用 attention 蒸馏缺乏特定机制；原高光本来就是软惩罚，新增容忍角是否只是超参数；Spark3R 小改动是否只是启发式组合；高光 GT 缺失与 Teacher 偏差；短序列实际加速不足；训练预算和评估对齐不公平。必须分别用上述正交消融、可靠区域证据、等 KV 预算和端到端速度实验回应。
