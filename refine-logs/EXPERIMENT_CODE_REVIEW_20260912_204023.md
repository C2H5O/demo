# Baseline-I 实验代码审查

审查者：独立 Codex 子代理，GPT-5.6-Sol（xhigh）

结论：**修复后无阻塞问题。**

## 首轮发现与修复

1. 首版 residual reduction 将 batch 内所有有效帧直接平均，使有效帧更多的 clip 权重更高。现已改为：每帧 confidence 加权均值 → 每个 sample 的有效帧均值 → 有监督 sample 均值。
2. 最小正 float32 depth 虽满足 finite/positive，直接 reciprocal 仍可能溢出。现已在 valid depth 上以 `eps` 做安全下限后再求倒数，invalid depth 不参与 reciprocal。
3. 首版 `stats/depth_confidence_fallback_ratio` 在 Baseline-I 中误用了 affine fallback。现已分别记录 confidence fallback 与 affine fallback。

## 复审确认

- 每个 sample 在完整 `[16,H,W]` 上只拟合一个 scale 和 shift，输出 shape 为 `[B]`；显式拒绝非 16 帧输入。
- disparity 和 weighted least-squares 使用 FP32；scale/shift 在 residual 前 detach。
- Smooth-L1 采用逐帧 confidence 加权归约，再按 sample 和 batch 平衡。
- 旧 `compute_direct_depth_distillation_loss` 未改语义，默认 `depth_mode=raw_depth_l1`。
- Baseline-I 配置从 Baseline-E 继承，除实验标识、输出路径与三个 depth 设置外相同。
- diff 未触及 attention、dataset、model、inference、VDA 或 TAE 实现。

## 验证

- 新 loss、旧 loss与 attention 定向测试：`41 passed`。
- 完整测试：`152 passed, 13 warnings`。
- 本机缺少 DA3 checkpoint、配置和 `depth_anything_3` 包，因此未运行真实 DA3/VGGT GPU dry-run、正式训练或 benchmark。

环境说明：基础 `D:\Anaconda` 环境的失败来自缺少 OpenCV/数值环境不匹配；项目指定的 `vggtodistill3r` 环境中全部通过。
