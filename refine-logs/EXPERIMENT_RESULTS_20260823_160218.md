# Cross-clip teacher projection 当前结果

**日期**：2026-08-23
**计划**：`refine-logs/EXPERIMENT_PLAN.md`

## 实现结果

- 学生把 `[B,16,3,448,560]` 展平为独立帧，只提取 frozen DUNE blocks
  `[2,5,8,11]`，仅 Fast3R DPT head 可训练，输出 `[B,16,448,560,3]` local points。
- 数据索引覆盖每个合法 stride-one 起点，左/右均严格使用 15 个绝对同帧 target，
  不跨 sequence，边界只使用存在的一侧。
- frozen base teacher raw cache 与 offline aligned cache 分根；cache 保存 depth/local/global
  XYZ、confidence、mask、K、W2C extrinsic 与 provenance，并对相机/几何完整性 fail closed。
- 投影由 student XYZ 与 teacher K 生成采样 grid，teacher tensor detach，损失只包含
  projection、highlight surface、highlight-aware inverse-depth smoothness 三项。
- VDA 默认、Endo3R 保留；二者都对重叠窗口的同一绝对帧聚合后与 SCARED GT 评估。
- 可视化支持 teacher cache 或 student checkpoint，输出 fixed/adaptive depth、NPY、panel
  及每帧独立 camera-local PLY。

## 本地验证

| 检查 | 结果 |
|---|---|
| 全仓库 pytest | 114 passed, 2 skipped |
| crossclip focused pytest | 19 passed |
| Python 语法检查 | 通过 |
| cache/alignment/train/eval/visualization CLI help | 通过 |
| git diff check | 通过（仅 Windows LF→CRLF 提示） |
| fresh-agent 修复后复审 | APPROVE；无 blocking |

三条 warning 均为现有/新入口使用 `torch.cuda.amp.*` 的 FutureWarning，不影响当前功能。

## 尚未产生的科学结果

本轮未运行真实 VGGT-Omega teacher forward、完整 cache、训练 dry-run、正式训练或评估，
也未下载 checkpoint。因此：

- 没有可报告的真实 teacher adjacent-clip scale drift 数值；
- 没有 VDA/Endo3R 指标；
- 没有训练收敛或方法优于基线的证据；
- 当前研究结论为 **inconclusive / implementation-ready**。

下一步是在 GPU 服务器先生成少量 raw cache 并可视化，再运行 audit/alignment 和训练
dry-run；全部 sanity 通过后才生成完整 train/test cache 并正式训练。
