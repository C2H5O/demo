# Cross-clip teacher projection 实验代码审查

**日期**：2026-08-23
**方式**：fresh-agent GPT-5.6-Sol，xhigh；修复后限定复审。

## 最终结论

**APPROVE；无 blocking issue。** 核心模型、15+15 映射、边界/序列隔离、离线尺度
对齐、Student→Teacher 投影方向与严格三项损失均符合固定方法。

## 首轮 Major 与修复

首轮发现 cache generator 对 dense geometry、K、extrinsics 统一 `nan_to_num`，而 validator
只查 shape/dtype/finite，可能让退化的全零相机/cache 静默通过。

修复后：

- 只有已被 `valid_mask=false` 保护的 dense depth/XYZ/confidence 像素可显式清零；
- valid 像素非有限立即失败，K/extrinsics 任意非有限立即失败；
- 每帧检查 minimum valid fraction、positive valid depth、non-negative confidence、
  `xyz_local.z≈depth`、positive focal、K determinant/底行与 R determinant；
- metadata 记录并交叉核验逐帧 valid fraction、depth range 与 confidence mean；
- aligned cache 同步更新尺度相关 metadata。

## 首轮 Minor 与修复

1. `scale_alignment.enabled` 曾是死配置：现为 false 时明确拒绝执行 alignment。
2. normalized confidence 全零会关闭 projection：现按 sample 回退 uniform，并记录左右
   effective weight sum；新增测试覆盖。
3. 未使用且缺少 final crop width 的 flip-intrinsics API：删除 flip 参数。
4. Endo3R YAML 中未消费的 patch-artifact 字段：删除死配置。

## 复审证据与非阻断备注

- reviewer focused tests：19 passed；`git diff --check` 通过。
- 主进程全仓库：114 passed, 2 skipped；语法与 CLI help 通过。
- 复审关闭原 Major 和四项 Minor，verdict 为 APPROVE。
- 非阻断：R 目前检查 `det(R)≈1`，未来可额外检查 `R^T R≈I`；`torch.cuda.amp.*`
  产生 FutureWarning，但当前兼容性和数值路径不受影响。
