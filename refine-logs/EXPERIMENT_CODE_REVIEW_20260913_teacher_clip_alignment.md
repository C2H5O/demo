# Baseline-I Teacher Cross-Clip Scale Alignment 代码审查

审查者：独立 Codex 子代理，GPT-5.6-Sol（xhigh）

结论：**最终静态复核未发现阻断项。**

## 范围

本次只做静态代码审查。按照用户执行限制，没有运行 pytest、alignment
脚本、dry-run、训练、GPU 任务或数据生成。

## 首轮发现与修复

1. audit_teacher_clip_alignment() 最初未核对顶层 source_cache_root 和逐 clip
   的 cache_relative_path。现已把二者列为 metadata 必需字段，并在 audit
   中与指定 raw cache root 和重建结果严格比较。
2. Baseline-E 继承测试最初整体替换 I 的 training 字典，无法阻止未来误改
   LR、epochs、gradient accumulation 或 AMP。现只允许实验标识、新 alignment
   section、I 的输出路径和 resume 字段不同，其余 resolved config 必须与 E 相同。
3. Dataset 最初直接导入 cache.teacher_clip_alignment 仅用于类型标注，可能与
   两个 package 的 eager __init__ 形成导入环。现改为无运行时 cache 依赖的
   类型标注，alignment 对象仍由 trainer 注入。

## 最终复核

- losses/direct_teacher_distillation_loss.py 与 Baseline-E 提交 17fd9aaa
  对应文件无差异；旧 Student affine-disparity 函数、路由和诊断已删除。
- 相邻 clip 通过显式 absolute_frame_ids 取 8 帧交集；每帧计算
  median(D_prev / D_cur)，再取 frame median 的中位数。
- 累积方向为 g_cur = g_prev * r_cur_to_prev，anchor clip scale 固定为 1。
- metadata 和 audit 覆盖 clip chain、source identity、valid pixel 数、scale
  finite/positive、pair/sequence/macro AbsRel 统计；极端 scale 只告警，不 clamp。
- 开关默认关闭。Baseline-I 启用后只缩放 Teacher depth、可选 xyz 和 W2C
  translation；rotation、intrinsics、confidence、valid mask 与 online attention 不变。
- 单元测试源码覆盖已知尺度、anchor、三 clip 累积、loader geometry、旧 affine
  路径缺失、E 配置继承、metadata audit 和 resume 隔离。

## 验证边界

只完成代码与 diff 静态审查；所有可执行验证等待用户在服务器运行。
