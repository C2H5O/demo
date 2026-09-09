# Baseline G：VDA window role KV sampling

实现状态：代码已接入，尚未运行模型或 benchmark。当前阶段仅作语法、配置和 diff 静态检查。
新增 CPU 合约测试供下一阶段运行，不把测试文件的存在当作测试已通过。

本工作目录位于现有 `codex/baseline-g-spark3r-proposed` 分支。当前分支内同时提供 E dense、
F fixed-stride 和 G role 三种推理配置；没有修改独立 F 分支，也没有提交或推送。

## 1. 阅读代码后确认的实际架构

这不是原版 Video Depth Anything 网络：当前项目是 **DA3-Small + VDA 风格窗口与融合**。
`inference/student_video.py` 是完整序列推理入口，`models/student/da3_small_student.py`
负责 DA3 输入归一化、backbone、depth-only DPT、相机头和密集输出。

已核对本机上游 `external/Depth-Anything-3` 源码，commit 为
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`，与 `scripts/setup_da3.sh` 的 pin 一致。
新 worktree 不复制外部源码、数据或权重；执行时通过现有环境/PYTHONPATH 提供上游包。

关键上游位置（相对于该上游库）：

| 文件 | 实际职责 |
|---|---|
| `src/depth_anything_3/model/dinov2/dinov2.py` | 构造 ViT-S/14，传递 backbone kwargs |
| `src/depth_anything_3/model/dinov2/vision_transformer.py` | token layout、参考视图重排、local/global 交替、位置编码、恢复输出顺序 |
| `src/depth_anything_3/model/dinov2/layers/block.py` | norm → attention → residual → MLP |
| `src/depth_anything_3/model/dinov2/layers/attention.py` | 一次线性层生成 Q/K/V；Q/K norm；RoPE；SDPA 或显式 matmul |
| `src/depth_anything_3/model/reference_view_selector.py` | 原生参考视图选择及 `[reference, remaining]` 重排 |

本地 `checkpoints/da3-small/config.json` 指定 `alt_start=4`、`qknorm_start=4`、
`rope_start=4`；实际 global block 为 **0-based `[5,7,9,11]`**。
普通 patch 之前有 CLS，随后在 `alt_start` 被原生 camera token 替换；当前 Small 没有 register token。
`models/attention_capture.py` 已有从 `alt_start-2` block 输出恢复参考视图 permutation 的 hook 约定。
另一个已有 Flash 工作目录的 `models/flash_attention.py` 使用 instance-local `TorchFunctionMode`
接管 SDPA；本实现沿用这种局部接入方式，不复制上游 attention forward。

原 F/G 的 KV 实现只有配置与拒绝运行的占位入口；没有已完成的 Spark3R KV 算法可直接调用。
本次 F 是 Spark3R 风格的 **frame temporal stride 对照**，并非完整 Spark3R 算法或官方实现。

## 2. 原窗口构造保持不变

项目没有名为 `INFER_LEN` 的常量，对应常量是：

```python
WINDOW = 32
OVERLAP = 10
BLEND = 8
STEP = 22
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
```

`start` 来自 `range(0, len(frames), 22)`。先构造
`ids[j] = min(start+j, len(frames)-1)`，后续窗口再用上一窗口 `KEYFRAMES`
对应的来源帧替换前 10 个 slot。这里 `ids` 是 **sequence position**，不是文件名帧 ID。

| 窗口 | 原始输入 slot 的 sequence position（无尾部 padding 示例） |
|---|---|
| 0 | `[0,1,…,31]` |
| 1 | `[0,12,24,…,31,32,…,53]` |
| 2 | `[0,34,46,…,53,54,…,75]` |

第一窗口 32 个角色均为 `new`。后续窗口的 slot `0:2` 为 `key`，`2:10` 为 `overlap`，
`10:32` 为 `new`。角色在构造/替换输入来源的同一位置生成，和对应的 padding 标记一起传递。
没有在 attention 中根据 slot 盲猜角色。

原有前 2 帧 disparity 对齐、8 帧 blending、滚动 anchor 更新、输出裁剪及提前结束逻辑原样保留。
因此某些短视频会按照既有步进构造一个主要为 padding 的后续窗口；本实现不改这个行为。

`WindowFrameMetadata` 分开保存窗口 ID、sequence positions、absolute dataset frame IDs、roles、
padding 和首窗标记。`sequence_frames()` 用原有 `extract_frame_id()` 从文件名提取真实 dataset ID。
直接传入普通 tensor 列表时，没有 dataset ID，audit 明确标记 `absolute_id_source=sequence_position`。

## 3. 选择规则与尾部 fallback

`inference/kv_sampling.py` 不接触 RGB、特征、depth、GT 或随机数。

`uniform_select(indices, count)` 在有序候选列表上保留端点，内部位置采用
`round_half_up(j*(n-1)/(count-1))`，用整数运算避免 Python 银行家舍入差异。
只取 1 帧时选偏左中点；候选不足时全部返回一次。

正常后续窗口 G 的选择：

| 角色 | 输入 slot | 选择 slot | 数量 |
|---|---|---|---|
| key | `[0,1]` | `[0,1]` | 2 |
| overlap | `[2,…,9]` | `[2,9]`，覆盖 overlap 两端 | 2 |
| new | `[10,…,31]` | `[10,17,24,31]`，每隔 7 个位置 | 4 |
| 合计 | 32 | `[0,1,2,9,10,17,24,31]` | 8 |

首窗 G 不伪造参考角色，连续 32 帧均匀取 `[0,4,9,13,18,22,27,31]`。
F 的正常窗口（包括首窗）固定 stride=4，取原窗口 slot `[0,4,8,12,16,20,24,28]`。
stride 不作用于 DA3 重排后的顺序。

尾部输入仍 pad 到 32，所有 pad Query 继续执行，最终输出仍由原融合逻辑裁剪。
KV 候选不包含 padding，也不重复选择同一个来源帧：重复来源优先保留 key，再 overlap，再 new。
先尽量满足各角色配额，然后在剩余有效候选的时间范围内均匀补齐未使用预算。
F 在其固定 stride 候选不足时，也从其余有效候选补齐。两种方法共用有效候选规则与总预算：

```text
F/G selected frame count = min(8, 非 padding 的唯一来源帧数量)
```

比如后续窗口只有 1 个真实 new frame：先取 2 key + 2 overlap + 1 new，再从剩余 overlap 补 3；
只有 7 个真实帧的首窗则保留 7 帧，而不是重复选 padding 凑 8。
所有 selected slot 唯一、合法；有更多未选 Query 并不影响其产生 dense depth。

## 4. 真正减少 KV 的位置

`inference/da3_kv_attention.py` 的 adapter 只临时包裹真实 global block 的 attention.forward。
原上游 forward 继续执行：

```text
full x → original qkv Linear → Q/K normalization → original RoPE
       → SDPA 入口：Q 不变；用同一组 indices gather K 和 V
       → 原 SDPA → 原 projection → 原 residual/MLP → 原 DPT
```

`frame_slots_to_token_indices()` 独立于 frame selection。它先把选中的原窗口 slot 映射到
DA3 内部 `[reference, remaining-in-original-order]` 的位置，再展开该 frame 完整 patch range；
同时加入 **所有输入帧** 的特殊 token 前缀，包括未选 KV frame 和 padded frame 的特殊 token。
K/V 使用同一个 `torch.gather(..., dim=2, index=indices)` 索引 tensor。

参考 permutation 的取得复用现有 attention capture hook 思路：在 `alt_start-2` block 输出上
重复调用**已有的确定性原生 selector**，并遵循原生 threshold 和 cam-token 条件。
这只是追踪 DA3 原本就会执行的重排，不改变 KV frame set，也不新增动态关键帧策略。
这次重复 selector 的开销会计入 model forward 时间；后续性能结论必须包含它。

没有构造 N×N mask，也没有先生成完整 attention score。
SDPA 执行后检查输出 token 维度仍等于 Q；每个窗口检查所有目标层恰好执行一次真实 SDPA。
启用 reduction 时使用上游现有 fused SDPA 分支；离开推理作用域时恢复原来的方法与设置。
所有修改是 model instance 局部的；没有修改上游源码或进程全局 SDPA。

448×560 输入、14px patch 的当前 DA3-Small：

```text
P = 32 × 40 = 1280
Q patch tokens = 32 × 1280 = 40960
KV patch tokens = 8 × 1280 = 10240
all special tokens = 32

actual Q: [B, 6, 40992, 64]
actual K/V after gather: [B, 6, 10272, 64]
per-head attention: 40992 × 10272
```

以上是源码推导的正常窗口尺寸，**本次没有运行模型取得 shape 实测**。
运行时 adapter 在 kernel 调用边界记录并检查真实尺寸。
patch retention 为 25%；包含所有特殊 token 的实际 token retention 约为 25.06%。
local attention、相机解码器、DPT、全帧输出都不剪裁。

## 5. Position 与模型状态

上游先给完整 token 加 learned spatial position；local RoPE 使用原空间坐标。
当前 global RoPE 的 `pos_nodiff` 对 patch 使用常量位置、special 使用零位置，
不是把 absolute frame ID 编码为 temporal position。
本实现完全沿用该设计，在原 Q/K RoPE **之后** gather K，不重新编号位置、不重新施加 RoPE。

当前模型每帧 1 个 CLS/camera token，无 register token。
索引 helper 按 `1 + num_register_tokens` 保留完整特殊前缀，不硬编码为“整个 selected frame 才留 special”。
实际输入尺寸不符合已审计 layout 时会报错，不静默错选 token。

没有新参数、checkpoint/state_dict 变更、训练步骤或 loss 改动。
G/F 配置明确关闭 `dataset.highlight.enabled`；评估和可视化原有 RGB-only 路径也关闭 mask。
历史 E 权重的训练配置字段作为 provenance 保留，但推理不实例化 loss/Teacher 或训练 attention capture。
训练入口已有 `training_required=false` 拒绝保护，保持不变。
本次没有执行高光、纹理可靠性、ring anchor、光流、聚类或 token-level priority。

## 6. 配置与记录

共享默认值在 `configs/baselines/_kv_sampling.yaml`，顶层配置为：

```yaml
kv_sampling:
  enabled: false
  method: vda_role
  retention_ratio: 0.25
  vda_role:
    key_frames: 2
    overlap_frames: 2
    new_frames: 4
  first_window:
    method: uniform
    num_frames: 8
  spark3r_fixed_stride:
    temporal_stride: 4
  debug: false
  debug_max_windows: 2
  profile_attention: false
```

| 模式 | 配置 | 行为 |
|---|---|---|
| E dense | `configs/inference/E_dense.yaml` | enabled=false，原 dense 输入/输出，默认不安装 attention wrapper/hook |
| F stride | `configs/baselines/F.yaml` | enabled=true，method=spark3r_fixed_stride |
| G role | `configs/baselines/G.yaml` 或 `configs/baseline.yaml` | enabled=true，method=vda_role |
| G audit | `configs/inference/G_debug.yaml` | G + 前两个窗口日志 + global SDPA profiling |

三者使用同一 E checkpoint 和推理设置。原 `configs/baselines/E.yaml` 的训练配置不变。
修改 retention_ratio 时，role 配额总和、首窗 budget、F 的 stride 所得 budget 必须匹配；
配置检查拒绝不公平的 F/G 完整窗口预算。目标层目前从真实 global 层列表得到，统一使用一个预算。
selection、token mapping、kernel execution 分离，后续可在 adapter 中扩展 layer budget，但本版没有 schedule。

每条 sequence 的结果记录：同步 model forward 秒数、sequence pipeline 秒数、两种 FPS、
峰值 allocated/reserved CUDA bytes、按层聚合的实际 Q/KV shape，以及前两个窗口的 provenance/selection 示例。
峰值计数在每个序列开始 reset；allocated 包含已经加载的模型和其他仍存活的 CUDA 分配，
reserved 也可能包含 allocator 缓存，不应误称 attention-only 显存。
FPS 的分子是唯一输出帧数量，不是重复窗口帧数量。

`profile_attention=true` 时用 CUDA events（CPU 用墙钟）累计 global **SDPA kernel 调用**时间，
明确不包含 QKV projection、norm、RoPE、selector 和 gather；这些仍包含在 model forward 与 pipeline 计时中。
默认关闭 profiler，因此 `global_sdpa_seconds=null`，不伪造 attention 耗时。
pipeline 包含 RGB decode/transfer、选择、模型、融合、output callback、audit 和 adapter setup/restore，
不包括模型加载、后续 GT/TAE scoring 或最终结果 JSON 序列化。首次运行未排除 warm-up。

enabled=false 且不开 debug/profiling 时，dense token 数来自 encoder layout，并标注
`encoder_layout_dense`；打开 debug/profiling 可记录 `observed_sdpa_inputs`。
F/G 默认就记录实际 SDPA 输入尺寸。三种模式做正式比较时使用相同 profiling/debug 设置。

## 7. 后续命令（本次未执行）

在本工作目录根运行。训练配置继承了服务器的路径，完整评估前应核对
student safetensors/config.json、raw SCARED RGB/GT/camera root 和 E checkpoint。
不同 worktree 不会自动获得另一个目录的 outputs；`--checkpoint` 必须指向真实 E 权重。
不要为这次实验重新训练或另存修改权重。

本机只跑 CPU 合约测试、不加载 checkpoint 的例子：

```powershell
Set-Location D:\Projects\vggtoda3\.worktrees\baseline_G
$env:PYTHONPATH = 'D:\Projects\vggtoda3\external\Depth-Anything-3\src;' + $env:PYTHONPATH
python -m pytest tests/test_vda_role_kv.py tests/test_student_video.py -q
```

需要使用已安装 PyTorch/pytest 和 DA3 模型依赖的 Python 环境。
若 DA3 package 不可导入，tiny-upstream 测试会 skip；这种结果不能视为 attention 集成通过。
测试使用小型随机初始化的真实上游 Transformer，检查真实参考重排和矩形 SDPA，不加载任何权重文件。

先做两窗口人工审计（下面的 checkpoint 路径需替换）：

```powershell
$eCheckpoint = '替换为现有 baseline_E/last.pt 的绝对路径'
python evaluate_crossclip_projection.py --config configs/inference/G_debug.yaml --checkpoint "$eCheckpoint" --limit-windows 2
```

Linux 服务器等价命令：

```bash
export PYTHONPATH="/public/home/2024141520249/Documents/Projects/vggtoda3/external/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
E_CKPT='/replace/with/existing/baseline_E/last.pt'
python evaluate_crossclip_projection.py --config configs/inference/G_debug.yaml --checkpoint "$E_CKPT" --limit-windows 2
```

日志应显示首窗与后续窗口不同的 selected slots、全部 roles/absolute IDs、角色计数和实际 Q/KV 长度。
对 32-frame 正常后续窗口，G 的 role counts 必须是 2/2/4；尾部不足会按上文 fallback。
`--limit-windows` 是跨序列的全局窗口预算，调试结果不能作为完整测试集指标。

确认路径和小测试后，三种完整评估命令（同一 shell 内定义好 `E_CKPT`）：

```bash
python evaluate_crossclip_projection.py --config configs/inference/E_dense.yaml --checkpoint "$E_CKPT"
python evaluate_crossclip_projection.py --config configs/baselines/F.yaml --checkpoint "$E_CKPT"
python evaluate_crossclip_projection.py --config configs/baselines/G.yaml --checkpoint "$E_CKPT"
```

对应结果分别为 `outputs/baseline_E_dense/evaluation_test.json`、
`outputs/baseline_F/evaluation_test.json`、`outputs/baseline_G/evaluation_test.json`。
可视化入口使用同一个 infer_student_video，选择 G/F 配置即可启用相同策略。
真实 speed、显存、AbsRel、δ1、TAE 需要后续运行验证；这里不宣称加速或精度保持。
