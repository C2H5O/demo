# VGGT-to-MASt3R V1

## 当前实验协议

- Branch：`vggtomast3r`
- Base：`origin/feature/distill3r-student@b311d13`
- Student 输入：ordered `(I_t,I_{t+2})`，`pair_stride=2`，`pair_step=1`
- Teacher：冻结的 pretrained VGGT-Omega base；不注入、不加载 LoRA
- Teacher 推理粒度：一次只输入一个 RGB frame
- Teacher cache：每个源 frame 一个 FP32 NPZ；2 帧和 8 帧样本在读取时组合
- 分辨率：`448×560`

这一协议由用户在 2026-08-22 明确替换了原先的 LoRA two-frame teacher
协议。它不再声称只改变 student architecture；teacher conditioning 和 teacher
variant 都是实验变量，正式比较必须相应重跑 baseline。

## Teacher frame cache

入口：

```bash
python generate_teacher_frame_cache.py \
  --config configs/vggtomast3r_v1.yaml --split train
python generate_teacher_frame_cache.py \
  --config configs/vggtomast3r_v1.yaml --split test
```

兼容入口 `generate_teacher_pair_cache.py` 会调用相同的 frame generator，
不会再运行 pair-conditioned teacher。该兼容入口不接受 `--limit`，因为 pair 数量
与去重后的 frame 数量不同；小规模检查请直接使用 frame 入口的 `--limit`（单位为帧）。

Generator 强制：

- `teacher.variant=base`
- `teacher.frozen=true`
- `teacher.lora_checkpoint=null`
- `inject_lora=False`、`load_lora=False`
- 输入 shape `[1,3,448,560]`，模型内部 sequence length 为 1
- `cache_dtype=float32`

默认目录：

```text
data/teacher_cache_vggtomega_base_frame_448x560/
  train|test/
    dataset_XX/keyframe_X/sequence_id/frame_FRAMEINDEX_FRAMEID.npz
```

Schema `vggtomega-base-frame-v1`：

```text
dataset_id, keyframe_id, sequence_id
frame_id, frame_index, frame_name
image_shape, teacher_variant, inference_frame_count
depth, xyz_local, confidence, valid_mask
intrinsics, extrinsics
coordinate_convention, cache_format_version
base_checkpoint, metadata_json
```

`depth/xyz_local/confidence` 保存为 FP32。旧 pair cache、旧 8-frame cache、
LoRA cache 或非单帧推理 cache 均不能通过新 reader 的 schema 检查。

## 2 帧与 8 帧组合

`datasets.teacher_frame_cache.compose_teacher_frame_caches` 接收有序的 2 个或
8 个 frame metadata，并堆叠：

```python
{
    "depth": ...,       # [T,H,W]
    "xyz_local": ...,   # [T,H,W,3]
    "confidence": ...,  # [T,H,W]
    "valid_mask": ...,  # [T,H,W]
    "intrinsics": ...,  # [T,3,3]
    "extrinsics": ...,  # [T,3,4]
}
```

这里的每一帧都在各自的 camera-local 坐标系。单帧独立推理不产生跨帧共享的
world gauge，因此禁止从这些 cache 构造 `B-in-A`、global fused cloud 或跨帧 pose
监督。8 帧组合只是复用八个独立 frame cache，不会重新运行 teacher。

## Student 与 loss

Student 继续使用 pinned official DUNE-S/14 + MASt3R binocular decoder/head。
DUNE frozen/eval，decoder 与 downstream heads trainable。

为匹配两个独立 local teacher target，训练把 `(A,B)` 和 `(B,A)` 合并成一个
`2B` MASt3R batch，并分别取两半的 `pred1["pts3d"]`：

```python
{
    "pts3d_ref": ...,         # A in camera A
    "pts3d_other_local": ..., # B in camera B
}
```

不再暴露或监督 `pts3d_other_in_ref`。两帧 local maps 共同做
confidence-weighted、joint-scale-normalized Charbonnier point loss；SCARED metric
depth 仍只监督 `pts3d_ref[...,2]`：

```text
L_total = 1.0 * L_teacher_frame_local_point
        + 0.1 * L_SCARED_reference_depth
```

旧 pair-cache 协议训练出的 checkpoint 缺少 `teacher.cache_protocol=frame_local_v1`，
trainer 会拒绝 resume，evaluation 与 visualization 也会拒绝加载，避免跨协议误用。

因为双向 local 解码把有效 decoder batch 扩成 `2B`，默认 dataloader batch 从
4 调整为 2（有效方向 batch 仍为 4）；如显存允许可在 YAML 中提高。

## 可视化

```bash
python visualize_teacher_pair_cache.py \
  --config configs/vggtomast3r_v1.yaml --split train --pair-index 0
```

脚本从两个 frame cache 组合结果，同时输出固定色阶和 p1–p99 自适应色阶、原始
FP32 NPY、confidence 与两个独立 camera-local PLY。它不会把两帧点云融合，并在
metadata 中明确记录坐标警告。

## 服务器流程

```bash
git clone --recursive --branch vggtomast3r https://github.com/C2H5O/demo.git vggtomast3r
cd vggtomast3r
git submodule update --init --recursive
pip install -r requirements.txt
pip install -r requirements-vggtomast3r.txt
bash scripts/download_vggtomast3r_checkpoints.sh
python scripts/verify_vggtomast3r_environment.py

# Required assets:
# data/SCARED
# checkpoints/vggt_omega/vggt_omega_1b_512.pt

python generate_teacher_frame_cache.py --config configs/vggtomast3r_v1.yaml --split train
python generate_teacher_frame_cache.py --config configs/vggtomast3r_v1.yaml --split test
python visualize_teacher_pair_cache.py --config configs/vggtomast3r_v1.yaml --split train --pair-index 0

python train_vggtomast3r.py --config configs/vggtomast3r_v1.yaml --dry-run
python train_vggtomast3r.py --config configs/vggtomast3r_v1.yaml
python evaluate_vggtomast3r.py --config configs/vggtomast3r_v1.yaml
python evaluate_vggtomast3r.py --config configs/vggtomast3r_v1.yaml --protocol endo3r
```

## 验证边界

本地 unit tests 不依赖真实 SCARED 或大 checkpoint。真实 base checkpoint 的
单帧 448×560 forward、少量 cache smoke、训练 dry-run 与完整 cache/训练必须在
GPU 服务器验证。本实现保留旧 Distill3R 8-frame dataset/cache/trainer，不覆盖旧
cache root。
