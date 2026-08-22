# VGGT-to-MASt3R V1 实验代码审查

**日期**：2026-08-22
**审查方式**：fresh-agent GPT-5.6-Sol，xhigh；修复后对当前工作树复审。

## 最终结论

**BLOCKING：无。NON-BLOCKING：无。** 实现可进入真实 checkpoint/data/CUDA smoke；
本地未执行真实 teacher forward，不把静态与 mock 测试冒充实验结果。

## 首轮 blocking finding 与修复

1. 兼容 `generate_teacher_pair_cache.py --limit N` 曾把 pair 上限误当作 frame 上限，
   `--limit 1` 不能生成首个 `(t,t+2)` pair 需要的两个 cache。
   - 修复：compatibility CLI 移除 `--limit`；Python compatibility API 对非空 limit
     明确报错，并引导使用 frame generator（其 limit 明确以 frame 计数）。

## 首轮 non-blocking findings 与修复

1. generator 在切片前应验证完整 adapted output shape。
   - 修复：严格验证 depth/xyz/conf/mask/intrinsics/extrinsics 的 B=1、S=1 与空间形状。
2. reader schema 校验不足。
   - 修复：强制 point/depth/confidence/matrices 为 FP32、mask 为 bool、矩阵 shape
     为 3x3/3x4，并检查所有浮点数组 finite。
3. 自定义 manifest 可能因 dataset/keyframe/frame_id 相同发生 cache 路径碰撞。
   - 修复：路径加入 sanitized `sequence_id` 与 `frame_index`，文件名同时保留 frame ID。
4. evaluation/visualization 可能误读旧 pair-cache 协议 checkpoint。
   - 修复：集中实现 `require_student_cache_protocol`；trainer resume、VDA、Endo3R 和
     student pair visualization 都要求 `frame_local_v1`。

## 复审证据

- fresh-agent 复审：五项修复全部确认，0 blocking，0 non-blocking。
- reviewer focused tests：27 passed，1 skipped。
- 主进程全仓库：95 passed，2 skipped。
- 主进程 focused：31 passed，1 skipped。
- `compileall`、三个 CLI `--help`、`git diff --check`：通过。

## 核心语义确认

- teacher 是完全 frozen、无 LoRA 的 pretrained VGGT-Omega base。
- teacher 每次仅接收一个 source RGB frame；每帧一个版本化 FP32 NPZ。
- composition API 只接受严格的 2 或 8 帧有序 metadata，不重新运行 teacher。
- 各帧 output 保持独立 camera-local 坐标，不伪造共享 reference/world gauge。
- student 双向 local 解码输出 A-local/B-local；loss、VDA/Endo3R 与 visualization 已迁移。
- 旧 pair/clip cache 和旧协议 checkpoint fail closed。
