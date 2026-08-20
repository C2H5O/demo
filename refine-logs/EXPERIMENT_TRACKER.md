# VGGT-to-MASt3R V1 实验跟踪

| Run | 状态 | 备注 |
|---|---|---|
| R001 | DONE | 全仓库 78 passed、1 skipped；focused V1 18 passed |
| R002 | DONE | pinned MASt3R/DUNE 在现有 vggtodistill3r env import 通过 |
| R003 | BLOCKED-BY-ASSETS | 本地无 joint/DUNE checkpoint；服务器运行 full verify |
| R004 | BLOCKED-BY-ASSETS | 按边界不执行完整 cache；服务器先做 sample pair smoke |
| R005 | BLOCKED-BY-ASSETS | 需要真实 SCARED、pair cache 与 checkpoint |
| R006 | NOT STARTED | 用户明确禁止本地长期训练 |
| R007 | IMPLEMENTED | synthetic patch-boundary test 通过；真实指标待 R006 |
| R008 | IMPLEMENTED | 固定范围 panel/PLY 脚本完成；真实 artifact 待 checkpoint/cache |
