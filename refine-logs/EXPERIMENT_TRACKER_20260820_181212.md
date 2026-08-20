# VGGT-to-MASt3R V1 实验跟踪

| Run | 状态 | 备注 |
|---|---|---|
| R001 | DONE | 17 个 focused synthetic/mock tests 通过 |
| R002 | IN PROGRESS | submodule 已固定；环境 import smoke 待脚本修正后复跑 |
| R003 | BLOCKED-BY-ASSETS | 本地没有两个官方 checkpoint；服务器运行 |
| R004 | BLOCKED-BY-ASSETS | 不执行完整 cache；服务器最多先做 sample pair smoke |
| R005 | BLOCKED-BY-ASSETS | 需要真实 SCARED、pair cache 与 checkpoint |
| R006 | NOT STARTED | 用户明确禁止本地长期训练 |
| R007 | IMPLEMENTED | synthetic patch-boundary test 通过；真实结果待 R006 |
| R008 | IMPLEMENTED | 脚本已实现；真实 artifact 待 checkpoint/cache |
