# VGGT-to-MASt3R V1 实验跟踪

| Run | 状态 | 备注 |
|---|---|---|
| R001 | DONE | 全仓库 95 passed、2 skipped；frame-cache/V1 focused 31 passed、1 skipped |
| R002 | DONE | pinned MASt3R/DUNE 在现有环境 import 通过 |
| R003 | BLOCKED-BY-ASSETS | 本地无完整 checkpoint/CUDA 资产；服务器运行真实 448x560 forward |
| R004 | IMPLEMENTED-PENDING-SERVER | 冻结 base 单帧 generator、FP32 schema、2/8 composition 与防碰撞路径已实现；真实 cache smoke 待服务器 |
| R005 | BLOCKED-BY-ASSETS | 需要真实 SCARED、完整 frame cache 与 student checkpoint 运行 dry-run |
| R006 | NOT STARTED | 用户未授权本地长期训练；完整 cache/训练/evaluation 待服务器 |
| R007 | IMPLEMENTED | patch-boundary metric synthetic test 通过；真实指标待 R006 |
| R008 | IMPLEMENTED-PENDING-ARTIFACT | fixed/adaptive depth、NPY、confidence、分离 local PLY 已实现；真实 artifact 待 cache |
| R009 | SUPERSEDED | 旧 8-frame/LoRA pair-cache 审查结论被 2026-08-22 用户指定的 frozen-base single-frame 协议替换 |
| R010 | DONE | fresh-agent GPT-5.6-Sol xhigh 复审：首轮 1 blocking + 4 non-blocking 均修复；最终无 blocking/non-blocking |
