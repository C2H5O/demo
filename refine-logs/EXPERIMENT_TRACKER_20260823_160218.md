# Cross-clip teacher projection 实验跟踪

**日期**：2026-08-23
**分支**：`feature/crossclip-teacher-projection`

| Run | 状态 | 证据/备注 |
|---|---|---|
| R001 | DONE | 全仓库 `114 passed, 2 skipped`；focused `19 passed`；py_compile、五个 CLI help、diff check 通过 |
| R002 | DONE | cache 对 valid fraction、positive depth、XYZ-Z、confidence、K、R、metadata stats 严格校验；退化 K 与 zero-confidence fallback 有测试 |
| R003 | PENDING-SERVER | 用户禁止本地大规模 teacher inference；需真实 VGGT-Omega/SCARED/CUDA |
| R004 | IMPLEMENTED-PENDING-DATA | audit/alignment 已实现；尚无真实 cache scale ratio，不报告伪统计 |
| R005 | PENDING-SERVER | trainer dry-run 入口已实现；真实 cache/checkpoint/CUDA 未运行 |
| R006 | NOT STARTED | VDA evaluator 已实现并使用 SCARED GT；无本协议 checkpoint/指标 |
| R007 | NOT STARTED | Endo3R evaluator 已保留并使用 SCARED GT；无本协议 checkpoint/指标 |
| R008 | IMPLEMENTED-PENDING-ARTIFACT | teacher/student fixed/adaptive depth 与逐帧 local PLY 已实现，真实输出待服务器 |
| R009 | DONE | fresh-agent GPT-5.6-Sol xhigh：首轮 1 Major + 4 Minor；修复后复审 APPROVE，无 blocking |

## 当前 Git 提交

- `d8ff29d feat: use DUNE intermediate features with Fast3R head`
- `8da15a3 feat: add stride-one cross-clip teacher sampling`
- `0d3b32d feat: add teacher-coordinate projection losses`
- 文档/config/eval/visualization/integrity 修复：等待最终本地提交。
