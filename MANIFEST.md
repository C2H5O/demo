# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-08-20 18:31 | /experiment-bridge | .gitignore | implementation | ignore ARIS metadata and preserve external source gitlinks |
| 2026-08-20 18:31 | /experiment-bridge | .gitmodules | implementation | pin official MASt3R and DUNE HTTPS submodules |
| 2026-08-20 18:31 | /experiment-bridge | external/MASt3R | implementation | official MASt3R gitlink |
| 2026-08-20 18:31 | /experiment-bridge | external/DUNE | implementation | official DUNE gitlink |
| 2026-08-20 18:31 | /experiment-bridge | README.md | implementation | V1 entry points and coordinate warning |
| 2026-08-20 18:31 | /experiment-bridge | docs/vggtomast3r_v1.md | implementation | complete experiment workflow |
| 2026-08-20 18:31 | /experiment-bridge | configs/vggtomast3r_v1.yaml | implementation | V1 configuration |
| 2026-08-20 18:31 | /experiment-bridge | utils/geometry.py | implementation | camera-from-world transforms |
| 2026-08-20 18:31 | /experiment-bridge | datasets/scared_pair_dataset.py | implementation | strict pair dataset and cache reader |
| 2026-08-20 18:31 | /experiment-bridge | cache/generate_teacher_pair_cache.py | implementation | VGGT-Omega pair cache exporter |
| 2026-08-20 18:31 | /experiment-bridge | generate_teacher_pair_cache.py | implementation | pair cache CLI |
| 2026-08-20 18:31 | /experiment-bridge | models/student/official_mast3r.py | implementation | pinned official source loader |
| 2026-08-20 18:31 | /experiment-bridge | models/student/dune_mast3r_adapter.py | implementation | DUNE-MASt3R student adapter |
| 2026-08-20 18:31 | /experiment-bridge | models/student/__init__.py | implementation | student exports |
| 2026-08-20 18:31 | /experiment-bridge | losses/vggtomast3r_loss.py | implementation | minimal V1 objective |
| 2026-08-20 18:31 | /experiment-bridge | losses/__init__.py | implementation | loss exports |
| 2026-08-20 18:31 | /experiment-bridge | trainers/vggtomast3r_trainer.py | implementation | V1 trainer |
| 2026-08-20 18:31 | /experiment-bridge | train_vggtomast3r.py | implementation | training CLI |
| 2026-08-20 18:31 | /experiment-bridge | evaluation/evaluate_vggtomast3r.py | implementation | streaming Endo3R evaluator |
| 2026-08-20 18:31 | /experiment-bridge | evaluation/vggtomast3r_metrics.py | implementation | patch artifact metric |
| 2026-08-20 18:31 | /experiment-bridge | evaluate_vggtomast3r.py | implementation | evaluation CLI |
| 2026-08-20 18:31 | /experiment-bridge | visualization/vggtomast3r_pair.py | implementation | fixed-range pair visualization |
| 2026-08-20 18:31 | /experiment-bridge | visualize_vggtomast3r.py | implementation | visualization CLI |
| 2026-08-20 18:31 | /experiment-bridge | requirements-vggtomast3r.txt | implementation | incremental dependencies |
| 2026-08-20 18:31 | /experiment-bridge | scripts/download_vggtomast3r_checkpoints.sh | implementation | resumable official checkpoint downloads |
| 2026-08-20 18:31 | /experiment-bridge | scripts/verify_vggtomast3r_environment.py | implementation | source/checkpoint/forward verification |
| 2026-08-20 18:31 | /experiment-bridge | tests/test_vggtomast3r_geometry.py | implementation | coordinate tests |
| 2026-08-20 18:31 | /experiment-bridge | tests/test_vggtomast3r_pair_cache.py | implementation | pair/cache tests |
| 2026-08-20 18:31 | /experiment-bridge | tests/test_vggtomast3r_model.py | implementation | model/freeze/depth tests |
| 2026-08-20 18:31 | /experiment-bridge | tests/test_vggtomast3r_loss_metrics.py | implementation | loss/metric tests |
| 2026-08-20 18:31 | /experiment-bridge | idea-stage/docs/research_contract.md | implementation | active claim contract |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN_20260820_181212.md | implementation | versioned experiment plan |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | latest experiment plan |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260820_181212.md | implementation | initial tracker |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260820_183053.md | implementation | updated tracker |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest tracker |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW_20260820_182928.md | implementation | versioned fresh-agent review |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW.md | implementation | latest fresh-agent review |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS_20260820_183053.md | implementation | versioned initial results |
| 2026-08-20 18:31 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS.md | implementation | latest initial results |
| 2026-08-22 14:39 | /experiment-bridge | cache/generate_teacher_frame_cache.py | implementation | frozen base teacher single-frame FP32 cache generator |
| 2026-08-22 14:39 | /experiment-bridge | generate_teacher_frame_cache.py | implementation | single-frame cache CLI |
| 2026-08-22 14:39 | /experiment-bridge | cache/generate_teacher_pair_cache.py | implementation | compatibility alias with unambiguous frame semantics |
| 2026-08-22 14:39 | /experiment-bridge | generate_teacher_pair_cache.py | implementation | compatibility CLI without ambiguous pair limit |
| 2026-08-22 14:39 | /experiment-bridge | datasets/teacher_frame_cache.py | implementation | versioned frame schema, validation, paths, and 2/8 composition |
| 2026-08-22 14:39 | /experiment-bridge | datasets/scared_pair_dataset.py | implementation | two independent camera-local target reader |
| 2026-08-22 14:39 | /experiment-bridge | datasets/__init__.py | implementation | frame cache exports |
| 2026-08-22 14:39 | /experiment-bridge | models/student/dune_mast3r_adapter.py | implementation | one-call bidirectional local decoding |
| 2026-08-22 14:39 | /experiment-bridge | losses/vggtomast3r_loss.py | implementation | A-local and B-local teacher supervision |
| 2026-08-22 14:39 | /experiment-bridge | trainers/vggtomast3r_trainer.py | implementation | frame protocol training and resume guard |
| 2026-08-22 14:39 | /experiment-bridge | utils/checkpoint.py | implementation | shared frame-local checkpoint protocol guard |
| 2026-08-22 14:39 | /experiment-bridge | evaluation/evaluate_vggtomast3r.py | implementation | Endo3R local-depth consumer and protocol guard |
| 2026-08-22 14:39 | /experiment-bridge | evaluation/evaluate_vggtomast3r_vda.py | implementation | default VDA local-depth consumer and protocol guard |
| 2026-08-22 14:39 | /experiment-bridge | visualization/vggtomast3r_teacher_frame_cache.py | implementation | composed frame-cache depth/confidence/local-cloud export |
| 2026-08-22 14:39 | /experiment-bridge | visualize_teacher_pair_cache.py | implementation | composed frame-cache visualization CLI |
| 2026-08-22 14:39 | /experiment-bridge | visualization/vggtomast3r_pair.py | implementation | student A-local/B-local visualization |
| 2026-08-22 14:39 | /experiment-bridge | visualization/vggtomast3r_teacher_cache.py | implementation | removed obsolete pair/clip cache visualizer |
| 2026-08-22 14:39 | /experiment-bridge | configs/vggtomast3r_v1.yaml | implementation | frozen base frame protocol and effective batch settings |
| 2026-08-22 14:39 | /experiment-bridge | README.md | implementation | frame cache quick start and coordinate contract |
| 2026-08-22 14:39 | /experiment-bridge | docs/vggtomast3r_v1.md | implementation | complete frozen-base frame cache workflow |
| 2026-08-22 14:39 | /experiment-bridge | idea-stage/docs/research_contract.md | implementation | revised claim boundary for independent local frames |
| 2026-08-22 14:39 | /experiment-bridge | tests/test_vggtomast3r_pair_cache.py | implementation | frame validation, collision, and 2/8 composition tests |
| 2026-08-22 14:39 | /experiment-bridge | tests/test_vggtomast3r_model.py | implementation | bidirectional local output contract tests |
| 2026-08-22 14:39 | /experiment-bridge | tests/test_vggtomast3r_loss_metrics.py | implementation | local target loss tests |
| 2026-08-22 14:39 | /experiment-bridge | tests/test_vggtomast3r_evaluation.py | implementation | local depth and checkpoint protocol tests |
| 2026-08-22 14:39 | /experiment-bridge | tests/test_vggtomast3r_teacher_cache_visualization.py | implementation | coordinate-safe composed cache visualization test |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN_20260822_143910.md | implementation | versioned frozen-base single-frame experiment plan |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_PLAN.md | implementation | latest experiment plan |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER_20260822_143910.md | implementation | versioned implementation tracker |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest implementation tracker |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW_20260822_143910.md | implementation | versioned fresh-agent review and fix audit |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_CODE_REVIEW.md | implementation | latest code review |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS_20260822_143910.md | implementation | versioned local verification results |
| 2026-08-22 14:39 | /experiment-bridge | refine-logs/EXPERIMENT_RESULTS.md | implementation | latest local verification results |
