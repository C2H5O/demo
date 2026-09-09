# Baseline G

Selected configuration: `configs/baselines/G.yaml` (also `configs/baseline.yaml`).

See [experiment plan](docs/BASELINE_EXPERIMENTS_4090.md).

Training required: False. Implementation complete; runtime validation pending.

G uses VDA window-role frame KV sampling: all queries, 2 key + 2 overlap + 4
new patch-KV frames, and all special tokens. It does not use highlight, texture,
ring-anchor or token-content selection. The E checkpoint and training code are
unchanged. F is an equal-budget fixed-stride control, not a full Spark3R port.

See [implementation, shape audit and test commands](docs/VDA_ROLE_KV_SAMPLING.md).
