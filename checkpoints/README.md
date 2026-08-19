# Local checkpoints

Place pretrained and generated model weights under this project only:

```text
checkpoints/
  vggt_omega/
    vggt_omega_1b_512.pt
  teacher_lora/
    last.pt
  dune/
    dune_vitsmall14_336.pth
```

The default YAML files already use these paths. Training outputs remain under
`outputs/`; copy or configure a selected checkpoint here when a stable input
path is needed. All weight files are intentionally excluded from Git.
