from __future__ import annotations

import argparse
from pathlib import Path

from trainers.teacher_lora_trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VGGT-Omega MLP LoRA training")
    parser.add_argument("--config", default="configs/teacher_lora_finetune.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(Path(args.config), args.dry_run, args.max_steps, args.resume)


if __name__ == "__main__":
    main()
