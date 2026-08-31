from __future__ import annotations

import argparse
from pathlib import Path

from trainers.direct_teacher_distillation_trainer import (
    train_direct_teacher_distillation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train same-clip VGGT-Omega to DA3-Small direct distillation"
    )
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train_direct_teacher_distillation(
        Path(args.config), args.dry_run, args.resume, args.max_steps
    )


if __name__ == "__main__":
    main()
