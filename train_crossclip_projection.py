from __future__ import annotations

import argparse
from pathlib import Path

from trainers.crossclip_projection_trainer import train_crossclip_projection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the 16-frame DUNE-to-Fast3R-head cross-clip experiment"
    )
    parser.add_argument("--config", default="configs/crossclip_teacher_projection.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train_crossclip_projection(
        Path(args.config), args.dry_run, args.resume, args.max_steps
    )


if __name__ == "__main__":
    main()
