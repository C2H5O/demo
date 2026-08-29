from __future__ import annotations

import argparse
from pathlib import Path

from trainers.crossclip_projection_trainer import train_crossclip_projection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the 16-frame VGGT-Omega-to-DA3-Small experiment"
    )
    parser.add_argument("--config", default="configs/vggtoda3.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train_crossclip_projection(
        Path(args.config), args.dry_run, args.resume, args.max_steps
    )


if __name__ == "__main__":
    main()
