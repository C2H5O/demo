from __future__ import annotations

import argparse
from pathlib import Path

from trainers.vggtomast3r_trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vggtomast3r_v1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train(Path(args.config), args.dry_run, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
