from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_crossclip_projection import evaluate_vggt_omega_online


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline-T VGGT-Omega online with the formal VDA sequence protocol"
    )
    parser.add_argument("--config", default="configs/baselines/T.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--protocol", choices=("vda",), default="vda")
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Debug window budget. Omit for the complete SCARED test evaluation.",
    )
    args = parser.parse_args()
    evaluate_vggt_omega_online(
        Path(args.config),
        checkpoint=args.checkpoint,
        output=args.output,
        limit_clips=args.limit_windows,
        protocol=args.protocol,
    )


if __name__ == "__main__":
    main()
