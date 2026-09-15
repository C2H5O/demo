from __future__ import annotations

import argparse
from pathlib import Path

from endodaveval.pipeline import STAGES, run_pipeline


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Official EndoDAV SCARED evaluator")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--stage", choices=STAGES, default="all")
    value.add_argument("--limit-sequences", type=int)
    return value


def main() -> None:
    args = parser().parse_args()
    result = run_pipeline(args.config, args.stage, args.limit_sequences)
    print("completed {} for {} sequence(s)".format(args.stage, result["sequence_count"]))
