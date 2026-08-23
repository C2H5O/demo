from __future__ import annotations

import argparse
from pathlib import Path

from cache.align_crossclip_teacher_cache import align_crossclip_teacher_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit adjacent 16-frame teacher clips and write offline-aligned caches"
    )
    parser.add_argument("--config", default="configs/crossclip_teacher_projection.yaml")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--aligned-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = align_crossclip_teacher_cache(
        Path(args.config),
        args.split,
        args.audit_only,
        args.overwrite,
        args.raw_root,
        args.aligned_root,
        args.report,
    )
    print(
        "Teacher scale audit complete: sequences={} audit_only={}".format(
            len(report["sequences"]), report["audit_only"]
        )
    )


if __name__ == "__main__":
    main()
