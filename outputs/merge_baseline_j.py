"""Compatibility entry point for merging the configured Baseline-J checkpoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.merge_student_checkpoint import main


if __name__ == "__main__":
    main()
