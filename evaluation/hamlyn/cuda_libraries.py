"""Discover CUDA shared-library directories without importing torch."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def cuda_library_directories(
    python_prefix: Path | None = None,
    site_packages: Iterable[Path] | None = None,
    environment: Mapping[str, str] | None = None,
    system_cuda_roots: Sequence[Path] = (Path("/usr/local/cuda"),),
) -> tuple[Path, ...]:
    python_prefix = Path(sys.prefix) if python_prefix is None else Path(python_prefix)
    environment = os.environ if environment is None else environment
    if site_packages is None:
        site_packages = tuple(Path(value) for value in site.getsitepackages())

    candidates: list[Path] = []
    for root in site_packages:
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.is_dir():
            candidates.extend(sorted(nvidia_root.glob("*/lib")))
        candidates.append(Path(root) / "torch" / "lib")
    candidates.append(python_prefix / "lib")

    cuda_home = environment.get("CUDA_HOME") or environment.get("CUDA_PATH")
    cuda_roots = ([Path(cuda_home)] if cuda_home else []) + list(system_cuda_roots)
    for root in cuda_roots:
        candidates.extend((root / "lib64", root / "lib"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return tuple(unique)


def main() -> None:
    print(os.pathsep.join(str(path) for path in cuda_library_directories()))


if __name__ == "__main__":
    main()
