from __future__ import annotations

from pathlib import Path

from evaluation.hamlyn.cuda_libraries import cuda_library_directories


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cuda_library_discovery_prefers_environment_local_directories(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "endodav"
    site_packages = prefix / "lib/python3.9/site-packages"
    nvrtc = site_packages / "nvidia/cuda_nvrtc/lib"
    cudnn = site_packages / "nvidia/cudnn/lib"
    torch_lib = site_packages / "torch/lib"
    prefix_lib = prefix / "lib"
    cuda_home = tmp_path / "cuda"
    for directory in (nvrtc, cudnn, torch_lib, prefix_lib, cuda_home / "lib64"):
        directory.mkdir(parents=True, exist_ok=True)

    actual = cuda_library_directories(
        python_prefix=prefix,
        site_packages=(site_packages,),
        environment={"CUDA_HOME": str(cuda_home)},
        system_cuda_roots=(),
    )

    assert actual == (
        nvrtc.resolve(),
        cudnn.resolve(),
        torch_lib.resolve(),
        prefix_lib.resolve(),
        (cuda_home / "lib64").resolve(),
    )


def test_endodav_launcher_sets_library_path_before_direct_inference() -> None:
    launcher = (PROJECT_ROOT / "scripts/eval_endodav.bash").read_text(
        encoding="utf-8"
    )
    assert "-m evaluation.hamlyn.cuda_libraries" in launcher
    assert 'export LD_LIBRARY_PATH="' in launcher
    assert (
        'exec "${ENDODAV_PYTHON}" evaluate_hamlyn.py --method endodav "$@"'
        in launcher
    )
    assert "--preflight" not in launcher


def test_eval_all_uses_isolated_endodav_launcher() -> None:
    script = (PROJECT_ROOT / "scripts/eval_all.bash").read_text(encoding="utf-8")
    assert 'bash scripts/eval_endodav.bash --stage all "${FORCE_ARGS[@]}"' in script
