#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}"

run_baseline() {
    local baseline_id="$1"
    local project_dir="$2"
    local config_path="$3"
    local log_path="${LOG_DIR}/baseline_${baseline_id}.log"

    printf 'Starting Baseline %s on CUDA_VISIBLE_DEVICES=%s\n' \
        "${baseline_id}" "${GPU_ID}"
    (
        cd "${SCRIPT_DIR}/${project_dir}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" \
            train_direct_teacher_distillation.py \
            --config "${config_path}" 2>&1 | tee "${log_path}"
    )
}

run_baseline B vggtoda3-baseline-B configs/baselines/B.yaml
run_baseline C vggtoda3-baseline-C configs/baselines/C.yaml
run_baseline D vggtoda3-baseline-D configs/baselines/D.yaml

printf 'Baselines B, C, and D completed successfully. Logs: %s\n' "${LOG_DIR}"
