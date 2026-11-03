#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 {Exp|SwiGlu} [physical_npu_id]" >&2
    echo "Environment: CANN_BENCH_DIR, PYTHON, CANN_BENCH_RESULTS_ROOT" >&2
}

OPERATOR="${1:-}"
if [[ -n "${2:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$2"
fi
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    echo "Set ASCEND_RT_VISIBLE_DEVICES or pass physical_npu_id; device-id inside the evaluator is always 0." >&2
    usage
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/public_triton"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CANN_BENCH_DIR="${CANN_BENCH_DIR:-${WORKSPACE_DIR}/cann-bench-2.0.0}"
PYTHON="${PYTHON:-${WORKSPACE_DIR}/.venv-cannbench/bin/python}"
EXPECTED_TASK_REF="059cb61d41b279ad021160517bd2a5f6b7a05e01"

case "${OPERATOR}" in
    Exp)
        TASK_DIR="tasks/level1/exp"
        REPORT_SLUG="exp"
        ;;
    SwiGlu)
        TASK_DIR="tasks/level1/swi_glu"
        REPORT_SLUG="swiglu"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python is not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -d "${CANN_BENCH_DIR}/.git" && ! -f "${CANN_BENCH_DIR}/.git" ]]; then
    echo "CANN-Bench worktree not found: ${CANN_BENCH_DIR}" >&2
    exit 1
fi
CANN_BENCH_DIR="$(cd "${CANN_BENCH_DIR}" && pwd)"
ACTUAL_TASK_REF="$(git -C "${CANN_BENCH_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_TASK_REF}" != "${EXPECTED_TASK_REF}" && "${ALLOW_TASK_REF_MISMATCH:-0}" != "1" ]]; then
    echo "CANN-Bench ref mismatch: ${ACTUAL_TASK_REF}" >&2
    echo "Expected official-tasks 2.0.0 source: ${EXPECTED_TASK_REF}" >&2
    echo "The metadata changed after this ref; do not mix scores across refs." >&2
    exit 1
fi

RUNTIME_ROOT="${CANN_BENCH_RUNTIME_ROOT:-$(dirname "${CANN_BENCH_DIR}")/.cannbench-runtime}"
RESULTS_ROOT="${CANN_BENCH_RESULTS_ROOT:-${CANN_BENCH_DIR}/reports/forgekernel-repro}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${RESULTS_ROOT}/${REPORT_SLUG}-${RUN_ID}"
mkdir -p "${REPORT_DIR}" "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/ascend-log" "${RUNTIME_ROOT}/triton-cache" "${RUNTIME_ROOT}/pip-cache"

export PATH="$(dirname "${PYTHON}"):${PATH}"
export HOME="${CANN_BENCH_HOME:-$(dirname "${CANN_BENCH_DIR}")}"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export ASCEND_PROCESS_LOG_PATH="${RUNTIME_ROOT}/ascend-log"
export ASCEND_LOG_PATH="${RUNTIME_ROOT}/ascend-log"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/triton-cache"
export PIP_CACHE_DIR="${RUNTIME_ROOT}/pip-cache"

cd "${CANN_BENCH_DIR}"
./scripts/run_evaluation.sh \
    --source-dir "${SOURCE_DIR}" \
    --task-dir "${TASK_DIR}" \
    --operator "${OPERATOR}" \
    --device-id 0 \
    --processes-per-card 1 \
    --timeout-per-process 1200 \
    --warmup 3 \
    --repeat 5 \
    --reports-dir "${REPORT_DIR}" 2>&1 | tee "${REPORT_DIR}/run.log"
