#!/bin/bash
# PyPTO / CANN-Bench GPU job entry point. The candidate and reference run in separate trees.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
PYTHON="${FK_ASCEND_PYTHON:-python3}"
export PATH="$(dirname "$PYTHON"):${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CANN_BENCH_DIR:?CANN_BENCH_DIR is required}/src:${PYTHONPATH:-}"
export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:?PTO_TILE_LIB_CODE_PATH is required}"

TASK="${1:-help}"
shift || true
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"

echo "=== gpu_entry.sh (PyPTO, CANN-Bench) ==="
echo "  task:     $TASK"
echo "  hostname: $(hostname)"
echo "  python:   $PYTHON"
echo "  NPU:      ${ASCEND_RT_VISIBLE_DEVICES:-unset} (logical device 0)"
echo "  CANN:     $CANN_BENCH_DIR"

run_with_timeout() {
    echo "[gpu_entry.sh] timeout=${TASK_TIMEOUT}s"
    set +e
    timeout --signal=KILL "$TASK_TIMEOUT" "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 137 ]; then
        echo "[gpu_entry.sh] KILLED: task exceeded ${TASK_TIMEOUT}s timeout"
        exit 137
    fi
    return "$rc"
}

case "$TASK" in
    correctness|perf|all|cardref|ref|cann)
        run_with_timeout "$PYTHON" eval/cann_bench.py --workspace "$WORKSPACE"
        ;;
    ncu)
        echo "[gpu_entry.sh] NCU is not an Ascend/PyPTO channel; use the CANN trace_view report"
        ;;
    cmd)
        run_with_timeout bash -c "$*"
        ;;
    help|*)
        sed -n '2,8p' "$0"
        exit 1
        ;;
esac

echo ""
echo "[gpu_entry.sh] task=$TASK finished"
