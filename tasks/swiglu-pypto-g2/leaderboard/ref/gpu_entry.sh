#!/bin/bash
# PyPTO / CANN-Bench GPU job entry point. The candidate and reference run in separate trees.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"
PYTHON="${FK_ASCEND_PYTHON:-python3}"
export PATH="$(dirname "$PYTHON"):${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CANN_BENCH_DIR:?CANN_BENCH_DIR is required}/src:${PYTHONPATH:-}"
if [ -n "${PTO_TILE_LIB_CODE_PATH:-}" ]; then
    export PTO_TILE_LIB_CODE_PATH
fi

TASK="${1:-help}"
shift || true
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"

echo "=== gpu_entry.sh (PyPTO, CANN-Bench) ==="
echo "  task:     $TASK"
echo "  hostname: $(hostname)"
echo "  python:   $PYTHON"
echo "  NPU:      ${ASCEND_RT_VISIBLE_DEVICES:-unset} (logical device 0)"
echo "  CANN:     $CANN_BENCH_DIR"

# A gate task judges and times the card's whole declared case set. ``scout`` is the
# explicit request for its representative subset (``FK_CANN_CASES`` may name ids
# instead); the adapter documents the subset in ``workload_scope``, which the loop
# refuses as a gate reading. An ambient FK_CANN_CASES must therefore never narrow a
# gate task: that would turn a pricing reading into the row a KEEP was decided on.
case "$TASK" in
    preflight|correctness|perf|all|cardref|ref|cann|profiler|msprof)
        if [ -n "${FK_CANN_CASES:-}" ]; then
            echo "  cases:    all (ignoring FK_CANN_CASES=${FK_CANN_CASES} for gate task '$TASK')"
            unset FK_CANN_CASES
        fi
        ;;
    scout)
        export FK_CANN_CASES="${1:-scout}"
        echo "  cases:    $FK_CANN_CASES (pricing reading, not a gate)"
        ;;
    *)
        ;;
esac

run_with_timeout() {
    echo "[gpu_entry.sh] timeout=${TASK_TIMEOUT}s"
    local started=$SECONDS
    set +e
    # SIGTERM first so the adapter can flush its log and its cost line; KILL only
    # if it ignores that. A bare KILL discarded everything, which is how a probe
    # that spent forty minutes in JIT came back saying nothing about where the
    # time went -- the one fact needed to decide whether to narrow the job.
    timeout --signal=TERM --kill-after=30 "$TASK_TIMEOUT" "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] || [ "$rc" -eq 143 ]; then
        echo "[gpu_entry.sh] TIMEOUT after $((SECONDS - started))s of ${TASK_TIMEOUT}s" \
             "(task=$TASK, signal rc=$rc). The log above is the only record of which" \
             "phase consumed it: JIT compilation and device evaluation share this budget."
        exit 137
    fi
    return "$rc"
}

case "$TASK" in
    preflight)
        run_with_timeout "$PYTHON" eval/cann_bench.py --workspace "$WORKSPACE" --preflight
        ;;
    correctness|perf|all|cardref|ref|cann)
        run_with_timeout "$PYTHON" eval/cann_bench.py --workspace "$WORKSPACE"
        ;;
    scout)
        # The card's representative subset (or the ids given as the next argument).
        # One job, one build: the adapter narrows the official task root instead of
        # re-invoking the evaluator per case. Never a gate -- see workload_scope.
        run_with_timeout "$PYTHON" eval/cann_bench.py --workspace "$WORKSPACE"
        ;;
    profiler|msprof)
        # Profiling is collected together with the official evaluation; no unrelated
        # synthetic kernel or CUDA channel can satisfy this task's mechanism gates.
        run_with_timeout "$PYTHON" eval/cann_bench.py --workspace "$WORKSPACE"
        ;;
    ncu)
        echo "[gpu_entry.sh] unavailable: NCU cannot profile Ascend" >&2
        exit 2
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
