# Sourced by ForgeKernel on the Ascend 950PR host before every remote GPU job.
# No agent credentials belong here.
FORGE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORGE_WORKSPACE="$(cd "${FORGE_REPO_ROOT}/.." && pwd)"

export ASCEND_HOME_PATH=/mnt/home/dataset/shw/Ascend/cann-9.2.0
if [ -f /mnt/home/dataset/shw/Ascend/ascend-toolkit/set_env.sh ]; then
    set +u
    # shellcheck disable=SC1091
    . /mnt/home/dataset/shw/Ascend/ascend-toolkit/set_env.sh
    set -u
elif [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    . "${ASCEND_HOME_PATH}/set_env.sh"
    set -u
fi

export FK_ASCEND_PYTHON="${FK_ASCEND_PYTHON:-${FORGE_WORKSPACE}/.venv-cannbench/bin/python}"
export CANN_BENCH_DIR="${CANN_BENCH_DIR:-${FORGE_WORKSPACE}/cann-bench-2.0.0}"
export PTO_TILE_LIB_CODE_PATH="${PTO_TILE_LIB_CODE_PATH:-/home/shw/Workspace/pto-isa}"
export PYTHONPATH="${CANN_BENCH_DIR}/src:${PYTHONPATH:-}"
export PATH="$(dirname "${FK_ASCEND_PYTHON}"):${PATH}"

# The system root is small/full. All recurring writes stay on the data disk.
export HOME=/mnt/home/dataset/shw
export TMPDIR="${FORGE_WORKSPACE}/.tmp"
export PIP_CACHE_DIR="${FORGE_WORKSPACE}/.pip-cache"
export TRITON_CACHE_DIR="${FORGE_WORKSPACE}/.triton-cache"
export ASCEND_PROCESS_LOG_PATH="${FORGE_WORKSPACE}/.ascend-log"
export ASCEND_LOG_PATH="${FORGE_WORKSPACE}/.ascend-log"
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${ASCEND_PROCESS_LOG_PATH}"
