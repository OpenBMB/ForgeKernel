# Sourced on an Ascend host before a ForgeKernel job. No credentials belong here.
FORGE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORGE_WORKSPACE="$(cd "${FORGE_REPO_ROOT}/.." && pwd)"

if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    . "${ASCEND_HOME_PATH}/set_env.sh"
    set -u
fi

export FK_ASCEND_PYTHON="${FK_ASCEND_PYTHON:-python3}"
export CANN_BENCH_DIR="${CANN_BENCH_DIR:-${FORGE_WORKSPACE}/cann-bench}"
if [ -n "${PTO_TILE_LIB_CODE_PATH:-}" ]; then
    export PTO_TILE_LIB_CODE_PATH
fi
export PYTHONPATH="${CANN_BENCH_DIR}/src:${PYTHONPATH:-}"
export PATH="$(dirname "${FK_ASCEND_PYTHON}"):${PATH}"

export TMPDIR="${TMPDIR:-${FORGE_WORKSPACE}/.tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${FORGE_WORKSPACE}/.pip-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${FORGE_WORKSPACE}/.triton-cache}"
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-${FORGE_WORKSPACE}/.ascend-log}"
export ASCEND_LOG_PATH="${ASCEND_LOG_PATH:-${ASCEND_PROCESS_LOG_PATH}}"
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${ASCEND_PROCESS_LOG_PATH}"
