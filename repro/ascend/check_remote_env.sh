#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/remote_env.sh"

ACTUAL_REF="$(git -C "${CANN_BENCH_DIR}" rev-parse HEAD)"
EXPECTED_REF="${EXPECTED_TASK_REF:-059cb61d41b279ad021160517bd2a5f6b7a05e01}"
if [ "${ACTUAL_REF}" != "${EXPECTED_REF}" ]; then
    echo "CANN-Bench ref mismatch: ${ACTUAL_REF}" >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
"${FK_ASCEND_PYTHON}" -I - <<'PY'
import pypto
import torch
import torch_npu
print("pypto", pypto.__version__, pypto.__file__)
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
print("device", torch_npu.npu.get_device_name(0))
print("npu smoke", torch.ones(16, device="npu").sum().item())
PY
echo "CANN-Bench ref ${ACTUAL_REF}"
echo "remote PyPTO environment OK"
