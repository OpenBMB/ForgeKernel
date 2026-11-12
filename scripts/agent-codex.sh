#!/usr/bin/env bash
# One Codex CLI turn. FK_AGENT_DRIVER=goal selects codex-goal.py's checked loop.
# Prompt on stdin, final reply on stdout; logs/errors stay on stderr.
set -eu
MODEL=${1:?model}
EFFORT=${2:-}
TIMEOUT=${3:-7200}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${FK_AGENT_PY:-python3}
if [ "${FK_AGENT_DRIVER:-}" = goal ]; then
    # A generated reference has its own (usually longer) budget/deadline.
    # codex-goal.py owns that clock and terminates its process group on expiry.
    exec "$PY" "$HERE/codex-goal.py" "$MODEL" "$EFFORT" "$TIMEOUT"
fi
exec "$PY" "$HERE/timeout-run.py" "$TIMEOUT" "$PY" "$HERE/codex-run.py" "$MODEL" "$EFFORT"
