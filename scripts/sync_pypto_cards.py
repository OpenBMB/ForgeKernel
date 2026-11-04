#!/usr/bin/env python3
"""Refresh generated evaluator files in PyPTO task cards before taking a run snapshot.

This is an explicit maintenance command, not a run-time overlay: existing runs
retain their frozen harness. Review the generated diff before launching a task.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/forgekernel/card_template"


def sync(task: Path) -> None:
    for name in ("card", "ref"):
        tree = task / "leaderboard" / name
        if not tree.is_dir():
            continue
        for source, target in (("eval/cann_bench.py", "eval/cann_bench.py"),
                               ("eval/ascend_profiler.py", "eval/ascend_profiler.py"),
                               ("pypto_gpu_entry.sh", "gpu_entry.sh")):
            shutil.copy2(TEMPLATE / source, tree / target)


if __name__ == "__main__":
    import sys
    for task in sys.argv[1:]:
        sync(Path(task))
