#!/usr/bin/env python3
"""Check GQA's narrowed-job mechanism against the pinned CANN-Bench before a campaign.

The scout path is only worth having if it is the same measurement over fewer cases.
This script proves that from the outside, without the GPU: it builds the isolated
official task root the adapter builds, then reads BOTH roots with CANN-Bench's own
``CannCaseLoader`` and compares every case field that a reading depends on (id,
rel path, input shapes, dtypes, attrs, published baseline and ``t_hw``).

    export CANN_BENCH_DIR=/path/to/CANN-Bench            # pinned worktree
    python3 scripts/validate_pypto_scout.py cases
    python3 scripts/validate_pypto_scout.py gate-refusal

``cases`` needs the bench worktree; ``gate-refusal`` needs nothing but this repo.
The job-level checks (a scout job must evaluate exactly the selected cases, and a
job the adapter stops must still report its cost) are the commands printed at the
end: they need the NPU, so they are run there, not here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "src/forgekernel/card_template/eval/cann_bench.py"


def _adapter():
    sys.path.insert(0, str(ADAPTER.parent))
    spec = importlib.util.spec_from_file_location("cann_bench_validate", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task_config(task: Path) -> dict:
    return json.loads((task / "leaderboard/card/eval/cann_task.json").read_text())


def check_cases(task: Path) -> int:
    """The isolated root and the official root must describe the same cases."""
    cann_dir = Path(os.environ.get("CANN_BENCH_DIR") or "").expanduser()
    if not cann_dir.is_dir():
        print("CANN_BENCH_DIR must name the pinned CANN-Bench worktree", file=sys.stderr)
        return 2
    sys.path.insert(0, str(cann_dir / "src"))
    try:
        from kernel_eval.benches.cann_loader import CannCaseLoader
    except ImportError as exc:
        print(f"cannot import CANN-Bench's loader ({exc}); check CANN_BENCH_DIR",
              file=sys.stderr)
        return 2
    config = _task_config(task)
    selection = _adapter().case_selection(config, "scout")
    # The pinned bench holds every operator and case numbers repeat across them
    # (gqa_1 and mla_1 are both "_1"), so the official side has to be the one task
    # this card names -- resolved by the same helper the adapter narrows with, not
    # by guessing from the directory name.
    _, _, rel_path = _adapter()._official_task_paths(cann_dir, config)
    workdir = Path(tempfile.mkdtemp(prefix="fk-scout-validate-"))
    try:
        scout_op, root = _adapter().scoped_task_dir(cann_dir, config, selection.names,
                                                    workdir)
        official = {c.case_id: c for c in CannCaseLoader(
            tasks_root=str(cann_dir / "tasks")).scan_all() if c.rel_path == rel_path}
        scoped = {c.case_id: c for c in CannCaseLoader(
            tasks_root=str(root / "tasks")).scan_all() if c.rel_path == rel_path}
        wanted = list(selection.names)
        print(f"official cases: {len(official)}; isolated root: {len(scoped)} "
              f"({scout_op.relative_to(root).as_posix()})")
        problems = []
        for name in wanted:
            number = name.rsplit("_", 1)[-1]
            match = [cid for cid in official if cid.rsplit("_", 1)[-1] == number]
            if len(match) != 1:
                problems.append(f"{name}: {len(match)} official cases carry that number")
                continue
            official_case, scoped_case = official[match[0]], scoped.get(match[0])
            if scoped_case is None:
                problems.append(f"{match[0]}: absent from the isolated root")
                continue
            differs = []
            for field in ("rel_path", "operator", "input_shapes", "dtypes", "attrs",
                          "value_ranges", "baseline_perf_us", "t_hw_us"):
                left, right = getattr(official_case, field), getattr(scoped_case, field)
                if left != right:
                    differs.append(f"{field}: {right!r} != official {left!r}")
            problems.extend(f"{match[0]}.{problem}" for problem in differs)
            print(f"  {match[0]}: "
                  f"{'shapes/dtypes/attrs/baselines match' if not differs else 'DIFFERS: ' + '; '.join(differs)}")
        if extra := sorted(set(scoped) - set(official)):
            problems.append(f"isolated root holds cases the task does not have: {extra}")
        if len(scoped) != len(wanted):
            problems.append(f"isolated root holds {len(scoped)} cases, not {len(wanted)}")
        problems.extend(_adapter().scope_violations(
            {"operators": [{"cases": [{"case_id": cid} for cid in scoped]}]}, wanted))
        if problems:
            print("\nFAIL")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\nPASS: the scout root is the official task over "
              f"{len(wanted)} of {len(official)} cases; ids, goldens and baselines match")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def check_gate_refusal(task: Path) -> int:
    """A scoped reading must be refused as a gate, by the loop's own parser."""
    sys.path.insert(0, str(ROOT / "src"))
    from forgekernel.loop.parse import parse_eval
    config = _task_config(task)
    selection = _adapter().case_selection(config, "scout")
    result = {"correctness": {"pass": True},
              "performance": {"device_median_ms": 0.1, "bwd_median_ms": 0.1},
              "workload_scope": selection.scope}
    verdict = parse_eval("=== JSON ===\n" + json.dumps(result))
    if verdict.correctness or verdict.device_median_ms is not None:
        print("FAIL: a scoped eval JSON was read as a gate")
        return 1
    print(f"PASS: a {len(selection.names)}-case reading is refused as a gate "
          f"({verdict.error[:120]}...)")
    return 0


REMAINING = """\
On the Ascend host (NPU, one job each; nothing here can be checked on CPU):

1. scout evaluates exactly the selected cases
   bash gpu_entry.sh scout
   # expect: 'cases: scout (pricing reading, not a gate)',
   #         '[cann-adapter] scope: 5 of 20 cases (gqa_1,gqa_5,gqa_8,gqa_11,gqa_16)',
   #         an official report holding exactly those five case_ids,
   #         'workload_scope' in the eval JSON, and 'cost': {'cases_requested': 5, ...}
   # also check: the job's wall clock is materially below a full gate's.

2. the gate is still the whole declared set
   bash gpu_entry.sh all
   # expect: no scope line, 20 cases in the report, no 'workload_scope',
   #         'cost': {'cases_requested': 'all', ...}
   FK_CANN_CASES=scout bash gpu_entry.sh all   # must print the 'ignoring' line and run 20

3. a job the adapter stops still reports its cost
   FK_CANN_EVAL_BUDGET=600 bash gpu_entry.sh scout
   # expect exit 137, 'evaluator stopped: the evaluator exceeded its 600s budget',
   # an eval JSON with 'cost': {'interrupted': true, ...} and 'workload_scope',
   # empty correctness/performance readings, and no process left holding the NPU
   # (npu-smi info after the job).

4. the teacher/recon path
   # notes/experiment.json: {"hypothesis": ..., "edits": [...], "repeat": 3,
   #                         "cases": "scout"}
   python3 notes/ascend_probe.py notes/experiment.json
   # expect 'PROBE SCOUT ...' and never 'PROBE PASS': a scout cannot certify a lever.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["cases", "gate-refusal", "all"],
                        nargs="?", default="all")
    parser.add_argument("--task", default=str(ROOT / "tasks/gqa-pypto"))
    args = parser.parse_args(argv)
    task = Path(args.task).resolve()
    status = 0
    if args.check in ("cases", "all"):
        status |= check_cases(task)
    if args.check in ("gate-refusal", "all"):
        status |= check_gate_refusal(task)
    print("\n" + REMAINING)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
