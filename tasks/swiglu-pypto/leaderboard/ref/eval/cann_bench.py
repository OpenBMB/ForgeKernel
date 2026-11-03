#!/usr/bin/env python3
"""Evaluate a PyPTO candidate with CANN-Bench and emit ForgeKernel's eval JSON.

The official CANN-Bench report is the only correctness and timing source. PyPTO uses
``--perf-metric-strategy trace_view``: its ``elapsed_us`` values are AI-Core E2E timings,
not host wall time. The aggregate exposed to ForgeKernel is the geometric mean over all
public cases, with no reference fields (a reference measured through its own tree is not
an in-job reference).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXPECTED_CANN_REF = "059cb61d41b279ad021160517bd2a5f6b7a05e01"
BASIS = "cann_trace_view"


def geometric_mean(values: list[float]) -> float:
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError("geometric mean needs finite positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def report_to_eval(report: dict) -> dict:
    """Convert one CANN-Bench single-pass report into ForgeKernel's eval contract."""
    operators = report.get("operators") or []
    if len(operators) != 1:
        return {"correctness": {"pass": False,
                                "error": f"expected one operator report, got {len(operators)}"},
                "performance": {}}
    op = operators[0]
    cases = op.get("cases") or []
    failures = []
    for case in cases:
        accuracy = case.get("accuracy") or {}
        if case.get("status") != "success" or not accuracy.get("passed", False):
            detail = (case.get("error_msg") or accuracy.get("error_msg")
                      or case.get("failure_type") or "accuracy failed")
            failures.append(f"{case.get('case_id', '?')}: {detail}")
    total = int(op.get("total_cases") or len(cases))
    passed = int(op.get("passed_cases") or 0)
    correct = bool(cases) and passed == total == len(cases) and not failures
    performance: dict = {}
    if correct:
        try:
            names, milliseconds = [], []
            for case in cases:
                elapsed = float(case["elapsed_us"])
                if elapsed <= 0 or not math.isfinite(elapsed):
                    raise ValueError(f"invalid elapsed_us={elapsed}")
                names.append(str(case["case_id"]))
                milliseconds.append(elapsed / 1000.0)
            aggregate = geometric_mean(milliseconds)
            workloads = [
                {"name": name, "bwd_median_ms": ms, "device_median_ms": ms}
                for name, ms in zip(names, milliseconds)
            ]
            performance = {
                "aggregate": "geomean",
                "n_workloads": len(workloads),
                "bwd_median_ms": aggregate,
                "device_median_ms": aggregate,
                "device_window_basis": BASIS,
                "workloads": workloads,
            }
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"performance data unavailable: {exc}")
            correct = False
    return {
        "correctness": {
            "pass": correct,
            "passed_cases": passed,
            "total_cases": total,
            "error": "; ".join(failures) if failures else "",
        },
        "performance": performance,
    }


def _latest_report(root: Path) -> Path | None:
    candidates = [p for p in root.glob("*_eval_*.json") if p.is_file()
                  and not p.name.startswith("forge_")]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _run_logged(cmd: list[str], *, cwd: Path, env: dict, log: Path) -> int:
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        return proc.wait()


def evaluate(workspace: Path) -> dict:
    config = json.loads((workspace / "eval" / "cann_task.json").read_text(encoding="utf-8"))
    cann_dir = Path(os.environ.get("CANN_BENCH_DIR", "")).expanduser().resolve()
    if not cann_dir.is_dir():
        raise RuntimeError("CANN_BENCH_DIR must name the CANN-Bench 2.0.0 worktree")
    actual_ref = subprocess.run(["git", "-C", str(cann_dir), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    if actual_ref != EXPECTED_CANN_REF and os.environ.get("ALLOW_TASK_REF_MISMATCH") != "1":
        raise RuntimeError(f"CANN-Bench ref is {actual_ref}, expected {EXPECTED_CANN_REF}; "
                           "baseline metadata differs across refs")
    python = os.environ.get("FK_ASCEND_PYTHON", sys.executable)
    reports = workspace / config.get("reports_dir", ".cann_results")
    reports.mkdir(parents=True, exist_ok=True)
    source = (workspace / config.get("source_dir", ".")).resolve()
    if config.get("clean_build", True):
        # kernel_eval reuses an existing wheel. That is wrong for an optimization loop:
        # every job must build the current commit, never the previous candidate's artifact.
        for name in ("build", "dist", "cann_bench.egg-info"):
            shutil.rmtree(source / name, ignore_errors=True)
        for name in ("cann_bench/kernel_impl.py", "_compile.log"):
            (source / name).unlink(missing_ok=True)
    timeout = str(int(os.environ.get("CANN_OPERATOR_TIMEOUT",
                                    max(60, int(os.environ.get("TASK_TIMEOUT", "7200")) - 60))))
    cmd = [
        python, "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann",
        "--source-dir", str(source),
        "--task-dir", config["task_dir"],
        "--operator", config["operator"],
        "--device-id", "0",
        "--processes-per-card", "1",
        "--timeout-per-operator", timeout,
        "--warmup", str(config.get("warmup", 3)),
        "--repeat", str(config.get("repeat", 5)),
        "--perf-metric-strategy", "trace_view",
        "--reports-dir", str(reports),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cann_dir / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    # kernel_eval.cli imports cann_bench before it rebuilds the submission. A previous
    # candidate with an import-time error must not poison every later REVERT/RETRY round.
    removed = subprocess.run([python, "-m", "pip", "uninstall", "-y", "cann_bench"],
                             capture_output=True, text=True, cwd=cann_dir)
    print("[cann-adapter] previous candidate removal:",
          (removed.stdout + removed.stderr).strip()[-500:], flush=True)
    print("[cann-adapter]", " ".join(cmd), flush=True)
    rc = _run_logged(cmd, cwd=cann_dir, env=env, log=reports / "evaluator.log")
    report_path = _latest_report(reports)
    if report_path is None:
        tail = (reports / "evaluator.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        return {"correctness": {"pass": False,
                                "error": f"CANN-Bench exited {rc} without a JSON report: {tail}"},
                "performance": {}}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report_to_eval(report)
    result["cann_bench"] = {"report": str(report_path), "evaluator_rc": rc,
                            "tasks_version": report.get("tasks_version"),
                            "hardware": (report.get("setup_info", {}).get("environment", {})
                                         .get("npu"))}
    (reports / "forge_eval.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    try:
        result = evaluate(Path(args.workspace).resolve())
    except Exception as exc:  # harness configuration failure: visible in the job log
        print(f"[cann-adapter] harness error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("=== JSON ===")
    print(json.dumps(result, separators=(",", ":")))
    # A wrong candidate is a valid measurement outcome, not a transport failure. Returning
    # zero lets ForgeKernel record REVERT with the official error instead of losing the log.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
