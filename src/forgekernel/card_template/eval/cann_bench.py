#!/usr/bin/env python3
"""Evaluate a PyPTO candidate with CANN-Bench and emit ForgeKernel's eval JSON.

The official CANN-Bench report is the only correctness and timing source. PyPTO uses
``--perf-metric-strategy trace_view``: its ``elapsed_us`` values are AI-Core E2E timings,
not host wall time. The aggregate exposed to ForgeKernel is the geometric mean over all
public cases, with no reference fields (a reference measured through its own tree is not
an in-job reference).

Three outcomes, never folded together. A candidate the official judge rejects is a
**correctness** failure and the student's to fix. An evaluator that exited non-zero
or produced no report is a **job** failure. A report that does not carry the window
this card declared is a **measurement** failure: the kernel may be perfectly
correct -- the first GQA M0 gate was 20/20 and still unusable, because the card
collected ``kernel_details`` while the adapter read ``trace_view``. Reporting that
third case as a correctness failure points the next iteration at the kernel
instead of the configuration, which is the one direction no evidence can correct.
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
import tempfile
import time

# Also importable by CPU-only tests without installing the card as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ascend_profiler import collect, emit_artifact

BASIS = "cann_trace_view"


def geometric_mean(values: list[float]) -> float:
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError("geometric mean needs finite positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


CASE_FIELDS = ("baseline_perf_us", "speedup", "t_hw_us", "perf_score", "op_times",
               "memory_peak_mb", "memory_baseline_mb", "perf_collection_failed",
               "device_kernels", "official_elapsed_us", "elapsed_us_source",
               "performance_error_msg", "accuracy", "status")
SCORE_FIELDS = ("score", "compilation_score", "compile_runtime_score", "function_score",
                "performance_score", "avg_speedup", "compile_runtime_fail_cases")


def report_to_eval(report: dict, *, config: dict | None = None) -> dict:
    config = config or {}
    """Convert one CANN-Bench single-pass report into ForgeKernel's eval contract."""
    operators = report.get("operators") or []
    if len(operators) != 1:
        return {"correctness": {"pass": False,
                                "error": f"expected one operator report, got {len(operators)}"},
                "performance": {}}
    op = operators[0]
    cases = op.get("cases") or []
    official = {
        "tasks_version": report.get("tasks_version"),
        "framework_version": report.get("framework_version"),
        "environment": (report.get("setup_info") or {}).get("environment", {}),
        "operator": op.get("operator"),
        "overall_score": report.get("overall_score"),
        "scores": {k: op.get(k) for k in SCORE_FIELDS},
        "cases": [{"case_id": c.get("case_id"), "elapsed_us": c.get("elapsed_us"),
                   **{k: c.get(k) for k in CASE_FIELDS}} for c in cases],
        "comparison": "benchmark baseline and hardware bound; NOT campaign reference",
    }
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
    basis = str(config.get("timing_basis") or BASIS)
    backend = str(config.get("profiler_backend") or "msprof")
    official["timing_basis"] = basis
    performance: dict = {}
    measurement: dict = {}
    if correct:
        try:
            names, milliseconds = [], []
            for case in cases:
                elapsed_us, elapsed_source = _case_window(case, basis)
                case["official_elapsed_us"] = elapsed_us
                case["elapsed_us_source"] = elapsed_source
                official["cases"][len(names)].update({
                    "official_elapsed_us": elapsed_us, "elapsed_us_source": elapsed_source})
                elapsed = float(elapsed_us)
                if elapsed <= 0 or not math.isfinite(elapsed):
                    raise MeasurementUnavailable(
                        f"{case.get('case_id', '?')}: {basis} window is {elapsed}")
                names.append(str(case["case_id"]))
                milliseconds.append(elapsed / 1000.0)
            aggregate = geometric_mean(milliseconds)
            workloads = [
                {"name": name, "bwd_median_ms": ms, "device_median_ms": ms,
                 "cann_bench": {k: case.get(k) for k in CASE_FIELDS}}
                for name, ms, case in zip(names, milliseconds, cases)
            ]
            performance = {
                "aggregate": "geomean",
                "n_workloads": len(workloads),
                "bwd_median_ms": aggregate,
                "device_median_ms": aggregate,
                "device_window_basis": basis,
                "workloads": workloads,
            }
        except MeasurementUnavailable as exc:
            # The official judge already passed every case. The kernel is not at
            # fault and must not be told it is: this is the card's declared window
            # missing from the report the card asked for.
            measurement = {"status": "unavailable", "timing_basis": basis,
                           "perf_metric_strategy": config.get("perf_metric_strategy"),
                           "reason": str(exc)}
            performance = {}
    return {
        "correctness": {
            "pass": correct,
            "passed_cases": passed,
            "total_cases": total,
            "error": "; ".join(failures) if failures else "",
        },
        "performance": performance,
        **({"measurement": measurement} if measurement else {}),
        "cann_bench": official,
        # ``backend`` is the tool this job ran, never the window the gate is timed
        # in: a card collects msprof and still times on trace_view. The window is
        # ``performance.device_window_basis`` / ``cann_bench.timing_basis``.
        "profiler": {"backend": backend, "timing_basis": basis,
                     "status": "available" if performance else "unavailable",
                     "source": f"CANN-Bench {basis}",
                     "cases": [{"case_id": c.get("case_id"),
                                "trace_view": (c.get("op_times") or {}).get("trace_view")}
                               for c in cases]},
    }


def case_scope(config: dict, requested: str | None) -> dict:
    """Which cases this job evaluates, as evaluator argv plus a scope declaration.

    Every case of a shape-specialized operator is its own JIT compile, so a
    20-case job pays 20 compiles whether the question needs them or not. That is
    what made GQA uniterable: a single architecture question cost a full gate, and
    a few compiles exhausted an outer-loop budget. A scout job answers the
    question on a handful of representative cases instead.

    Two things keep this from becoming a cheap way to claim a win. The card must
    declare its evaluator's case-filter flag -- this adapter does not guess a
    vendor CLI contract, and without the declaration a narrowed job is refused
    rather than silently run whole. And the returned ``scope`` travels into the
    eval JSON, where loop/parse.py already refuses any reading that carries one:
    a subset is evidence for the teacher, never a gate.
    """
    names = [c.strip() for c in str(requested or "").split(",") if c.strip()]
    if not names:
        return {"argv": [], "n": 0, "scope": None}
    flag = str(config.get("case_filter_flag") or "").strip()
    if not flag:
        raise RuntimeError(
            "FK_CANN_CASES asked for a subset but eval/cann_task.json declares no "
            "case_filter_flag: add the flag this CANN-Bench build uses to select "
            "cases (verify it on hardware first) — refusing to run the full set "
            "under a name that says otherwise")
    known = [str(c) for c in config.get("cases") or []]
    if known and (unknown := [n for n in names if n not in known]):
        raise RuntimeError(f"unknown case id(s) {unknown}; this task declares {known}")
    total = len(known) or None
    return {"argv": [flag, ",".join(names)], "n": len(names),
            "scope": {"workloads": len(names), "of": total or "unknown",
                      "cases": names, "rep": "scout",
                      "why": "subset for architecture pricing; not a gate reading"}}


class MeasurementUnavailable(Exception):
    """The report does not carry the window this card declared."""


def _case_window(case: dict, basis: str) -> tuple[object, str]:
    """One case's declared timing window, or MeasurementUnavailable with the reason."""
    if basis == BASIS:
        trace = (case.get("op_times") or {}).get("trace_view") or {}
        if trace.get("aicore_e2e") is None:
            raise MeasurementUnavailable(
                f"{case.get('case_id', '?')}: timing_basis is {BASIS!r} but the report "
                f"carries no op_times.trace_view.aicore_e2e — the evaluator was most "
                f"likely run with a perf_metric_strategy other than 'trace_view'")
        return trace["aicore_e2e"], "op_times.trace_view.aicore_e2e"
    if case.get("elapsed_us") is None:
        raise MeasurementUnavailable(
            f"{case.get('case_id', '?')}: timing_basis is {basis!r} but the report "
            f"carries no elapsed_us")
    return case["elapsed_us"], "case.elapsed_us"


def evaluator_failure(result: dict, rc: int) -> None:
    """A partial report is evidence of failure, never a complete gate measurement."""
    detail = result["correctness"].get("error") or ""
    result["correctness"].update({"pass": False,
        "error": (detail + "; " if detail else "")
                 + f"CANN-Bench evaluator exited {rc}; partial report is not a gate"})
    result["performance"] = {}


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
    if not config.get("sealed_source"):
        return _evaluate(workspace, config)
    import closed_ref
    payload = workspace / config["sealed_source"]
    manifest = closed_ref.unseal_bundle(payload.read_bytes(), closed_ref.key_from_file(str(payload)))
    # The package is materialized only on the execution machine, never the controller.
    with tempfile.TemporaryDirectory(prefix=".sealed_source-", dir=workspace) as tmp:
        root = Path(tmp)
        for name, text in manifest["files"].items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("sealed source path escapes package")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            if path.suffix == ".sh":
                path.chmod(0o700)
        try:
            return _evaluate(workspace, {**config, "source_dir": str(root)})
        finally:
            subprocess.run([os.environ.get("FK_ASCEND_PYTHON", sys.executable),
                            "-m", "pip", "uninstall", "-y", "cann_bench"],
                           capture_output=True, timeout=60)


def _evaluate(workspace: Path, config: dict) -> dict:
    if not os.environ.get("CANN_BENCH_DIR"):
        raise RuntimeError("CANN_BENCH_DIR is required")
    cann_dir = Path(os.environ["CANN_BENCH_DIR"]).expanduser().resolve()
    if not cann_dir.is_dir():
        raise RuntimeError("CANN_BENCH_DIR must name a CANN-Bench worktree")
    actual_ref = subprocess.run(["git", "-C", str(cann_dir), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    expected_ref = str(config.get("expected_cann_ref") or "").strip()
    if (expected_ref and actual_ref != expected_ref
            and os.environ.get("ALLOW_TASK_REF_MISMATCH") != "1"):
        raise RuntimeError(f"CANN-Bench ref is {actual_ref}, expected {expected_ref}; "
                           "baseline metadata differs across refs")
    python = os.environ.get("FK_ASCEND_PYTHON", sys.executable)
    report_root = workspace / config.get("reports_dir", ".cann_results")
    report_root.mkdir(parents=True, exist_ok=True)
    # No stale report can turn a crashed evaluator into a passing measurement.
    reports = Path(tempfile.mkdtemp(prefix="eval-", dir=report_root))
    backend = config.get("profiler_backend", "cann_trace_view")
    if backend not in ("cann_trace_view", "msprof"):
        raise ValueError(f"unsupported Ascend profiler backend: {backend}")
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
        "--perf-metric-strategy", config.get("perf_metric_strategy", "trace_view"),
        "--reports-dir", str(reports),
    ]
    scope = case_scope(config, os.environ.get("FK_CANN_CASES"))
    cmd += scope["argv"]
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
    # A job's wall clock is the only cost reading that is always available, and
    # without it a compile-bound campaign and a slow kernel are indistinguishable
    # from the outside: the GQA attempt lost outer-loop budget to JIT compiles that
    # left no trace anywhere a diagnosis could read. Written whatever the outcome,
    # so a timed-out or crashed job still says where its time went.
    started = time.monotonic()
    rc = _run_logged(cmd, cwd=cann_dir, env=env, log=reports / "evaluator.log")
    cost = {"evaluator_seconds": round(time.monotonic() - started, 1),
            "budget_seconds": int(timeout), "cases_requested": scope["n"] or "all",
            "note": "wall clock, build and device time together; a per-phase split "
                    "needs a compile hook certified on hardware"}
    report_path = _latest_report(reports)
    if report_path is None:
        tail = (reports / "evaluator.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        emit_artifact("cann/evaluator.log", (reports / "evaluator.log").read_bytes()[-1048576:])
        return {"correctness": {"pass": False,
                                "error": f"CANN-Bench exited {rc} without a JSON report: {tail}"},
                "performance": {}, "cann_bench": {"cost": cost, "case_scope": scope["scope"]},
                "profiler": {"backend": backend, "status": "unavailable"}}
    raw = report_path.read_bytes()
    report = json.loads(raw)
    result = report_to_eval(report, config=config)
    artifacts = [emit_artifact("cann/official_report.json", raw),
                 emit_artifact("cann/evaluator.log",
                               (reports / "evaluator.log").read_bytes()[-1048576:])]
    if rc:
        evaluator_failure(result, rc)
    if backend == "msprof":
        result["profiler"], prof_artifacts = collect(reports, backend)
        artifacts.extend(prof_artifacts)
    if scope["scope"]:
        # A subset is a pricing reading and must never clear a gate. loop/parse.py
        # refuses any eval JSON carrying workload_scope, so this is the existing
        # guard rather than a second rule that could drift from it.
        result["workload_scope"] = scope["scope"]
    result["cann_bench"].update({"report": str(report_path), "evaluator_rc": rc,
                                  "commit": actual_ref, "artifacts": artifacts,
                                  "cost": cost, "case_scope": scope["scope"]})
    (reports / "forge_eval.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            config = json.loads((Path(args.workspace) / "eval/cann_task.json").read_text())
            if config.get("profiler_backend") == "msprof" and not shutil.which("msprof"):
                raise RuntimeError("msprof executable unavailable")
            import pypto
            import torch
            import torch_npu
            value = torch.ones(4, device="npu").sum().item()
            if value != 4:
                raise RuntimeError("NPU smoke failed")
            print(f"ASCEND-PREFLIGHT-OK pypto={pypto.__version__} "
                  f"torch_npu={torch_npu.__version__} "
                  f"backend={config.get('profiler_backend', BASIS)}")
            return 0
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
