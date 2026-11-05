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

A fourth axis is orthogonal to all three: **scope**. Shape specialization makes every
case its own JIT compile, so a 20-case gate pays 20 compiles whatever the question was,
and architecture exploration could only be bought at the gate's price. ``FK_CANN_CASES``
narrows one job to ``scout`` (the card's declared representative set) or to explicit ids
(``gqa_8``, ``gqa_1,gqa_8``). The pinned CANN-Bench has no case-set flag -- ``--case-id``
takes a single int -- so a subset is expressed the one way its loader already supports:
an isolated official task root whose ``cases.yaml`` lists exactly the selected cases,
with every other task file and the ``metadata/`` baselines symlinked to the official
ones. A scoped result always carries ``workload_scope``, which ``loop/parse.py``
refuses as a gate reading: a subset prices an architecture decision for the teacher,
it never clears a gate.

Cost is written whatever the outcome, including a job the adapter itself stopped: wall
clock, budget and requested case count. A compile-bound campaign and a slow kernel look
identical from the outside without it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import NamedTuple

# Also importable by CPU-only tests without installing the card as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ascend_profiler import collect, emit_artifact, resolve_backend, resolve_timing_basis

BASIS = "cann_trace_view"

# Which cases one job evaluates. ``all`` and its spelling variants are the gate's whole
# declared set; ``scout`` is the card's declared representative subset. Anything else is
# a comma-separated list of case ids.
CASE_ENV = "FK_CANN_CASES"
FULL_SELECTORS = ("all", "full", "gate", "ref", "reference", "confirm")
SCOUT_SELECTOR = "scout"
# The adapter stops the evaluator this much before the job's own budget runs out: the
# reserve is what pays for killing the evaluator's workers and writing the partial eval.
BUDGET_ENV = "FK_CANN_EVAL_BUDGET"
TERM_GRACE_S = 20.0


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


class CaseSelection(NamedTuple):
    """The cases one job evaluates, and the scope record when it is a subset.

    ``names`` empty means the whole declared set (no narrowing, no scope record): a
    card that asks for nothing must keep running the full gate.
    """
    names: list[str]
    alias: str                 # "full" | "scout" | "explicit"
    scope: dict | None


class RunOutcome(NamedTuple):
    """One evaluator run: its exit, and whether this adapter stopped it."""
    rc: int
    interrupted: bool
    reason: str
    seconds: float


def report_to_eval(report: dict, *, config: dict | None = None) -> dict:
    config = config or {}
    """Convert one CANN-Bench single-pass report into ForgeKernel's eval contract."""
    backend = resolve_backend(config)
    basis = resolve_timing_basis(config)
    measurement = {"backend": backend, "timing_basis": basis, "status": "not_measured"}
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
        "timing_basis": basis,
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
    accuracy_ok = bool(cases) and passed == total == len(cases) and not failures
    performance: dict = {}
    if accuracy_ok:
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
            measurement.update(status="available", n_workloads=len(workloads))
        except (MeasurementUnavailable, KeyError, TypeError, ValueError) as exc:
            # The cases passed the official checker; what is missing is the timing
            # window this card declared. That is a measurement/configuration outcome,
            # not a wrong kernel, so it must not enter the correctness/REVERT path.
            measurement.update(status="unavailable", reason=str(exc),
                               perf_metric_strategy=config.get("perf_metric_strategy"))
    return {
        "correctness": {
            "pass": accuracy_ok,
            "passed_cases": passed,
            "total_cases": total,
            "error": "; ".join(failures) if failures else "",
        },
        "performance": performance,
        "cann_bench": official,
        "profiler": {"backend": backend, "timing_basis": basis,
                     "status": "available" if performance else "unavailable",
                     "source": f"CANN-Bench official report ({basis})",
                     "cases": [{"case_id": c.get("case_id"),
                                "trace_view": (c.get("op_times") or {}).get("trace_view")}
                               for c in cases]},
        **({"measurement": measurement} if accuracy_ok else {}),
    }


class MeasurementUnavailable(Exception):
    """The report does not carry the window this card declared."""


class Interrupted(Exception):
    """The evaluator was stopped: its own budget, or a stop signal for this adapter."""


def _case_number(name: str) -> int | None:
    """The case number in a ForgeKernel case id (``gqa_11`` -> 11), else None."""
    tail = str(name).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def case_selection(config: dict, requested: str | None) -> CaseSelection:
    """Which cases this job evaluates, from the card's declaration and ``FK_CANN_CASES``.

    ``scout`` resolves to the card's ``scout_cases``; explicit ids are accepted as
    written; nothing (or ``all``/``full``/``gate``/``ref``/``reference``/``confirm``)
    keeps the whole declared set, so a KEEP or a gate is never read off a subset. A
    typo, an unknown id or a duplicated id is refused rather than quietly running the
    full set under a narrowed name -- that would make the job's own cost reading a
    fiction.
    """
    raw = str(requested or "").strip()
    declared = [str(c) for c in config.get("cases") or []]
    if not raw or raw.lower() in FULL_SELECTORS:
        return CaseSelection([], "full", None)
    if raw.lower() == SCOUT_SELECTOR:
        names = [str(c).strip() for c in config.get("scout_cases") or [] if str(c).strip()]
        if not names:
            raise RuntimeError(f"{CASE_ENV}=scout but eval/cann_task.json declares no "
                               f"scout_cases: refusing to guess a representative set")
        alias = SCOUT_SELECTOR
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if not names:
            raise RuntimeError(f"{CASE_ENV}={raw!r} names no case id: refusing to run the "
                               f"full set under a name that says otherwise")
        alias = "explicit"
    if duplicates := sorted({n for n in names if names.count(n) > 1}):
        raise RuntimeError(f"duplicate case id(s) {duplicates} in {CASE_ENV}={raw!r}")
    if declared and (unknown := [n for n in names if n not in declared]):
        raise RuntimeError(f"unknown case id(s) {unknown}; this task declares {declared}")
    if declared and set(names) == set(declared):
        # The whole declared set named explicitly is the gate, not a subset.
        return CaseSelection([], "full", None)
    return CaseSelection(names, alias, {
        "workloads": ",".join(names),
        "rep": names[0],
        "of": len(declared) or "unknown",
        "alias": alias,
        "why": "subset for architecture pricing; not a gate reading",
    })


def _official_task_paths(cann_dir: Path, config: dict) -> tuple[Path, Path, str]:
    """(operator dir, bench root, bench-relative path) of the official task named here."""
    raw = str(config.get("task_dir") or "").strip()
    if not raw:
        raise RuntimeError("eval/cann_task.json names no task_dir")
    op_dir = Path(raw)
    if not op_dir.is_absolute():
        op_dir = cann_dir / raw
    op_dir = op_dir.resolve()
    if not (op_dir / "cases.yaml").is_file():
        raise RuntimeError(f"a narrowed job needs an operator task dir holding cases.yaml; "
                           f"{op_dir} holds none")
    bench_root = op_dir.parent
    for parent in op_dir.parents:
        if parent.name == "tasks":
            bench_root = parent
            break
    return op_dir, bench_root, op_dir.relative_to(bench_root).as_posix()


def _official_cases(op_dir: Path) -> dict[int, dict]:
    import yaml  # only a narrowed job rewrites the case list, so only it needs this
    doc = yaml.safe_load((op_dir / "cases.yaml").read_text(encoding="utf-8")) or {}
    cases = {}
    for raw in doc.get("cases") or []:
        number = raw.get("case_id")
        if isinstance(number, int):
            cases[number] = raw
    return cases


def _scout_workdir(cann_dir: Path) -> Path:
    """Where an isolated task root is built: under the bench tree when it is writable."""
    try:
        base = cann_dir / ".forgekernel-scout"
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        return Path(tempfile.gettempdir())


def scoped_task_dir(cann_dir: Path, config: dict, names: list[str],
                    workdir: Path) -> tuple[Path, Path]:
    """Build an isolated official task root holding exactly ``names``; (op dir, root).

    The upstream CLI can filter to one ``--case-id`` int and nothing else, so a case set
    is expressed the way its loader already reads it: an operator directory whose
    ``cases.yaml`` lists those cases, under a ``tasks/`` bench root that
    ``resolve_task_dir`` and ``CannCaseLoader`` accept unchanged. ``proto.yaml``,
    ``golden.py`` and the rest stay symlinks to the official files, and the official
    ``metadata/`` (plus ``suite_manifest.json``) is linked next to the isolated root, so
    ids, golden inputs and published baselines are byte-identical to a full run's.
    """
    op_dir, bench_root, rel_path = _official_task_paths(cann_dir, config)
    official = _official_cases(op_dir)
    selected = []
    for name in names:
        number = _case_number(name)
        if number is None or number not in official:
            raise RuntimeError(f"{CASE_ENV} names {name!r}, which "
                               f"{op_dir / 'cases.yaml'} does not hold")
        selected.append(number)
    root = Path(tempfile.mkdtemp(prefix="scout-", dir=workdir))
    dest_root = root / "tasks"
    dest_op = dest_root / rel_path
    dest_op.mkdir(parents=True)
    for path in sorted(op_dir.iterdir()):
        if path.name == "cases.yaml":
            continue  # the filtered list is the only file this job may rewrite
        (dest_op / path.name).symlink_to(path.resolve())
    import yaml
    (dest_op / "cases.yaml").write_text(
        yaml.safe_dump({"cases": [official[n] for n in sorted(selected)]},
                       sort_keys=False, allow_unicode=True, width=4096),
        encoding="utf-8")
    # BaselineStore looks for metadata/ from the bench root upward and reads
    # suite_manifest.json beside it (a versioned suite forbids cross-platform fallback
    # without one): both are linked so a scout case keeps its published baseline.
    for probe in (bench_root, *bench_root.parents):
        if (probe / "metadata").is_dir():
            (dest_root / "metadata").symlink_to((probe / "metadata").resolve())
            manifest = probe / "suite_manifest.json"
            if manifest.is_file():
                (dest_root / "suite_manifest.json").symlink_to(manifest.resolve())
            break
        if probe == cann_dir:
            break
    return dest_op, root


def scope_violations(report: dict, names: list[str]) -> list[str]:
    """Cases the report evaluated that the request did not (or the other way round)."""
    operators = report.get("operators") or []
    cases = (operators[0].get("cases") if operators else []) or []
    reported, unparsed = set(), []
    for case in cases:
        number = _case_number(str(case.get("case_id") or ""))
        if number is None:
            unparsed.append(str(case.get("case_id")))
        else:
            reported.add(number)
    wanted = {n for n in (_case_number(name) for name in names) if n is not None}
    problems = []
    if unparsed:
        problems.append(f"case(s) {unparsed} carry no case number")
    if missing := sorted(wanted - reported):
        problems.append(f"selected case(s) {missing} are absent from the report")
    if extra := sorted(reported - wanted):
        problems.append(f"unselected case(s) {extra} ran in a narrowed job")
    return problems


def evaluator_budget() -> int | None:
    """Seconds this adapter lets the evaluator run before stopping it itself.

    ``FK_CANN_EVAL_BUDGET`` is used as written: a caller that names a budget (the
    probe runner, a bounded experiment) has already decided what it can afford.
    Otherwise the job's budget (``TASK_TIMEOUT``, also enforced by gpu_entry's
    ``timeout``) decides, less a reserve. That reserve is the point: build and device
    time share the job's budget, and the shell timeout used to win the race, so the
    adapter died on SIGTERM before it could say what the job had spent.
    """
    explicit = str(os.environ.get(BUDGET_ENV) or "").strip()
    if explicit:
        try:
            value = int(float(explicit))
        except ValueError:
            raise RuntimeError(f"{BUDGET_ENV}={explicit!r} is not a number of seconds")
        return value if value > 0 else None
    raw = str(os.environ.get("TASK_TIMEOUT") or "").strip()
    try:
        budget = int(float(raw))
    except ValueError:
        return None
    if budget <= 0:
        return None
    return max(60, budget - min(120, max(30, budget // 20)))


def _case_window(case: dict, basis: str) -> tuple[object, str]:
    """One case's declared timing window, or a measurement diagnostic."""
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
    # A nonzero evaluator exit is a failed run, not a clean measurement-unavailable
    # reading: leave the ordinary correctness-failure path in charge.
    result.pop("measurement", None)


def _latest_report(root: Path) -> Path | None:
    candidates = [p for p in root.glob("*_eval_*.json") if p.is_file()
                  and not p.name.startswith("forge_")]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _end_group(proc: subprocess.Popen, grace_s: float = TERM_GRACE_S) -> None:
    """SIGTERM the evaluator's process group, then SIGKILL what is left.

    kernel_eval spawns worker processes per card; signalling the parent alone leaves
    them holding the NPU, so the group is the unit. Its own session (below) is what
    makes the group ours to end.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def _run_logged(cmd: list[str], *, cwd: Path, env: dict, log: Path,
                budget_s: int | None = None) -> RunOutcome:
    """Run the evaluator in its own session, logging every line it prints.

    The adapter owns stopping it: at ``budget_s`` (or on a stop signal to this
    process) the whole group is terminated and the log is kept. That is the
    difference between "the job timed out" and knowing what it spent.
    """
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                start_new_session=True)

        def pump() -> None:  # the evaluator is chatty for hours; nothing may block it
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                fh.write(line)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        interrupted, reason = False, ""
        try:
            rc = proc.wait(timeout=budget_s)
        except subprocess.TimeoutExpired:
            interrupted = True
            reason = f"the evaluator exceeded its {budget_s}s budget"
            _end_group(proc)
            rc = proc.returncode
        except (Interrupted, KeyboardInterrupt) as exc:
            interrupted = True
            reason = str(exc) or "the adapter received a stop signal"
            _end_group(proc)
            rc = proc.returncode
        reader.join(timeout=TERM_GRACE_S)
        if interrupted:
            fh.write(f"\n[cann-adapter] evaluator stopped: {reason}\n")
            fh.flush()
            print(f"[cann-adapter] evaluator stopped: {reason}", flush=True)
        seconds = time.monotonic() - started
    return RunOutcome(rc=0 if rc is None else rc, interrupted=interrupted,
                      reason=reason, seconds=seconds)


def _install_stop_handlers() -> dict[int, object]:
    """Turn a stop signal into ``Interrupted`` so the partial evidence is written.

    Returns the handlers that were in place, for the caller to restore: this process
    is the adapter's own, and a test or an embedding parent must not inherit them.
    """
    def stop(signum: int, _frame) -> None:
        raise Interrupted(f"the adapter was stopped by signal {signum}")

    previous: dict[int, object] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, stop)
    return previous


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
    # Resolve the routing once: the config the report converter sees, the backend
    # collection labels, and the preflight check are the same values.
    config = {**config, "profiler_backend": resolve_backend(config),
              "timing_basis": resolve_timing_basis(config)}
    backend = config["profiler_backend"]
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
    selection = case_selection(config, os.environ.get(CASE_ENV))
    task_dir = str(config["task_dir"])
    scout_root = None
    if selection.scope:
        task_dir, scout_root = scoped_task_dir(cann_dir, config, selection.names,
                                               _scout_workdir(cann_dir))
        task_dir = str(task_dir)
    cmd = [
        python, "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann",
        "--source-dir", str(source),
        "--task-dir", task_dir,
        "--operator", config["operator"],
        "--device-id", "0",
        "--processes-per-card", "1",
        "--timeout-per-operator", timeout,
        "--warmup", str(config.get("warmup", 3)),
        "--repeat", str(config.get("repeat", 5)),
        "--perf-metric-strategy", config.get("perf_metric_strategy", "trace_view"),
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
    if selection.scope:
        print(f"[cann-adapter] scope: {len(selection.names)} of "
              f"{selection.scope['of']} cases ({selection.scope['workloads']}) via an "
              f"isolated official task root at {task_dir}", flush=True)
    budget = evaluator_budget()
    try:
        outcome = _run_logged(cmd, cwd=cann_dir, env=env,
                              log=reports / "evaluator.log", budget_s=budget)
    finally:
        if scout_root is not None:
            shutil.rmtree(scout_root, ignore_errors=True)
            if scout_root.parent.name == ".forgekernel-scout":
                try:  # leave the shared bench root clean; another job may hold it
                    scout_root.parent.rmdir()
                except OSError:
                    pass
    # A job's wall clock is the only cost reading that is always available, and
    # without it a compile-bound campaign and a slow kernel are indistinguishable
    # from the outside: the GQA attempt lost outer-loop budget to JIT compiles that
    # left no trace anywhere a diagnosis could read. Written whatever the outcome,
    # so a timed-out or crashed job still says where its time went.
    cost = {"evaluator_seconds": round(outcome.seconds, 1), "budget_seconds": budget,
            "cases_requested": len(selection.names) or "all",
            "interrupted": outcome.interrupted,
            "note": "wall clock, build and device time together; a per-phase split "
                    "needs a compile hook certified on hardware"}
    if outcome.interrupted:
        cost["interrupt_reason"] = outcome.reason
        return _interrupted_eval(reports, outcome, cost, selection, config, backend)
    report_path = _latest_report(reports)
    if report_path is None:
        tail = (reports / "evaluator.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        emit_artifact("cann/evaluator.log", (reports / "evaluator.log").read_bytes()[-1048576:])
        result = {"correctness": {"pass": False,
                                  "error": f"CANN-Bench exited {outcome.rc} without a JSON report: {tail}"},
                  "performance": {},
                  "cann_bench": {"cost": cost, "case_scope": selection.scope},
                  "profiler": {"backend": backend, "timing_basis": config["timing_basis"],
                               "status": "unavailable"}}
        if selection.scope:
            result["workload_scope"] = selection.scope
        return result
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if selection.scope and (problems := scope_violations(report, selection.names)):
        # The isolated root is the only thing narrowing this job; a report over any
        # other case set means the mechanism did not hold. Fail closed rather than
        # price anything against a scope that did not happen.
        raise RuntimeError("narrowed job evaluated the wrong case set: "
                           + "; ".join(problems))
    result = report_to_eval(report, config=config)
    artifacts = [emit_artifact("cann/official_report.json", raw),
                 emit_artifact("cann/evaluator.log",
                               (reports / "evaluator.log").read_bytes()[-1048576:])]
    if outcome.rc:
        evaluator_failure(result, outcome.rc)
    if backend == "msprof":
        # Collection adds the exported tables / session records but must not replace
        # the report's own metadata: the timing basis (and the trace_view cases) are
        # what make the reading interpretable.
        evidence, prof_artifacts = collect(reports, backend)
        result["profiler"] = {**(result.get("profiler") or {}), **evidence,
                              "backend": backend,
                              "timing_basis": config["timing_basis"]}
        artifacts.extend(prof_artifacts)
    if selection.scope:
        # A subset is a pricing reading and must never clear a gate. loop/parse.py
        # refuses any eval JSON carrying workload_scope, so this is the existing
        # guard rather than a second rule that could drift from it.
        result["workload_scope"] = selection.scope
    result["cann_bench"].update({"report": str(report_path), "evaluator_rc": outcome.rc,
                                  "commit": actual_ref, "artifacts": artifacts,
                                  "cost": cost, "case_scope": selection.scope})
    (reports / "forge_eval.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    return result


def _interrupted_eval(reports: Path, outcome: RunOutcome, cost: dict,
                      selection: CaseSelection, config: dict, backend: str) -> dict:
    """What a stopped evaluator leaves behind: cost, logs, scope -- and no claim.

    Neither correctness nor performance is invented here. A partial official report is
    kept as evidence (it may name cases that already ran), never read as a verdict.
    """
    artifacts = [emit_artifact("cann/evaluator.log",
                               (reports / "evaluator.log").read_bytes()[-1048576:])]
    partial = _latest_report(reports)
    if partial is not None:
        artifacts.append(emit_artifact("cann/partial_report.json", partial.read_bytes()))
    result = {
        "correctness": {
            "pass": False,
            "error": (f"the adapter stopped the evaluator after {cost['evaluator_seconds']}s "
                      f"(budget {cost['budget_seconds']}s): {outcome.reason}; no complete "
                      f"official report exists, so this job has no correctness or "
                      f"performance reading"),
        },
        "performance": {},
        "cann_bench": {"report": str(partial) if partial else None,
                       "evaluator_rc": outcome.rc, "artifacts": artifacts,
                       "cost": cost, "case_scope": selection.scope},
        "profiler": {"backend": backend, "timing_basis": config["timing_basis"],
                     "status": "unavailable"},
    }
    if selection.scope:
        result["workload_scope"] = selection.scope
    (reports / "forge_eval.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        previous = _install_stop_handlers()
    except ValueError:  # not the main thread: the embedding caller owns the signals
        previous = {}
    try:
        if args.preflight:
            config = json.loads((Path(args.workspace) / "eval/cann_task.json").read_text())
            backend = resolve_backend(config)
            basis = resolve_timing_basis(config)
            if backend == "msprof" and not shutil.which("msprof"):
                raise RuntimeError("msprof executable unavailable")
            import pypto
            import torch
            import torch_npu
            value = torch.ones(4, device="npu").sum().item()
            if value != 4:
                raise RuntimeError("NPU smoke failed")
            print(f"ASCEND-PREFLIGHT-OK pypto={pypto.__version__} "
                  f"torch_npu={torch_npu.__version__} "
                  f"backend={backend} timing_basis={basis}")
            return 0
        result = evaluate(Path(args.workspace).resolve())
    except Interrupted as exc:
        print(f"[cann-adapter] stopped before the evaluator ran: {exc}", file=sys.stderr)
        return 137
    except Exception as exc:  # harness configuration failure: visible in the job log
        print(f"[cann-adapter] harness error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print("=== JSON ===")
    print(json.dumps(result, separators=(",", ":")))
    # A wrong candidate is a valid measurement outcome, not a transport failure. Returning
    # zero lets ForgeKernel record REVERT with the official error instead of losing the log.
    # A job this adapter stopped is the one exception: nothing was judged, so it keeps the
    # timeout exit status while its cost and scope stay in the log above.
    return 137 if (result.get("cann_bench") or {}).get("cost", {}).get("interrupted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
