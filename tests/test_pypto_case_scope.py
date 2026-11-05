"""A narrowed CANN-Bench job must be cheap, honest about its scope, and never a gate.

Shape specialization makes every case its own JIT compile, so a 20-case job costs
20 compiles whatever the question was. Narrowing is the lever that made GQA
iterable; these tests are what keep it from also becoming a way to claim a win on
five cases out of twenty.

The pinned CANN-Bench has no case-set flag (``--case-id`` takes one int), so the
subset is an isolated official task root. The tests below drive that path with a
stand-in ``kernel_eval.cli`` that reads the ``cases.yaml`` it is handed: they fail
if the adapter ever runs more cases than were selected, if it stops keeping the
official goldens/baselines, or if a scoped reading can reach a gate.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from forgekernel.loop.parse import parse_eval
from forgekernel.paths import repo_root

CASES = [f"gqa_{i}" for i in range(1, 21)]
SCOUT = ["gqa_1", "gqa_5", "gqa_8", "gqa_11", "gqa_16"]

# A stand-in for the pinned kernel_eval.cli: it records the task root and the case set
# it was asked to evaluate, then writes a report shaped like the official one. The
# adapter only ever sees argv and a report, so this is the real seam.
FAKE_CLI = '''\
import argparse, json, os, sys
from pathlib import Path
import yaml

p = argparse.ArgumentParser()
p.add_argument("command", nargs="?")
for flag in ("--bench-name", "--source-dir", "--task-dir", "--operator", "--device-id",
             "--processes-per-card", "--timeout-per-operator", "--warmup", "--repeat",
             "--perf-metric-strategy", "--reports-dir", "--case-id"):
    p.add_argument(flag)
a = p.parse_args()
task_dir = Path(a.task_dir).resolve()
bench_root = next(x for x in (task_dir, *task_dir.parents) if x.name == "tasks")
rel = task_dir.relative_to(bench_root).as_posix()
doc = yaml.safe_load((task_dir / "cases.yaml").read_text())
cases = []
for case in doc["cases"]:
    number = case["case_id"]
    cases.append({"case_id": f"{rel}_{number}", "status": "success",
                  "elapsed_us": 100.0 + number, "accuracy": {"passed": True},
                  "op_times": {"trace_view": {"aicore_e2e": 100.0 + number}}})
if os.environ.get("FAKE_EVAL_EXTRA"):
    cases.append({"case_id": f"{rel}_2", "status": "success", "elapsed_us": 102.0,
                  "accuracy": {"passed": True},
                  "op_times": {"trace_view": {"aicore_e2e": 102.0}}})
if os.environ.get("FAKE_EVAL_HANG"):
    print("compiling a case that never finishes...", flush=True)
    if os.environ.get("FAKE_EVAL_GRANDCHILD"):
        child = __import__("subprocess").Popen([sys.executable, "-c",
                                               "import time; time.sleep(600)"])
        Path(os.environ["FAKE_EVAL_GRANDCHILD"]).write_text(str(child.pid))
    import time as _t
    _t.sleep(600)
report = {"tasks_version": "2.0.0", "framework_version": "2.0.0",
          "setup_info": {"environment": {}},
          "operators": [{"operator": a.operator, "total_cases": len(cases),
                         "passed_cases": len(cases), "cases": cases}]}
out = Path(a.reports_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "cann_eval_report.json").write_text(json.dumps(report))
Path(os.environ["FAKE_EVAL_LOG"]).write_text(json.dumps(
    {"argv": sys.argv[1:], "task_dir": str(task_dir),
     "cases": [c["case_id"] for c in cases]}))
'''


def _adapter():
    path = repo_root() / "src/forgekernel/card_template/eval/cann_bench.py"
    spec = importlib.util.spec_from_file_location("cann_bench_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = {"operator": "GQA", "task_dir": "tasks/level4/gqa", "cases": CASES,
          "scout_cases": SCOUT, "profiler_backend": "msprof",
          "perf_metric_strategy": "trace_view", "timing_basis": "cann_trace_view"}


# --- which cases a job asks for ----------------------------------------------------

def test_no_request_runs_the_whole_set_and_declares_no_scope():
    selection = _adapter().case_selection(CONFIG, None)
    assert selection == _adapter().CaseSelection([], "full", None)


@pytest.mark.parametrize("requested", ["all", "full", "gate", "ref", "reference",
                                       "confirm", "ALL", " all "])
def test_every_full_set_spelling_is_the_whole_declared_set(requested):
    """A KEEP is read off the full set; no alias may quietly narrow it."""
    assert _adapter().case_selection(CONFIG, requested).scope is None


def test_the_scout_alias_resolves_to_the_card_declared_representatives():
    selection = _adapter().case_selection(CONFIG, "scout")
    assert selection.names == SCOUT
    assert selection.alias == "scout"
    assert selection.scope["of"] == 20
    assert selection.scope["workloads"] == ",".join(SCOUT)
    assert selection.scope["rep"] == SCOUT[0]


def test_explicit_ids_are_a_subset_too():
    selection = _adapter().case_selection(CONFIG, "gqa_8")
    assert selection.names == ["gqa_8"] and selection.scope["of"] == 20


def test_naming_the_whole_set_is_the_gate_not_a_subset():
    assert _adapter().case_selection(CONFIG, ",".join(reversed(CASES))).scope is None


def test_a_typo_or_a_duplicate_is_refused_rather_than_run_whole():
    with pytest.raises(RuntimeError, match="unknown case"):
        _adapter().case_selection(CONFIG, "gqa_1,gqa_99")
    with pytest.raises(RuntimeError, match="duplicate"):
        _adapter().case_selection(CONFIG, "gqa_1,gqa_1")
    with pytest.raises(RuntimeError, match="names no case id"):
        _adapter().case_selection(CONFIG, " , ")


def test_a_card_without_a_scout_set_is_refused_not_guessed():
    with pytest.raises(RuntimeError, match="scout_cases"):
        _adapter().case_selection({"cases": CASES}, "scout")


# --- the isolated official task root ----------------------------------------------

def _official_task(path: Path, case_numbers: range = range(1, 21)) -> Path:
    """An official-looking bench tree: tasks/level4/gqa + tasks/metadata/910b2.json."""
    op = path / "tasks/level4/gqa"
    op.mkdir(parents=True)
    (op / "proto.yaml").write_text("operator:\n  name: GQA\n  difficulty: L4\n")
    (op / "golden.py").write_text("# official golden\n")
    (op / "cases.csv").write_text("operator,case_id\nGQA,1\n")
    (op / "cases.yaml").write_text(yaml.safe_dump({"cases": [
        {"operator": "GQA", "case_id": n,
         "input_shape": [[2, 128, 32, 128], [2, 128, 8, 128], [2, 128, 8, 128]],
         "dtype": ["float16"] * 3, "attrs": {"is_causal": True},
         "value_range": [-1, 1], "note": f"case {n} — official"} for n in case_numbers]},
        sort_keys=False, allow_unicode=True))
    metadata = path / "tasks/metadata"
    metadata.mkdir(parents=True)
    (metadata / "910b2.json").write_text(json.dumps({"level4": {"gqa": {
        str(n): {"baseline_perf_us": 100.0 + n, "t_hw_us": 10.0 + n}
        for n in case_numbers}}}))
    (path / "tasks/suite_manifest.json").write_text(json.dumps({"suite": "cann"}))
    return op


def test_the_isolated_root_holds_exactly_the_selected_official_cases(tmp_path):
    """Same ids, same inputs, same goldens, same published baselines -- 5 files' worth."""
    cann = tmp_path / "cann"
    op = _official_task(cann)
    adapter = _adapter()
    scout_op, root = adapter.scoped_task_dir(cann, CONFIG, SCOUT, cann)
    try:
        assert root.parent == cann
        assert scout_op.relative_to(root).as_posix() == "tasks/level4/gqa"
        filtered = yaml.safe_load((scout_op / "cases.yaml").read_text())
        assert [c["case_id"] for c in filtered["cases"]] == [1, 5, 8, 11, 16]
        official = yaml.safe_load((op / "cases.yaml").read_text())["cases"]
        selected = {c["case_id"]: c for c in official if c["case_id"] in (1, 5, 8, 11, 16)}
        for case in filtered["cases"]:
            assert case == selected[case["case_id"]]
        # Every other official task file is the official file, not a copy of it.
        for name in ("proto.yaml", "golden.py", "cases.csv"):
            assert (scout_op / name).is_symlink()
            assert (scout_op / name).resolve() == (op / name).resolve()
        # BaselineStore reads metadata/ from the bench root upward and takes
        # suite_manifest.json beside it: both must be the official ones.
        assert (root / "tasks/metadata").resolve() == (tmp_path / "cann/tasks/metadata")
        assert (root / "tasks/suite_manifest.json").is_symlink()
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def test_a_case_the_official_task_does_not_have_is_refused(tmp_path):
    cann = tmp_path / "cann"
    _official_task(cann, range(1, 3))
    with pytest.raises(RuntimeError, match="does not hold"):
        _adapter().scoped_task_dir(cann, CONFIG, ["gqa_5"], cann)


# --- real execution through a stand-in evaluator ----------------------------------

@pytest.fixture()
def cann_tree(tmp_path):
    """A CANN-Bench worktree: git checkout + official task + a stand-in kernel_eval."""
    root = tmp_path / "cann"
    _official_task(root)
    package = root / "src/kernel_eval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(FAKE_CLI)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qm", "cann"], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    return root, head


def _workspace(tmp_path, head):
    workspace = tmp_path / "workspace"
    (workspace / "eval").mkdir(parents=True)
    (workspace / "eval/cann_task.json").write_text(json.dumps({
        "operator": "GQA", "task_dir": "tasks/level4/gqa", "cases": CASES,
        "scout_cases": SCOUT, "perf_metric_strategy": "trace_view",
        "timing_basis": "cann_trace_view", "expected_cann_ref": head,
        "cases_requested": CASES}))
    return workspace


def _fake_python(tmp_path) -> Path:
    """A python that answers the adapter's `pip uninstall` itself.

    The test must not uninstall anything from the interpreter running the suite; every
    other invocation is forwarded to it unchanged.
    """
    wrapper = tmp_path / "fk-python"
    wrapper.write_text("#!/bin/bash\n"
                       "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then exit 0; fi\n"
                       f"exec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    return wrapper


def _run_adapter(tmp_path, cann_tree, monkeypatch, *, cases=None, extra_env=None):
    root, head = cann_tree
    workspace = _workspace(tmp_path, head)
    record = tmp_path / "evaluator.json"
    monkeypatch.setenv("CANN_BENCH_DIR", str(root))
    monkeypatch.setenv("FK_ASCEND_PYTHON", str(_fake_python(tmp_path)))
    monkeypatch.setenv("FAKE_EVAL_LOG", str(record))
    monkeypatch.delenv("FK_CANN_CASES", raising=False)
    monkeypatch.delenv("FK_CANN_EVAL_BUDGET", raising=False)
    if cases is not None:
        monkeypatch.setenv("FK_CANN_CASES", cases)
    for key, value in (extra_env or {}).items():
        monkeypatch.setenv(key, value)
    adapter = _adapter()
    result = adapter.evaluate(workspace)
    return adapter, result, json.loads(record.read_text()), workspace


def test_a_scout_job_runs_exactly_the_selected_cases(tmp_path, cann_tree, monkeypatch):
    """One evaluator invocation, five cases, and the isolation is gone afterwards."""
    _, result, record, workspace = _run_adapter(tmp_path, cann_tree, monkeypatch,
                                                cases="scout")
    assert record["cases"] == ["level4/gqa_1", "level4/gqa_5", "level4/gqa_8",
                              "level4/gqa_11", "level4/gqa_16"]
    assert record["task_dir"].startswith(str(cann_tree[0]))
    assert record["argv"][record["argv"].index("--task-dir") + 1] != "tasks/level4/gqa"
    # aggregation is over the selected cases only
    assert result["performance"]["n_workloads"] == 5
    assert result["correctness"]["pass"] is True
    assert result["workload_scope"]["of"] == 20
    assert result["workload_scope"]["workloads"] == ",".join(SCOUT)
    assert result["cann_bench"]["cost"]["cases_requested"] == 5
    assert result["cann_bench"]["cost"]["interrupted"] is False
    # the isolated root is a scratch directory, not a new bench
    assert not (cann_tree[0] / ".forgekernel-scout").exists() or not any(
        (cann_tree[0] / ".forgekernel-scout").iterdir())
    assert json.loads(next((workspace / ".cann_results").glob("eval-*/forge_eval.json"))
                      .read_text())["workload_scope"]["of"] == 20


def test_a_scout_result_can_never_clear_a_gate(tmp_path, cann_tree, monkeypatch):
    """The scope rides the eval JSON into loop/parse.py's existing refusal.

    One guard, not two: a second rule in the adapter could drift from the one the
    CuTe DSL group cards already rely on.
    """
    _, result, _, _ = _run_adapter(tmp_path, cann_tree, monkeypatch, cases="scout")
    ev = parse_eval("=== JSON ===\n" + json.dumps(result))
    assert ev.correctness is False
    assert "workload_scope" in ev.error
    assert ev.device_median_ms is None


def test_a_full_job_keeps_the_official_task_root_and_a_gate_reading(tmp_path, cann_tree,
                                                                   monkeypatch):
    _, result, record, _ = _run_adapter(tmp_path, cann_tree, monkeypatch)
    assert record["cases"] == [f"level4/gqa_{n}" for n in range(1, 21)]
    assert record["task_dir"] == str((cann_tree[0] / "tasks/level4/gqa").resolve())
    assert result["performance"]["n_workloads"] == 20
    assert "workload_scope" not in result
    assert result["cann_bench"]["cost"]["cases_requested"] == "all"
    assert parse_eval("=== JSON ===\n" + json.dumps(result)).correctness is True


def test_a_narrowed_job_that_evaluates_more_cases_fails_closed(tmp_path, cann_tree,
                                                               monkeypatch):
    """A report over any other case set means the mechanism did not hold."""
    with pytest.raises(RuntimeError, match="wrong case set"):
        _run_adapter(tmp_path, cann_tree, monkeypatch, cases="gqa_1,gqa_5",
                     extra_env={"FAKE_EVAL_EXTRA": "1"})


# --- a stalled evaluator keeps its evidence ---------------------------------------

def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc to reap-check")
def test_a_stalled_evaluator_is_stopped_with_its_whole_process_group(tmp_path):
    """The adapter owns the timeout: the group dies and its log survives.

    gpu_entry's shell timeout used to be the only stop, and its SIGTERM landed on an
    adapter that had not computed anything yet -- a forty-minute compile-bound probe
    came back with no cost at all. This is the real subprocess test of the
    replacement: a grandchild (what kernel_eval's per-case workers are) must not
    outlive the job.
    """
    adapter = _adapter()
    log = tmp_path / "evaluator.log"
    child_pid = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"open(r'{child_pid}', 'w').write(str(child.pid))\n"
        "print('building case 1...', flush=True)\n"
        "time.sleep(600)\n")
    started = time.monotonic()
    outcome = adapter._run_logged([sys.executable, "-c", script], cwd=tmp_path,
                                  env={**os.environ}, log=log, budget_s=2)
    elapsed = time.monotonic() - started
    assert outcome.interrupted is True
    assert outcome.rc != 0 and 1 < elapsed < 60
    assert "building case 1..." in log.read_text()
    assert "evaluator stopped" in log.read_text()
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.1)
    assert not _alive(pid), "the evaluator's worker outlived the group kill"


def test_an_interrupted_job_still_reports_its_cost_and_scope(tmp_path, cann_tree,
                                                             monkeypatch, capsys):
    """No fabricated reading: cost, scope and logs -- and no correctness/performance."""
    root, head = cann_tree
    workspace = _workspace(tmp_path, head)
    monkeypatch.setenv("CANN_BENCH_DIR", str(root))
    monkeypatch.setenv("FK_ASCEND_PYTHON", str(_fake_python(tmp_path)))
    monkeypatch.setenv("FAKE_EVAL_LOG", str(tmp_path / "evaluator.json"))
    monkeypatch.setenv("FAKE_EVAL_HANG", "1")
    monkeypatch.setenv("FK_CANN_EVAL_BUDGET", "3")
    monkeypatch.setenv("FK_CANN_CASES", "scout")
    adapter = _adapter()
    assert adapter.main(["--workspace", str(workspace)]) == 137
    text = capsys.readouterr().out
    result = json.loads(text.split("=== JSON ===")[-1].strip())
    assert result["correctness"]["pass"] is False
    assert "adapter stopped the evaluator" in result["correctness"]["error"]
    assert result["performance"] == {}
    assert "measurement" not in result  # not the configuration-fault stop path
    cost = result["cann_bench"]["cost"]
    assert cost["interrupted"] is True and cost["cases_requested"] == 5
    assert cost["evaluator_seconds"] >= 2 and cost["budget_seconds"] == 3
    assert result["workload_scope"]["of"] == 20
    assert any(a["name"] == "cann/evaluator.log" for a in result["cann_bench"]["artifacts"])
    # ... and the log of the stopped job is in the job's own stdout, not only in a file
    assert "compiling a case that never finishes" in text


# --- cost is always visible -------------------------------------------------------

def test_cost_is_recorded_so_a_compile_bound_campaign_is_visible():
    from forgekernel.profiling import benchmark_digest
    cost = {"evaluator_seconds": 3400.0, "budget_seconds": 7140, "cases_requested": 20}
    digest = benchmark_digest({"cost": cost, "case_scope": None, "cases": []})
    assert digest["cost"]["evaluator_seconds"] == 3400.0
    assert digest["cost"]["budget_seconds"] == 7140


# --- the entry script's own seam ---------------------------------------------------

SHIPPED_CARDS = ["tasks/gqa-pypto/leaderboard/card", "tasks/gqa-pypto/leaderboard/ref",
                 "tasks/swiglu-pypto/leaderboard/card", "tasks/swiglu-pypto/leaderboard/ref",
                 "tasks/swiglu-pypto-g2/leaderboard/card",
                 "tasks/swiglu-pypto-g2/leaderboard/ref"]


@pytest.mark.parametrize("card", SHIPPED_CARDS + ["src/forgekernel/card_template"])
def test_gpu_entry_separates_a_scout_task_from_a_gate_task(card, tmp_path):
    """`scout` may narrow a job; `all` and the other gate tasks may not be narrowed.

    An ambient FK_CANN_CASES that reached a gate task would make the row a KEEP was
    read on a five-case subset -- the one failure this whole mechanism must not have.
    The probe path (`cmd:`) keeps the environment, and is how notes/ascend_probe.py
    reaches the adapter.
    """
    entry = repo_root() / card / ("gpu_entry.sh" if (repo_root() / card / "gpu_entry.sh")
                                  .is_file() else "pypto_gpu_entry.sh")
    fake = tmp_path / "fk-python"
    fake.write_text("#!/bin/bash\necho \"ADAPTER-SAW cases=[${FK_CANN_CASES:-unset}]\"\n")
    fake.chmod(0o755)
    env = {**os.environ, "CANN_BENCH_DIR": str(tmp_path), "FK_ASCEND_PYTHON": str(fake),
           "TASK_TIMEOUT": "30"}

    def run(*args, extra=None):
        return subprocess.run(["bash", str(entry), *args], cwd=entry.parent, env={**env,
                               **(extra or {})}, capture_output=True, text=True).stdout

    scout = run("scout", "gqa_1,gqa_8")
    assert "ADAPTER-SAW cases=[gqa_1,gqa_8]" in scout
    assert "not a gate" in scout
    assert "ADAPTER-SAW cases=[scout]" in run("scout")
    gate = run("all", extra={"FK_CANN_CASES": "scout"})
    assert "ADAPTER-SAW cases=[unset]" in gate
    assert "ignoring FK_CANN_CASES=scout" in gate
    probe = run("cmd", "true", extra={"FK_CANN_CASES": "scout"})
    assert "ADAPTER-SAW" not in probe  # the probe's own command saw it, not the entry
    assert "task=cmd finished" in probe
