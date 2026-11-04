from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from forgekernel.baseline import BASIS_CANN_TRACE_VIEW, HOST_FREE_BASES
from forgekernel.loop import lang_mode
from forgekernel.paths import repo_root
from forgekernel.submit.config import SubmitConfig
from forgekernel.submit.direct import _job_env
from forgekernel.task import load_task


def _adapter():
    path = repo_root() / "src/forgekernel/card_template/eval/cann_bench.py"
    spec = importlib.util.spec_from_file_location("cann_bench_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(passed: int = 2) -> dict:
    cases = []
    for i, elapsed in enumerate((100.0, 400.0), 1):
        cases.append({"case_id": f"level1/swi_glu_{i}", "status": "success",
                      "elapsed_us": elapsed,
                      "accuracy": {"passed": i <= passed, "error_msg": "bad" if i > passed else ""},
                      "op_times": {"trace_view": {"aicore_e2e": elapsed,
                                                 "aicore_e2e_jitter": 1.0}}})
    return {"tasks_version": "2.0.0", "operators": [{
        "operator": "SwiGlu", "total_cases": 2, "passed_cases": passed, "cases": cases}]}


def test_cann_report_becomes_geomean_eval():
    adapter = _adapter()
    result = adapter.report_to_eval(_report())
    assert result["correctness"]["pass"] is True
    perf = result["performance"]
    assert perf["n_workloads"] == 2
    assert perf["bwd_median_ms"] == perf["device_median_ms"]
    assert math.isclose(perf["device_median_ms"], 0.2)
    assert perf["device_window_basis"] == adapter.BASIS == BASIS_CANN_TRACE_VIEW
    assert BASIS_CANN_TRACE_VIEW in HOST_FREE_BASES
    assert "ref_device_ms" not in perf


def test_cann_accuracy_failure_has_no_timing():
    adapter = _adapter()
    result = adapter.report_to_eval(_report(passed=1))
    assert result["correctness"]["pass"] is False
    assert result["performance"] == {}
    assert "swi_glu_2" in result["correctness"]["error"]


def test_pypto_language_kb_and_provenance_routing():
    assert lang_mode.mode_of("pypto") == lang_mode.MODE_PYPTO
    assert lang_mode.is_pypto("pypto")
    assert not lang_mode.is_cuda("pypto")
    assert lang_mode.kb_root_name("pypto") == "hw_kb/pypto"
    assert lang_mode.owns_provenance("docs/kb-provenance/pypto/x.md", "pypto")
    assert not lang_mode.owns_provenance("docs/kb-provenance/cuda/x.md", "pypto")
    assert not lang_mode.owns_provenance("docs/kb-provenance/pypto/x.md", "cutedsl")


def test_npu_job_env_does_not_set_cuda_visible_devices():
    cfg = SubmitConfig(branch="main", exec_gpu="1", env={"FK_DEVICE_TYPE": "npu"})
    env = _job_env(cfg)
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_swiglu_pypto_task_loads():
    task = load_task(repo_root() / "tasks/swiglu-pypto")
    assert task.language == "pypto"
    assert task.loop_language == "pypto"
    assert task.arch == "ascend950"
    assert task.leaderboard.suite == "cann-bench"
    assert len(task.shapes) == 20


def test_consolidated_candidate_has_all_entrypoints():
    text = (repo_root() / "tasks/swiglu-pypto/leaderboard/card/kernel.py").read_text()
    compile(text, "kernel.py", "exec")
    for i in range(1, 9):
        assert f"Public reference class c{i}" in text
    assert "(5, \"float32\"): swi_glu_wrapper_c8" in text
