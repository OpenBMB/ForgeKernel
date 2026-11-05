"""The PyPTO backend's two configuration files must describe one measurement.

Each test here stands for a way the backend could report a number nobody can
trace, or aim an iteration at a kernel that is not at fault.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from forgekernel import profiling
from forgekernel.loop.parse import parse_eval
from forgekernel.paths import repo_root


def _adapter():
    path = repo_root() / "src/forgekernel/card_template/eval/cann_bench.py"
    spec = importlib.util.spec_from_file_location("cann_bench_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(*, trace_view: bool = True, elapsed: float = 120.0) -> dict:
    case = {"case_id": "gqa_1", "status": "success", "elapsed_us": elapsed,
            "accuracy": {"passed": True},
            "op_times": {"trace_view": {"aicore_e2e": elapsed}} if trace_view else {}}
    return {"tasks_version": "2.0.0", "operators": [
        {"operator": "GQA", "total_cases": 1, "passed_cases": 1, "cases": [case]}]}


# --- the channel default that stopped the meta layer -------------------------------

def test_pypto_default_backend_admits_the_channels_its_prompts_instruct():
    """``auto`` must not resolve to a backend that rejects meta/pypto/'s own channels.

    ``cann_trace_view`` allows only ``cann-trace-view``, so a teacher following
    distill_prompt.md into ``[VERIFY: msprof-direct]`` had every recon round
    rejected until the meta layer stopped with no measurement at fault.
    """
    backend = profiling.backend_for("pypto", "auto")
    assert backend == "msprof"
    cited = "\n".join(f"finding [VERIFY: {tag}]" for tag in
                      ("correctness", "cann-trace-view", "msprof-direct",
                       "msprof-derived", "probe"))
    assert profiling.channel_violations(cited, backend) == []


def test_ncu_channels_stay_closed_on_ascend():
    assert profiling.channel_violations("[VERIFY: ncu-direct]", "msprof")
    assert profiling.channel_violations("[VERIFY: sass-derived]", "msprof")
    with pytest.raises(ValueError):
        profiling.backend_for("pypto", "ncu")


# --- timing basis and collection backend are two axes ------------------------------

def test_timing_basis_must_match_the_strategy_that_produces_it():
    assert profiling.timing_basis_problems(
        {"timing_basis": "cann_kernel_details", "perf_metric_strategy": "kernel_details"}) == []
    assert profiling.timing_basis_problems(
        {"timing_basis": "cann_trace_view", "perf_metric_strategy": "trace_view"}) == []
    # The pairing that invalidated the first GQA M0 gate.
    problems = profiling.timing_basis_problems(
        {"timing_basis": "cann_trace_view", "perf_metric_strategy": "kernel_details"})
    assert problems and "perf_metric_strategy" in problems[0]
    assert profiling.timing_basis_problems({"timing_basis": "msprof_task_rows"})


def test_control_and_execution_backends_are_cross_checked_both_ways():
    # The direction that passed every previous check: gate lines admitted for a
    # channel the job never collects.
    assert profiling.card_backend_problem({"profiler_backend": "cann_trace_view"}, "msprof")
    # And the direction that rejected evidence the job did collect.
    assert profiling.card_backend_problem({"profiler_backend": "msprof"}, "cann_trace_view")
    assert profiling.card_backend_problem({"profiler_backend": "msprof"}, "msprof") is None


def test_every_shipped_pypto_card_declares_one_consistent_measurement():
    cards = sorted(Path(repo_root() / "tasks").glob("*/leaderboard/*/eval/cann_task.json"))
    assert cards, "no PyPTO card found to check"
    for path in cards:
        config = json.loads(path.read_text())
        loop_backend = profiling.backend_for("pypto", config["profiler_backend"])
        assert profiling.timing_basis_problems(config) == [], path
        assert profiling.card_backend_problem(config, loop_backend) is None, path


def test_candidate_and_reference_tree_share_one_window():
    for task in sorted(Path(repo_root() / "tasks").glob("*-pypto*")):
        bases = {t.parent.parent.name: json.loads(t.read_text()).get("timing_basis")
                 for t in sorted(task.glob("leaderboard/*/eval/cann_task.json"))}
        assert len(set(bases.values())) <= 1, f"{task.name}: {bases}"


# --- a configuration fault is not a kernel fault -----------------------------------

def test_missing_declared_window_is_a_measurement_fault_not_a_correctness_one():
    """20/20 on the official judge plus a missing window is not a wrong kernel.

    Reported as ``correctness: False``, the next iteration is aimed at a kernel
    that passed every case, and the configuration that actually broke stays put.
    """
    result = _adapter().report_to_eval(_report(trace_view=False), config={
        "timing_basis": "cann_trace_view", "perf_metric_strategy": "kernel_details"})
    assert result["correctness"]["pass"] is True
    assert result["correctness"]["error"] == ""
    assert result["performance"] == {}
    assert result["measurement"]["status"] == "unavailable"
    assert "trace_view" in result["measurement"]["reason"]

    ev = parse_eval("=== JSON ===\n" + json.dumps(result))
    assert ev.measurement["status"] == "unavailable"
    assert ev.correctness is True
    assert ev.device_median_ms is None


def test_a_real_accuracy_failure_is_still_a_correctness_failure():
    report = _report()
    report["operators"][0]["passed_cases"] = 0
    report["operators"][0]["cases"][0]["accuracy"] = {"passed": False, "error_msg": "max diff"}
    result = _adapter().report_to_eval(report, config={
        "timing_basis": "cann_trace_view", "perf_metric_strategy": "trace_view"})
    assert result["correctness"]["pass"] is False
    assert "max diff" in result["correctness"]["error"]
    assert "measurement" not in result


def test_profiler_backend_reports_the_tool_not_the_window():
    """A card collects msprof and still times on trace_view; one name cannot carry both."""
    result = _adapter().report_to_eval(_report(), config={
        "profiler_backend": "msprof", "timing_basis": "cann_trace_view",
        "perf_metric_strategy": "trace_view"})
    assert result["profiler"]["backend"] == "msprof"
    assert result["profiler"]["timing_basis"] == "cann_trace_view"
    assert result["performance"]["device_window_basis"] == "cann_trace_view"
    assert result["cann_bench"]["timing_basis"] == "cann_trace_view"
