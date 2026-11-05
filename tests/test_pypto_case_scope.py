"""A narrowed CANN-Bench job must be cheap, honest about its scope, and never a gate.

Shape specialization makes every case its own JIT compile, so a 20-case job costs
20 compiles whatever the question was. Narrowing is the lever that made GQA
iterable; these tests are what keep it from also becoming a way to claim a win on
four cases out of twenty.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from forgekernel.loop.parse import parse_eval
from forgekernel.paths import repo_root


def _adapter():
    path = repo_root() / "src/forgekernel/card_template/eval/cann_bench.py"
    spec = importlib.util.spec_from_file_location("cann_bench_scope", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = {"cases": [f"gqa_{i}" for i in range(1, 21)], "case_filter_flag": "--case-ids"}


def test_no_request_runs_the_whole_set_and_declares_no_scope():
    scope = _adapter().case_scope(CONFIG, None)
    assert scope == {"argv": [], "n": 0, "scope": None}


def test_a_subset_becomes_evaluator_argv_and_a_scope_declaration():
    scope = _adapter().case_scope(CONFIG, "gqa_1, gqa_8 ,gqa_11")
    assert scope["argv"] == ["--case-ids", "gqa_1,gqa_8,gqa_11"]
    assert scope["scope"]["workloads"] == 3
    assert scope["scope"]["of"] == 20
    assert scope["scope"]["rep"] == "scout"


def test_a_subset_can_never_clear_a_gate():
    """The scope rides the eval JSON into loop/parse.py's existing refusal.

    One guard, not two: a second rule here could drift from the one the CuTe DSL
    group cards already rely on.
    """
    adapter = _adapter()
    result = {"correctness": {"pass": True, "passed_cases": 3, "total_cases": 3},
              "performance": {"device_median_ms": 0.1, "bwd_median_ms": 0.1},
              "workload_scope": adapter.case_scope(CONFIG, "gqa_1,gqa_8,gqa_11")["scope"]}
    ev = parse_eval("=== JSON ===\n" + json.dumps(result))
    assert ev.correctness is False
    assert "workload_scope" in ev.error
    assert ev.device_median_ms is None


def test_a_narrowed_job_is_refused_rather_than_silently_run_whole():
    """Without a declared filter flag the adapter cannot narrow anything.

    Running all 20 cases under a name that says 4 would make a scout job's cost
    reading a fiction, and this adapter does not guess a vendor CLI contract.
    """
    with pytest.raises(RuntimeError, match="case_filter_flag"):
        _adapter().case_scope({"cases": CONFIG["cases"]}, "gqa_1")


def test_unknown_case_ids_are_refused():
    with pytest.raises(RuntimeError, match="unknown case"):
        _adapter().case_scope(CONFIG, "gqa_1,gqa_99")


def test_gqa_scout_set_covers_every_regime_and_head_dimension():
    """Four cases, four compiles, and no structural family left unmeasured.

    A scout set that misses a regime prices an architecture decision against
    shapes it does not apply to, which is worse than paying for the full gate.
    """
    task = Path(repo_root() / "tasks/gqa-pypto")
    config = json.loads((task / "leaderboard/card/eval/cann_task.json").read_text())
    scout = config["scout_cases"]
    assert set(scout) <= set(config["cases"])
    problem = json.loads((task / "leaderboard/card/eval/leaderboard/problem.json").read_text())
    axes = {r["name"].split("/")[-1]: r["workload"]["axes"]["input_shape"]
            for r in problem["group"]}

    def family(case: str) -> tuple[str, int]:
        (_, s_q, _, d), _, _ = axes[case]
        return ("decode" if s_q == 1 else "mtp" if s_q <= 8 else "prefill"), d

    covered = {family(c) for c in scout}
    every = {family(c) for c in config["cases"]}
    missing = {f for f in every if f[0] not in {c[0] for c in covered}
               or f[1] not in {c[1] for c in covered}}
    assert not missing, f"scout set misses {missing}"
    assert len(scout) <= 6, "a scout set this large is not cheaper than the gate"


def test_cost_is_recorded_so_a_compile_bound_campaign_is_visible():
    from forgekernel.profiling import benchmark_digest
    cost = {"evaluator_seconds": 3400.0, "budget_seconds": 7140, "cases_requested": 20}
    digest = benchmark_digest({"cost": cost, "case_scope": None, "cases": []})
    assert digest["cost"]["evaluator_seconds"] == 3400.0
    assert digest["cost"]["budget_seconds"] == 7140
