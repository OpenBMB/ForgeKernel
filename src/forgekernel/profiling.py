"""Explicit profiler routing and evidence-channel validation."""
from __future__ import annotations

import re

BACKENDS = ("ncu", "msprof", "cann_trace_view", "none")
# The default PyPTO evidence channel. Kept equal to the card adapter's
# ascend_profiler.DEFAULT_PROFILER_BACKEND: the control plane validates the same
# routing the frozen card will execute (tests/test_ascend_evidence.py holds it).
PYPTO_DEFAULT_BACKEND = "msprof"
ASCEND_BACKENDS = (PYPTO_DEFAULT_BACKEND, "cann_trace_view")

# A timing basis is a window, not a collection tool. Each basis is readable only
# from the official report produced by its matching CANN-Bench strategy.
TIMING_BASES = {
    "cann_trace_view": "trace_view",
    "cann_kernel_details": "kernel_details",
}
DEFAULT_TIMING_BASIS = "cann_trace_view"
CHANNELS = {
    "ncu": {"ncu-direct", "ncu-derived", "sass-derived"},
    "msprof": {"msprof-direct", "msprof-derived", "cann-trace-view"},
    "cann_trace_view": {"cann-trace-view"},
    "none": set(),
}


def backend_for(language: str, configured: str = "auto", ncu_available: bool = True) -> str:
    if configured == "auto":
        if language == "pypto":
            return PYPTO_DEFAULT_BACKEND
        return "ncu" if ncu_available else "none"
    if configured not in BACKENDS:
        raise ValueError(f"profiler_backend must be one of {BACKENDS}, got {configured!r}")
    if language == "pypto" and configured == "ncu":
        raise ValueError("NCU cannot profile a PyPTO/Ascend task")
    if language != "pypto" and configured in ASCEND_BACKENDS:
        raise ValueError("Ascend profilers require a PyPTO task")
    if configured == "ncu" and not ncu_available:
        return "none"
    return configured


def timing_basis_problems(config: dict) -> list[str]:
    """Why one card's declared timing window cannot be read from its report."""
    problems = []
    basis = str(config.get("timing_basis") or DEFAULT_TIMING_BASIS)
    strategy = str(config.get("perf_metric_strategy") or TIMING_BASES[DEFAULT_TIMING_BASIS])
    if basis not in TIMING_BASES:
        problems.append(f"timing_basis must be one of {sorted(TIMING_BASES)}, got {basis!r}")
    elif TIMING_BASES[basis] != strategy:
        problems.append(
            f"timing_basis {basis!r} is read from a {TIMING_BASES[basis]!r} report, but "
            f"perf_metric_strategy is {strategy!r}: the field the adapter reads would be "
            f"absent from the official report")
    backend = str(config.get("profiler_backend") or PYPTO_DEFAULT_BACKEND)
    if backend not in ASCEND_BACKENDS:
        problems.append(f"profiler_backend must be one of {list(ASCEND_BACKENDS)}, got {backend!r}")
    return problems


def card_backend_problem(config: dict, loop_backend: str) -> str | None:
    """Why the control-side and execution-side evidence backends disagree."""
    card = str(config.get("profiler_backend") or PYPTO_DEFAULT_BACKEND)
    if card not in ASCEND_BACKENDS:
        return (f"eval/cann_task.json profiler_backend is {card!r}; expected one of "
                f"{list(ASCEND_BACKENDS)}")
    if card != loop_backend:
        return (f"eval/cann_task.json collects {card!r} but loop.toml [loop].profiler_backend "
                f"validates evidence channels for {loop_backend!r}: a gate line would cite a "
                f"channel this job does not collect, or evidence it does collect would be "
                f"rejected — set both to the same backend")
    return None


def evidence_digest(profiler: dict) -> dict:
    """Bound prompt size; full CSV/log evidence remains in the job archive."""
    out = {k: profiler[k] for k in ("backend", "timing_basis", "status", "reason",
                                   "source", "unavailable", "cases")
           if k in profiler}
    if "tables" in profiler:
        out["tables"] = [{k: table[k] for k in
                          ("file", "source", "status", "scope", "kernels", "total_recorded_us")
                          if k in table} for table in profiler["tables"]]
    return out


def benchmark_digest(official: dict) -> dict:
    out = {k: official[k] for k in ("tasks_version", "commit", "environment", "scores",
                                   "overall_score", "comparison", "job", "cost",
                                   "timing_basis", "case_scope") if k in official}
    out["cases"] = [{k: c.get(k) for k in ("case_id", "elapsed_us", "baseline_perf_us",
                     "t_hw_us", "speedup", "perf_score", "op_times", "memory_peak_mb")}
                    for c in official.get("cases", [])]
    return out


def channel_violations(text: str, backend: str) -> list[str]:
    allowed = CHANNELS[backend] | {"correctness", "probe", "inferred", "none"}
    return [f"verification channel {tag!r} unavailable for profiler backend {backend!r}"
            for tag in sorted(set(re.findall(r"\[VERIFY:\s*([\w-]+)\s*\]", text)))
            if tag not in allowed]
