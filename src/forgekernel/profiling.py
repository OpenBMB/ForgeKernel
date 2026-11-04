"""Explicit profiler routing and evidence-channel validation."""
from __future__ import annotations

import re

BACKENDS = ("ncu", "msprof", "cann_trace_view", "none")
CHANNELS = {
    "ncu": {"ncu-direct", "ncu-derived", "sass-derived"},
    "msprof": {"msprof-direct", "msprof-derived", "cann-trace-view"},
    "cann_trace_view": {"cann-trace-view"},
    "none": set(),
}


def backend_for(language: str, configured: str = "auto", ncu_available: bool = True) -> str:
    if configured == "auto":
        return "cann_trace_view" if language == "pypto" else ("ncu" if ncu_available else "none")
    if configured not in BACKENDS:
        raise ValueError(f"profiler_backend must be one of {BACKENDS}, got {configured!r}")
    if language == "pypto" and configured == "ncu":
        raise ValueError("NCU cannot profile a PyPTO/Ascend task")
    if language != "pypto" and configured in ("msprof", "cann_trace_view"):
        raise ValueError("Ascend profilers require a PyPTO task")
    if configured == "ncu" and not ncu_available:
        return "none"
    return configured


def evidence_digest(profiler: dict) -> dict:
    """Bound prompt size; full CSV/log evidence remains in the job archive."""
    out = {k: profiler[k] for k in ("backend", "status", "reason", "source", "unavailable", "cases")
           if k in profiler}
    if "tables" in profiler:
        out["tables"] = [{k: table[k] for k in
                          ("file", "source", "status", "scope", "kernels", "total_recorded_us")
                          if k in table} for table in profiler["tables"]]
    return out


def benchmark_digest(official: dict) -> dict:
    out = {k: official[k] for k in ("tasks_version", "commit", "environment", "scores",
                                   "overall_score", "comparison", "job") if k in official}
    out["cases"] = [{k: c.get(k) for k in ("case_id", "elapsed_us", "baseline_perf_us",
                     "t_hw_us", "speedup", "perf_score", "op_times", "memory_peak_mb")}
                    for c in official.get("cases", [])]
    return out


def channel_violations(text: str, backend: str) -> list[str]:
    allowed = CHANNELS[backend] | {"correctness", "probe", "inferred", "none"}
    return [f"verification channel {tag!r} unavailable for profiler backend {backend!r}"
            for tag in sorted(set(re.findall(r"\[VERIFY:\s*([\w-]+)\s*\]", text)))
            if tag not in allowed]
