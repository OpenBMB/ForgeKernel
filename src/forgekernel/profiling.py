"""Explicit profiler routing and evidence-channel validation.

Two axes that look alike and are not. The *collection backend* is the tool a job
runs (``msprof`` exports task rows; ``cann_trace_view`` reads only what the
official report already carries). The *timing basis* is the window a reading is
measured in (``cann_trace_view`` AI-Core E2E, ``cann_kernel_details`` kernel-only).
A card may collect msprof and still time on trace_view -- every PyPTO card this
repo generates does exactly that -- so one name cannot stand for both.

Keeping them apart is what the channel validator depends on: ``[VERIFY:
msprof-direct]`` is discharged by the *collection* backend, not by the window the
gate is timed in. Collapsing the two made the default PyPTO backend
``cann_trace_view``, which rejects the msprof channels that ``meta/pypto/``
instructs the teacher to cite -- recon then re-prompted to exhaustion and the
meta layer stopped with no measurement at fault.
"""
from __future__ import annotations

import re

BACKENDS = ("ncu", "msprof", "cann_trace_view", "none")
CHANNELS = {
    "ncu": {"ncu-direct", "ncu-derived", "sass-derived"},
    "msprof": {"msprof-direct", "msprof-derived", "cann-trace-view"},
    "cann_trace_view": {"cann-trace-view"},
    "none": set(),
}

# Ascend collection backends, in the order a card should prefer them: msprof is a
# strict superset of what trace_view answers, so it is the PyPTO default.
ASCEND_BACKENDS = ("msprof", "cann_trace_view")

# A timing basis is a window, not a tool. Each one is readable only from the
# official report the matching ``--perf-metric-strategy`` produced, so the pair is
# a single declaration with two spellings and both halves must agree.
TIMING_BASES = {
    "cann_trace_view": "trace_view",
    "cann_kernel_details": "kernel_details",
}
DEFAULT_TIMING_BASIS = "cann_trace_view"


def backend_for(language: str, configured: str = "auto", ncu_available: bool = True) -> str:
    if configured == "auto":
        # msprof, not cann_trace_view: every PyPTO card generated here collects
        # msprof, and the PyPTO prompts cite msprof-direct/-derived. Defaulting to
        # the narrower backend fails closed against the harness's own prompts.
        return ASCEND_BACKENDS[0] if language == "pypto" else ("ncu" if ncu_available else "none")
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
    """Why one card's ``eval/cann_task.json`` cannot be measured, or an empty list.

    The adapter reads the window named by ``timing_basis`` out of the report that
    ``perf_metric_strategy`` asked the official evaluator for. A mismatched pair
    produces a report with no such field, which is a *measurement* fault -- the
    one that invalidated the first GQA M0 gate -- so it is caught here, at launch
    and at card generation, rather than surfacing later as a kernel verdict.
    """
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
    backend = str(config.get("profiler_backend") or ASCEND_BACKENDS[0])
    if backend not in ASCEND_BACKENDS:
        problems.append(f"profiler_backend must be one of {list(ASCEND_BACKENDS)}, got {backend!r}")
    return problems


def card_backend_problem(config: dict, loop_backend: str) -> str | None:
    """Why the control side's profiler backend disagrees with the card's, or None.

    ``loop.toml [loop].profiler_backend`` decides which ``[VERIFY: ...]`` channels
    the milestone and recon gates accept; ``eval/cann_task.json
    profiler_backend`` decides which tool the job actually runs. Nothing read both
    before, so either direction could pass every check and still be wrong: a
    control side set to ``msprof`` admits gate lines no job collects evidence for,
    and one set to ``cann_trace_view`` rejects the evidence the job did collect.
    """
    card = str(config.get("profiler_backend") or ASCEND_BACKENDS[0])
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
    out = {k: profiler[k] for k in ("backend", "timing_basis", "status", "reason", "source",
                                    "unavailable", "cases") if k in profiler}
    if "tables" in profiler:
        out["tables"] = [{k: table[k] for k in
                          ("file", "source", "status", "scope", "kernels", "total_recorded_us")
                          if k in table} for table in profiler["tables"]]
    return out


def benchmark_digest(official: dict) -> dict:
    # ``cost`` rides the digest so the outer loop can see a compile-bound campaign:
    # a job that spent its budget in JIT and a job that spent it on device both
    # come back as one wall-clock number otherwise, and the teacher cannot tell a
    # slow kernel from an unaffordable one.
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
