#!/usr/bin/env python3
"""Frozen Ascend base/variant experiment, independent of agent timing code.

Usage: python3 notes/ascend_probe.py notes/experiment.json
Spec: {"edits":[{"old":"unique text","new":"replacement"}], "repeat":3,
       "min_gain_pct":1.5, "hypothesis":"falsifiable claim", "cases":"scout"}

The card's whole official case set is the default and the only certification. ``cases``
narrows one experiment to a pricing reading: ``scout`` is the card's declared
representative set, any other value is a comma-separated id list (``gqa_8`` is a single
worst-case probe). A narrowed experiment prints ``PROBE SCOUT``, never ``PROBE PASS``:
five cases out of twenty price an architecture decision for the teacher, they do not
certify a lever. Certifying one still costs the full set.

``repeat`` is 3..5 for a full-set experiment (alternated repeats are its noise floor)
and 1..5 for a narrowed one: a single scoped round is one paired base/variant A/B, two
evaluations, which is the point of a scout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile


def apply_edits(source: str, edits: list) -> str:
    spans = []
    for edit in edits:
        old, new = edit["old"], edit["new"]
        if not old or source.count(old) != 1:
            raise ValueError("each old text must occur exactly once in the original kernel")
        start = source.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("overlapping edits")
    for start, end, new in reversed(spans):
        source = source[:start] + new + source[end:]
    return source


def evaluation_env(spec: dict) -> tuple[dict, str]:
    """The environment for one evaluation: case scope, and an adapter-owned budget.

    ``FK_CANN_CASES`` is the card adapter's own input (``scout`` or explicit ids); unset
    means the whole official set. ``FK_CANN_EVAL_BUDGET`` is set one reserve below this
    experiment's own timeout, so an evaluation that never finishes is stopped by the
    adapter -- which then writes its cost and its scope -- instead of being killed from
    here with nothing to show for it.
    """
    env = {**os.environ}
    scope = str(spec.get("cases") or "").strip()
    if scope:
        env["FK_CANN_CASES"] = scope
    else:
        env.pop("FK_CANN_CASES", None)
    timeout = int(spec.get("timeout_s", 2700))
    env["FK_CANN_EVAL_BUDGET"] = str(max(60, timeout - 60))
    return env, scope


def adapter_reading(stdout: str, rc: int, stderr: str, label: str) -> tuple[dict, float]:
    """One evaluation's eval JSON and the worst error norm it reported.

    The JSON is read before the exit status is judged: a job the adapter stopped still
    publishes structured cost and scope, and throwing that away would report "failed"
    for exactly the outcome whose point is knowing what it spent. A partial or stopped
    reading is surfaced as a diagnostic -- never as a PASS.
    """
    marker = stdout.rfind("=== JSON ===")
    result = None
    if marker >= 0:
        try:
            result = json.loads(stdout[marker:].splitlines()[1])
        except (IndexError, ValueError):
            result = None
    cost = ((result or {}).get("cann_bench") or {}).get("cost") or {}
    scope = (result or {}).get("workload_scope")
    if cost or scope:
        print("ASCEND-EVAL-COST " + json.dumps(
            {"label": label, "rc": rc, "cost": cost, "workload_scope": scope},
            separators=(",", ":")), flush=True)
    if result is None:
        raise RuntimeError(f"{label} evaluation produced no eval JSON (rc={rc}): "
                           f"{(stderr or stdout)[-1000:]}")
    correctness = result.get("correctness") or {}
    if rc or not correctness.get("pass"):
        raise RuntimeError(
            f"{label} evaluation produced no gate reading (rc={rc}): "
            f"{json.dumps({'correctness': correctness, 'cost': cost, 'workload_scope': scope}, separators=(',', ':'))[:1200]}")
    worst = 0.0
    for case in result.get("cann_bench", {}).get("cases", []):
        for output in (case.get("accuracy") or {}).get("output_results", []):
            err = output.get("max_diff")
            if isinstance(err, (float, int)) and math.isfinite(err):
                worst = max(worst, abs(err))
    return result, worst


def _forward_artifacts(stdout: str, prefix: str) -> None:
    """Re-emit the adapter's checksummed artifacts under this experiment's own names."""
    for line in stdout.splitlines():
        if line.startswith("FORGE-ARTIFACT: "):
            artifact = json.loads(line.split(": ", 1)[1])
            artifact["name"] = prefix + artifact["name"]
            print("FORGE-ARTIFACT: " + json.dumps(artifact), flush=True)


def run_experiment(workspace: Path, spec: dict) -> dict:
    requested = str(spec.get("cases") or "").strip()
    repeat = int(spec.get("repeat", 3))
    # A full-set experiment is a certification: alternated repeats are what give it a
    # two-sided noise floor. A narrowed one certifies nothing, so one paired A/B is a
    # legitimate pricing round -- two evaluations, i.e. 10 case runs over a 5-case scout
    # set rather than 30. That is what makes "falsify this architecture cheaply" real.
    if requested:
        if not 1 <= repeat <= 5:
            raise ValueError("repeat must be 1..5 for a scoped experiment")
    elif not 3 <= repeat <= 5:
        raise ValueError("repeat must be 3..5 for a full-set experiment")
    source = (workspace / "kernel.py").read_text()
    variant = apply_edits(source, spec["edits"])
    if source == variant:
        raise ValueError("no-op experiment")
    env, scope = evaluation_env(spec)
    records = {"base": [], "variant": []}
    workloads = None
    worst_error = 0.0
    python = os.environ.get("FK_ASCEND_PYTHON", "python3")
    with tempfile.TemporaryDirectory(prefix="ascend-experiment-") as tmp:
        for i in range(repeat):
            # Alternate order to expose rather than systematically favour drift.
            for label in (("base", "variant") if i % 2 == 0 else ("variant", "base")):
                tree = Path(tmp) / f"{label}-{i}"
                shutil.copytree(workspace, tree, ignore=shutil.ignore_patterns(
                    ".git", ".jobs", ".exec", ".cann_results", "build", "dist", "*.egg-info",
                    "__pycache__", "*.pyc", "*.local.toml"))
                (tree / "kernel.py").write_text(source if label == "base" else variant)
                try:
                    proc = subprocess.run(
                        [python, "eval/cann_bench.py", "--workspace", str(tree)],
                        cwd=tree, capture_output=True, text=True, env=env,
                        timeout=int(spec.get("timeout_s", 2700)))
                except subprocess.TimeoutExpired as exc:
                    # The adapter stops itself first (FK_CANN_EVAL_BUDGET), so this only
                    # fires if it could not; its partial eval still carries the cost.
                    out = exc.stdout or ""
                    if isinstance(out, bytes):
                        out = out.decode(errors="replace")
                    print(out, flush=True)
                    _forward_artifacts(out, f"probe/{label}-{i}/")
                    raise RuntimeError(
                        f"{label} evaluation exceeded {spec.get('timeout_s', 2700)}s; "
                        f"partial adapter output above") from exc
                _forward_artifacts(proc.stdout, f"probe/{label}-{i}/")
                result, worst = adapter_reading(proc.stdout, proc.returncode, proc.stderr, label)
                worst_error = max(worst_error, worst)
                if not result["correctness"]["pass"]:
                    raise RuntimeError(f"{label} failed official correctness: {result['correctness']}")
                perf = result["performance"]
                names = sorted(w["name"] for w in perf["workloads"])
                if workloads is not None and names != workloads:
                    raise ValueError("workload set changed within experiment")
                workloads = names
                records[label].append({"ms": perf["device_median_ms"],
                                       "workloads": perf["workloads"]})
                print(f"ASCEND-SAMPLE {label} round={i} ms={perf['device_median_ms']}", flush=True)
    base = [r["ms"] for r in records["base"]]
    var = [r["ms"] for r in records["variant"]]
    gain = 100 * (statistics.median(base) / statistics.median(var) - 1)
    noise = max(1.5, 100 * ((max(base) / min(base)) * (max(var) / min(var)) - 1))
    passed = gain > max(noise, float(spec.get("min_gain_pct", 1.5)))
    result = {"hypothesis": spec.get("hypothesis"),
              "kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "gain_pct": gain, "noise_pct": noise, "records": records,
              "repeat": repeat,
              "scope": (f"{len(workloads)} of the card's cases ({','.join(workloads)}), "
                        f"via FK_CANN_CASES={scope}; pricing only, NOT a gate reading"
                        if scope else
                        "full official public case set; alternating separate evaluations "
                        "in one job"),
              "cases": scope or "all",
              "certifies": not scope,
              "verdict": "SCOUT-PRICE" if scope else ("PASS" if passed else "FAIL")}
    print("ASCEND-PRICE-JSON: " + json.dumps(result), flush=True)
    if scope:
        # No PROBE PASS line on purpose: a subset has no certification to give, and a
        # PASS line is exactly how one would be read as one.
        print(f"PROBE SCOUT max_diff={worst_error} gain_pct={gain:.6f} "
              f"noise_pct={noise:.6f} cases={len(workloads)} (FK_CANN_CASES={scope}): "
              f"a scout prices an architecture decision for the next full-set gate; it "
              f"is not a gate reading and cannot certify a lever.", flush=True)
        return result
    print(f"PROBE {result['verdict']} max_diff={worst_error} gain_pct={gain:.6f} "
          f"noise_pct={noise:.6f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    run_experiment(Path.cwd(), json.loads(args.spec.read_text()))
