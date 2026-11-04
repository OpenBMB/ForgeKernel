#!/usr/bin/env python3
"""Frozen full-case Ascend base/variant experiment, independent of agent timing code.

Usage: python3 notes/ascend_probe.py notes/experiment.json
Spec: {"edits":[{"old":"unique text","new":"replacement"}], "repeat":3,
       "min_gain_pct":1.5, "hypothesis":"falsifiable claim"}
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


def run_experiment(workspace: Path, spec: dict) -> dict:
    repeat = int(spec.get("repeat", 3))
    if not 3 <= repeat <= 5:
        raise ValueError("repeat must be 3..5 for a priced experiment")
    source = (workspace / "kernel.py").read_text()
    variant = apply_edits(source, spec["edits"])
    if source == variant:
        raise ValueError("no-op experiment")
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
                proc = subprocess.run([python, "eval/cann_bench.py", "--workspace", str(tree)],
                                      cwd=tree, capture_output=True, text=True,
                                      timeout=int(spec.get("timeout_s", 2700)))
                # Preserve raw reports under unique artifact paths for this experiment.
                for line in proc.stdout.splitlines():
                    if line.startswith("FORGE-ARTIFACT: "):
                        artifact = json.loads(line.split(": ", 1)[1])
                        artifact["name"] = f"probe/{label}-{i}/" + artifact["name"]
                        print("FORGE-ARTIFACT: " + json.dumps(artifact), flush=True)
                marker = proc.stdout.rfind("=== JSON ===")
                if proc.returncode or marker < 0:
                    raise RuntimeError(f"{label} evaluation failed: {proc.stderr[-1000:]}")
                result = json.loads(proc.stdout[marker:].splitlines()[1])
                if not result["correctness"]["pass"]:
                    raise RuntimeError(f"{label} failed official correctness: {result['correctness']}")
                perf = result["performance"]
                names = sorted(w["name"] for w in perf["workloads"])
                if workloads is not None and names != workloads:
                    raise ValueError("workload set changed within experiment")
                workloads = names
                records[label].append({"ms": perf["device_median_ms"],
                                       "workloads": perf["workloads"]})
                for case in result.get("cann_bench", {}).get("cases", []):
                    for output in (case.get("accuracy") or {}).get("output_results", []):
                        err = output.get("max_diff")
                        if isinstance(err, (float, int)) and math.isfinite(err):
                            worst_error = max(worst_error, abs(err))
                print(f"ASCEND-SAMPLE {label} round={i} ms={perf['device_median_ms']}", flush=True)
    base = [r["ms"] for r in records["base"]]
    var = [r["ms"] for r in records["variant"]]
    gain = 100 * (statistics.median(base) / statistics.median(var) - 1)
    noise = max(1.5, 100 * ((max(base) / min(base)) * (max(var) / min(var)) - 1))
    passed = gain > max(noise, float(spec.get("min_gain_pct", 1.5)))
    result = {"hypothesis": spec.get("hypothesis"), "kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "gain_pct": gain, "noise_pct": noise, "records": records,
              "scope": "full official public case set; alternating separate evaluations in one job",
              "verdict": "PASS" if passed else "FAIL"}
    print("ASCEND-PRICE-JSON: " + json.dumps(result), flush=True)
    print(f"PROBE {result['verdict']} max_diff={worst_error} gain_pct={gain:.6f} noise_pct={noise:.6f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    run_experiment(Path.cwd(), json.loads(args.spec.read_text()))
