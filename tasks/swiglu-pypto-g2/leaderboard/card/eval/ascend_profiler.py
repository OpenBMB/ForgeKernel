"""Ascend evidence collection (stdlib only; shipped with each frozen task card).

CANN's torch_npu collector supplies kernel_details.csv. msprof exports the same
raw PROF sessions independently to op_summary CSVs. Neither is NCU, and neither
replaces the task's trace_view timing currency. Units and collection scope stay
explicit: totals include all recorded repetitions, NOT one operator invocation.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import zlib

ARTIFACT_MARKER = "FORGE-ARTIFACT: "
MAX_FILE_BYTES = 16 * 1024 * 1024


def emit_artifact(name: str, data: bytes) -> dict:
    if len(data) > MAX_FILE_BYTES:
        return {"name": name, "status": "unavailable", "reason": "artifact too large"}
    digest = hashlib.sha256(data).hexdigest()
    print(ARTIFACT_MARKER + json.dumps({"name": name, "sha256": digest,
          "encoding": "zlib+base64", "data": base64.b64encode(zlib.compress(data)).decode()}),
          flush=True)
    return {"name": name, "sha256": digest, "bytes": len(data)}


def parse_kernel_csv(text: str) -> dict:
    """Read kernel_details / msprof op_summary without inventing absent counters."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    columns = reader.fieldnames or []
    duration = next((k for k in ("Duration(us)", "Task Duration(us)", "Duration (us)")
                     if k in columns), None)
    name = next((k for k in ("Name", "Op Name", "Task Name") if k in columns), None)
    if not duration or not name:
        return {"status": "unavailable", "reason": "unsupported CSV columns", "columns": columns}
    kernels: dict = {}
    invalid = 0
    for row in reader:
        try:
            us = float(row[duration])
            if not math.isfinite(us) or us <= 0:
                raise ValueError("nonpositive duration")
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        label = row.get(name) or "(unnamed)"
        k = kernels.setdefault(label, {"name": label, "launches": 0, "total_us": 0.0,
                                      "task_types": [], "block_dims": []})
        k["launches"] += 1
        k["total_us"] += us
        for key, aliases in (("task_types", ("Type", "Task Type")),
                             ("block_dims", ("Block Dim", "Block Dim(num)"))):
            value = next((row[a] for a in aliases if row.get(a)), None)
            if value is not None and value not in k[key]:
                k[key].append(value)
    rows = sorted(kernels.values(), key=lambda k: -k["total_us"])
    total = sum(k["total_us"] for k in rows)
    for k in rows:
        k["mean_us"] = k["total_us"] / k["launches"]
        k["pct"] = 100 * k["total_us"] / total
    return {"status": "available" if rows else "unavailable",
            "scope": "all recorded launches; not per-call gate time",
            "columns": columns, "invalid_rows": invalid, "kernels": rows,
            "total_recorded_us": total}


def collect(reports: Path, backend: str) -> tuple[dict, list[dict]]:
    """Export this evaluation's raw sessions, archive CSV bytes and expose a digest."""
    evidence: dict = {"backend": backend, "status": "unavailable", "sessions": [],
                      "unavailable": ["NCU/SASS metrics", "per-instruction stalls"]}
    artifacts: list[dict] = []
    if backend != "msprof":
        evidence.update(status="available", source="CANN-Bench op_times.trace_view")
        return evidence, artifacts
    executable = shutil.which("msprof")
    evidence["executable"] = executable
    if not executable:
        evidence["reason"] = "msprof executable unavailable"
        return evidence, artifacts
    sessions = sorted(p for p in reports.rglob("PROF_*") if p.is_dir())
    for session in sessions:
        try:
            proc = subprocess.run([executable, "--export=on", f"--output={session}"],
                                  capture_output=True, text=True, timeout=120)
            record = {"session": str(session.relative_to(reports)), "rc": proc.returncode,
                      "log": (proc.stdout + proc.stderr)[-4000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            record = {"session": str(session.relative_to(reports)), "rc": None,
                      "log": str(exc)}
        evidence["sessions"].append(record)
    tables = []
    for path in sorted(reports.rglob("*.csv")):
        if path.name != "kernel_details.csv" and not path.name.startswith("op_summary_"):
            continue
        rel = str(path.relative_to(reports))
        raw = path.read_bytes()
        artifacts.append(emit_artifact("cann/" + rel, raw))
        table = parse_kernel_csv(raw.decode("utf-8-sig", errors="replace"))
        tables.append({"file": rel, "source": "kernel_details" if path.name == "kernel_details.csv"
                       else "msprof_op_summary", **table})
    evidence["tables"] = tables
    exported = [t for t in tables if t["source"] == "msprof_op_summary"
                and t["status"] == "available"]
    if exported and any(s["rc"] == 0 for s in evidence["sessions"]):
        evidence["status"] = "available"
    else:
        evidence["reason"] = ("msprof export failed" if sessions else "no raw PROF sessions")
        if any(t["status"] == "available" for t in tables):
            evidence["status"] = "partial"
    artifacts.append(emit_artifact("cann/msprof.json", json.dumps(evidence, indent=2).encode()))
    return evidence, artifacts
