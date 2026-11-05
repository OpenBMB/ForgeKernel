#!/usr/bin/env python3
"""Create a ForgeKernel PyPTO task from CANN-Bench's public SwiGlu reference.

The initial student candidate is the same PUBLIC implementation, consolidated into one
editable kernel.py. This is an optimization run, not a claim that the student invented the
baseline. The reference also lives in a separate teacher-only tree for source distillation
and independent measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from forgekernel import arch as arch_mod
from forgekernel import profiling

ADAPTER = Path("src/forgekernel/card_template/eval/cann_bench.py")
GPU_ENTRY = Path("src/forgekernel/card_template/pypto_gpu_entry.sh")
PROFILER = Path("src/forgekernel/card_template/eval/ascend_profiler.py")


def put(root: Path, rel: str, text: str, executable: bool = False) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def consolidate_reference(reference: Path) -> str:
    """Combine c1..c8 into one source file without sharing global tile constants."""
    root = reference / "cann_bench" / "swi_glu"
    chunks = ["# Consolidated from CANN-Bench's public PyPTO SwiGlu c1..c8 classes.\n",
              "# The initial candidate is public code; subsequent changes are agent-made.\n"]
    for i in range(1, 9):
        path = root / f"c{i}" / "swi_glu_impl.py"
        text = path.read_text(encoding="utf-8")
        for name in ("swi_glu_kernel_npu", "swi_glu_module1_wrapper",
                     "swi_glu_impl", "swi_glu_wrapper"):
            text = re.sub(rf"\b{name}\b", f"{name}_c{i}", text)
        text = re.sub(r"\b(T\d+)\b", rf"\1_C{i}", text)
        chunks.append(f"\n# ===== Public reference class c{i} =====\n{text}")
    chunks.append(r'''

# Dispatch contract from the public reference: (ndim, dtype) -> implementation class.
_CLASSES = {
    (2, "float16"): swi_glu_impl_c1,
    (2, "float32"): swi_glu_impl_c2,
    (2, "bfloat16"): swi_glu_impl_c3,
    (3, "bfloat16"): swi_glu_impl_c4,
    (4, "float32"): swi_glu_impl_c5,
    (5, "float16"): swi_glu_impl_c6,
    (3, "float32"): swi_glu_impl_c7,
    (5, "float32"): swi_glu_wrapper_c8,
}


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    signature = (int(input.ndim), str(input.dtype).rsplit(".", 1)[-1])
    try:
        implementation = _CLASSES[signature]
    except KeyError as exc:
        raise ValueError(f"no PyPTO SwiGlu class for {signature}") from exc
    return implementation(input, dim=dim)
''')
    return "\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path, help="new tasks/<name> directory")
    parser.add_argument("--cann-bench-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="CANN-Bench examples/pypto_example")
    parser.add_argument("--arch", default="ascend950pr")
    parser.add_argument("--hardware", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--effort", default="")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-provider", default="")
    parser.add_argument("--teacher-effort", default="")
    parser.add_argument("--expected-cann-ref", default="")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--target-ratio", type=float, default=1.05)
    parser.add_argument("--pypto-source", default="")
    parser.add_argument("--pto-isa", default="")
    parser.add_argument("--dsl-source-dir", action="append", default=[])
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    task = args.task.resolve()
    cann = args.cann_bench_dir.resolve()
    reference = args.reference_dir.resolve()
    if task.exists():
        parser.error(f"task already exists: {task}")
    actual = subprocess.run(["git", "-C", str(cann), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    expected_ref = args.expected_cann_ref or actual
    if actual != expected_ref:
        parser.error(f"CANN-Bench ref is {actual}, expected {expected_ref}")
    args.arch = arch_mod.canonical_arch(args.arch)
    if not arch_mod.is_ascend_arch(args.arch):
        parser.error(f"not an Ascend architecture: {args.arch}")
    hardware = args.hardware or arch_mod.hardware_label(args.arch)
    task_dir = "tasks/level1/swi_glu"
    cases = yaml.safe_load((cann / task_dir / "cases.yaml").read_text(encoding="utf-8"))
    if isinstance(cases, dict):
        cases = cases.get("cases", cases.get("workloads"))
    if not isinstance(cases, list) or len(cases) != 20:
        parser.error("expected the 20 public SwiGlu cases")

    card = task / "leaderboard" / "card"
    ref = task / "leaderboard" / "ref"
    for tree in (card, ref):
        put(tree, "eval/cann_bench.py", (repo / ADAPTER).read_text(encoding="utf-8"))
        put(tree, "eval/ascend_profiler.py", (repo / PROFILER).read_text(encoding="utf-8"))
        put(tree, "gpu_entry.sh", (repo / GPU_ENTRY).read_text(encoding="utf-8"), True)
        put(tree, ".gitattributes", (repo / "src/forgekernel/card_template/.gitattributes")
            .read_text(encoding="utf-8"))
        put(tree, ".gitignore",
            "loop.toml\nsubmit.local.toml\n.jobs/\n.exec/\n.cann_results/\nreports/\n"
            "build/\ndist/\n*.egg-info/\ncann_bench/kernel_impl.py\n"
            "__pycache__/\n*.pyc\n_compile.log\n")

    # Spelled out rather than left to the adapter's defaults: the pair that decides
    # which window every gate reads is the one a reader must be able to see, and
    # the validator below is what makes the two halves one declaration.
    config = {"operator": "SwiGlu", "task_dir": task_dir, "warmup": 3, "repeat": 5,
              "profiler_backend": "msprof", "perf_metric_strategy": "trace_view",
              "timing_basis": "cann_trace_view", "expected_cann_ref": expected_ref}
    if problems := profiling.timing_basis_problems(config):
        parser.error("; ".join(problems))
    if why := profiling.card_backend_problem(
            config, profiling.backend_for("pypto", config["profiler_backend"])):
        parser.error(why)
    put(card, "eval/cann_task.json", json.dumps({**config, "source_dir": "."}, indent=2) + "\n")
    put(ref, "eval/cann_task.json", json.dumps({**config, "source_dir": "src"}, indent=2) + "\n")

    workloads = []
    for i, case in enumerate(cases):
        case_id = str(case.get("case_id", i + 1))
        name = f"level1/swi_glu_{case_id}"
        workload = {
            "uuid": name,
            "axes": {"case_id": case_id, "input_shape": case.get("input_shape"),
                     "dtype": case.get("dtype"), "attrs": case.get("attrs", {})},
            "input_groups": [],
        }
        workloads.append({"index": i, "name": name, "workload": workload})
    problem = {
        "schema": 1,
        "suite": "cann-bench",
        "definition": {"name": "SwiGlu", "axes": {},
                       "inputs": {"input": {"dtype": "tensor"}},
                       "outputs": {"output": {"dtype": "tensor"}},
                       "reference": "CANN-Bench golden.py"},
        "workload": workloads[0]["workload"],
        "group": workloads,
        "meta_workloads": [w["name"] for w in workloads],
        "dps": False,
        "kernel_file": "kernel.py",
        "entry": "swi_glu",
        "reference_impl": {"kind": "cann_bench_reference",
                           "name": "public PyPTO reference, measured in a separate tree"},
        "time_reference": False,
        "timing": "device",
        "tolerance": {"outputs": {}, "rule": "CANN-Bench MERE/MARE per public case"},
        "source": {"task": "official-tasks 2.0.0", "commit": expected_ref},
    }
    put(card, "eval/leaderboard/problem.json", json.dumps(problem, indent=2) + "\n")

    put(card, "build.sh", """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build dist cann_bench.egg-info cann_bench/kernel_impl.py
cp kernel.py cann_bench/kernel_impl.py
python setup.py bdist_wheel
test -n "$(find dist -maxdepth 1 -name 'cann_bench*.whl' -print -quit)"
""", True)
    put(card, "setup.py", """from setuptools import find_packages, setup
setup(name="cann_bench", version="1.0.0", packages=find_packages(),
      python_requires=">=3.10")
""")
    put(card, "cann_bench/__init__.py",
        "from .kernel_impl import swi_glu\n\n__all__ = [\"swi_glu\"]\n")
    put(card, "kernel.py", consolidate_reference(reference))
    put(card, "window.toml", """schema_version = 1
[reference]
window = "CANN-Bench public SwiGlu reference, one call per AI-Core E2E sample"
inputs = ["input"]
outputs = ["output"]
[student]
window = "cann_bench.swi_glu, one call per AI-Core E2E sample"
inputs = ["input"]
outputs = ["output"]
""")
    put(ref, "window.toml", (card / "window.toml").read_text(encoding="utf-8"))

    dsl_dirs = [p for p in (args.pypto_source, args.pto_isa, *args.dsl_source_dir) if p]
    loop = f"""# Read by forge run; machine-specific oracle paths are filled from the frozen ref tree.
[loop]
language = "pypto"
kb_arch = {json.dumps(args.arch)}
gate_window = "device"
oracle_mode = "source"
recon_op = ""
ref_mode = "ref"
ref_name = "CANN-Bench public PyPTO SwiGlu"
ref_shape = "20 public cases"
ref_hardware = {json.dumps(hardware)}
ref_from_goal = "off"
ref_price_op = ""
precision_contract = "off"
correctness_monitor = "off"
confirm_jobs = 3
eval_timeout_s = 7200
profiler_backend = "msprof"
ncu_available = false
job_env_preflight = "warn"
timeline_probe = "off"
stall_pricing = false
orch_census = false
dsl_source_dirs = {json.dumps(dsl_dirs)}
"""
    put(card, "loop.toml.example", loop)
    put(card, "README.md", """# SwiGlu — PyPTO / CANN-Bench

The official 20 public cases are the only judge. `kernel.py` is the student-owned candidate;
`build.sh` packages it as `cann_bench.swi_glu`. `eval/cann_bench.py` converts the official
`trace_view` report into ForgeKernel's correctness + geomean eval JSON. The reference runs in
the separate `_<op>ref` tree and is never an in-job comparator.
""")

    ignore = shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__",
                                    "_compile.log", ".cann_results", "reports")
    shutil.copytree(reference, ref / "src", ignore=ignore)
    put(task, "description.md",
        (cann / task_dir / "desc.md").read_text(encoding="utf-8")
        + "\n\nThe initial student candidate is the public PyPTO implementation; optimize it.\n")
    put(task, "shapes.json", "[]\n")
    common = []
    if args.provider:
        common.append(f'provider = {json.dumps(args.provider)}')
    if args.model:
        common.append(f'model = {json.dumps(args.model)}')
    if args.effort:
        common.append(f'effort = {json.dumps(args.effort)}')
    teacher = []
    if args.teacher_provider:
        teacher.append(f'provider = {json.dumps(args.teacher_provider)}')
    if args.teacher_model:
        teacher.append(f'model = {json.dumps(args.teacher_model)}')
    if args.teacher_effort:
        teacher.append(f'effort = {json.dumps(args.teacher_effort)}')
    agent_toml = ''
    if common:
        agent_toml += '[agent]\n' + '\n'.join(common) + '\n'
    if teacher:
        agent_toml += '[agent.teacher]\n' + '\n'.join(teacher) + '\n'
    put(task, "task.toml", f"""# PyPTO on Ascend, judged by CANN-Bench official-tasks 2.0.0.
[task]
language = "pypto"
arch = "{args.arch}"

[leaderboard]
suite = "cann-bench"
card = "leaderboard/card"
ref = "leaderboard/ref"

[target]
ratio = {args.target_ratio}
timing = "device"

{agent_toml}
[budget]
hours = {args.hours}
max_meta = 3
""")
    print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
