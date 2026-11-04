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

EXPECTED_CANN_REF = "059cb61d41b279ad021160517bd2a5f6b7a05e01"
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
    parser.add_argument("--model", default="deepseek-flash[1m]")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--target-ratio", type=float, default=1.05)
    parser.add_argument("--pypto-source", default="")
    parser.add_argument("--pto-isa", default="/home/shw/Workspace/pto-isa")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    task = args.task.resolve()
    cann = args.cann_bench_dir.resolve()
    reference = args.reference_dir.resolve()
    if task.exists():
        parser.error(f"task already exists: {task}")
    actual = subprocess.run(["git", "-C", str(cann), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != EXPECTED_CANN_REF:
        parser.error(f"CANN-Bench ref is {actual}, expected {EXPECTED_CANN_REF}")
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

    config = {"operator": "SwiGlu", "task_dir": task_dir, "warmup": 3, "repeat": 5,
              "profiler_backend": "msprof"}
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
        "source": {"task": "official-tasks 2.0.0", "commit": EXPECTED_CANN_REF},
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

    dsl_dirs = [p for p in (args.pypto_source, args.pto_isa) if p]
    loop = f"""# Read by forge run; machine-specific oracle paths are filled from the frozen ref tree.
[loop]
language = "pypto"
kb_arch = "ascend950"
gate_window = "device"
oracle_mode = "source"
recon_op = ""
ref_mode = "ref"
ref_name = "CANN-Bench public PyPTO SwiGlu"
ref_shape = "20 public cases"
ref_hardware = "Ascend950PR"
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
    put(task, "task.toml", f"""# PyPTO on Ascend 950PR, judged by CANN-Bench official-tasks 2.0.0.
[task]
language = "pypto"
arch = "ascend950"

[leaderboard]
suite = "cann-bench"
card = "leaderboard/card"
ref = "leaderboard/ref"

[target]
ratio = {args.target_ratio}
timing = "device"

[agent]
provider = {json.dumps(args.provider)}
model = {json.dumps(args.model)}
effort = "max"

[budget]
hours = {args.hours}
max_meta = 3
""")
    print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
