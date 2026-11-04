#!/usr/bin/env python3
"""Build a generic PyPTO CANN-Bench task with a strong, sealed library reference.

The initial GQA route is explicit; no fused library call is admitted as a student
implementation. Its kernel must be authored from the curriculum in PyPTO.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import yaml

from forgekernel.paths import repo_root
from forgekernel.task import load_task


def put(root: Path, rel: str, text: str, executable=False):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if executable:
        p.chmod(0o755)


def problem(cases, task_dir, operator, entry, commit):
    rows = []
    for i, case in enumerate(cases):
        case_id = str(case.get('case_id', i + 1))
        name = f'{task_dir}_{case_id}'
        workload = {'uuid': name, 'axes': {'case_id': case_id,
                    'input_shape': case.get('input_shape'), 'dtype': case.get('dtype'),
                    'attrs': case.get('attrs', {})}, 'input_groups': []}
        rows.append({'index': i, 'name': name, 'workload': workload})
    return {'schema': 1, 'suite': 'cann-bench', 'definition': {
        'name': operator, 'axes': {}, 'inputs': {'tensor': {'dtype': 'tensor'}},
        'outputs': {'y': {'dtype': 'tensor'}}, 'reference': 'CANN-Bench golden.py'},
        'workload': rows[0]['workload'], 'group': rows,
        'meta_workloads': [r['name'] for r in rows], 'dps': False,
        'kernel_file': 'kernel.py', 'entry': entry,
        'reference_impl': {'kind': 'separate_library_reference',
                           'name': 'Huawei CANN fused-attention library, separate tree'},
        'time_reference': False, 'timing': 'device',
        'tolerance': {'outputs': {}, 'rule': 'official CANN checker, no local hidden set'},
        'source': {'task': task_dir, 'commit': commit}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', type=Path)
    p.add_argument('--cann-bench-dir', type=Path, required=True)
    p.add_argument('--level', default='level4')
    p.add_argument('--operator-path', default='gqa')
    p.add_argument('--operator', default='GQA')
    p.add_argument('--entry', default='gqa')
    p.add_argument('--arch', default='ascend950')
    p.add_argument('--expected-cann-ref', default='059cb61d41b279ad021160517bd2a5f6b7a05e01')
    p.add_argument('--forgekernel-root', type=Path, default=repo_root())
    p.add_argument('--reference-mode', default='gqa-library',
                   choices=('gqa-library',))
    p.add_argument('--target-ratio', type=float, default=1.05)
    p.add_argument('--hours', type=float, default=16)
    p.add_argument('--max-meta', type=int, default=4)
    p.add_argument('--oracle', choices=('blackbox', 'source'), default='blackbox')
    p.add_argument('--student-model', default='deepseek-flash[1m]')
    p.add_argument('--provider', default='deepseek')
    p.add_argument('--pypto-source', default='/data/shwstone/Workspace/forgekernel-sources/pypto')
    p.add_argument('--pto-isa', default='/data/shwstone/Workspace/forgekernel-sources/pto-isa')
    args = p.parse_args(argv)
    root = args.forgekernel_root.resolve()
    dest = args.task.resolve()
    if dest.exists():
        p.error('task already exists')
    cann = args.cann_bench_dir.resolve()
    commit = subprocess.run(['git', '-C', str(cann), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != args.expected_cann_ref:
        p.error(f'unsupported CANN-Bench commit: {commit}')
    task_dir = f'{args.level}/{args.operator_path}'
    cases = yaml.safe_load((cann / task_dir / 'cases.yaml').read_text())['cases']
    card, ref = dest / 'leaderboard/card', dest / 'leaderboard/ref'
    template = root / 'src/forgekernel/card_template'
    for tree in (card, ref):
        tree.mkdir(parents=True)
        (tree / 'eval').mkdir()
        (tree / 'cann_bench').mkdir()
        for source, target in [('eval/cann_bench.py', 'eval/cann_bench.py'),
                               ('eval/ascend_profiler.py', 'eval/ascend_profiler.py'),
                               ('pypto_gpu_entry.sh', 'gpu_entry.sh')]:
            shutil.copy2(template / source, tree / target)
        put(tree, '.gitattributes', (template / '.gitattributes').read_text())
        put(tree, '.gitignore', 'loop.toml\nsubmit.local.toml\n.jobs/\n.exec/\n.cann_results/\n'
            'build/\ndist/\n*.egg-info/\n__pycache__/\n*.pyc\ncann_bench/kernel_impl.py\n')
        put(tree, 'setup.py', 'from setuptools import setup\n'
            'setup(name="cann_bench", version="1.0.0", packages=["cann_bench"])\n')
        put(tree, 'build.sh', '#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
            'rm -rf build dist cann_bench.egg-info cann_bench/kernel_impl.py\n'
            'cp kernel.py cann_bench/kernel_impl.py\npython setup.py bdist_wheel\n', True)
        put(tree, 'cann_bench/__init__.py', f'from .kernel_impl import {args.entry}\n')
    put(card, 'eval/cann_task.json', json.dumps({
        'operator': args.operator, 'task_dir': task_dir, 'source_dir': '.',
        'warmup': 3, 'repeat': 5, 'profiler_backend': 'msprof',
        'perf_metric_strategy': 'kernel_details', 'timing_basis': 'cann_trace_view'}, indent=2))
    put(ref, 'eval/cann_task.json', json.dumps({
        'operator': args.operator, 'task_dir': task_dir, 'source_dir': 'src',
        'warmup': 3, 'repeat': 5, 'profiler_backend': 'msprof',
        'perf_metric_strategy': 'kernel_details', 'timing_basis': 'cann_kernel_details'}, indent=2))
    put(card, 'eval/leaderboard/problem.json', json.dumps(problem(cases, task_dir,
        args.operator, args.entry, commit), indent=2))
    signature = 'query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, ' \
                'scaleValue: float = -1.0, is_causal: bool = False'
    put(card, 'kernel.py', 'import torch\nimport pypto\n\n'
        f'def {args.entry}({signature}) -> torch.Tensor:\n'
        '    raise NotImplementedError("implement PyPTO from the milestone")\n')
    library = (cann / 'scripts/baseline/refs/level4.py').read_text()
    (ref / 'src').mkdir()
    for name in ('build.sh', 'setup.py', 'cann_bench'):
        p = card / name
        if p.is_dir():
            shutil.copytree(p, ref / 'src' / name)
        else:
            shutil.copy2(p, ref / 'src' / name)
    put(ref, 'src/kernel.py', library + f'\n\ndef {args.entry}({signature}):\n'
        '    return gqa_ref([query, key, value],\n'
        '        {"scaleValue": scaleValue, "is_causal": is_causal})\n')
    for tree in (card, ref):
        put(tree, 'window.toml', 'schema_version = 1\n[reference]\nwindow = '
            '"Huawei fused-attention kernel_details sum, one call per sample"\n'
            'inputs = ["query", "key", "value"]\noutputs = ["y"]\n[student]\n'
            'window = "candidate, all kernels in kernel_details; trace_view AI-Core E2E for PyPTO"\n'
            'inputs = ["query", "key", "value"]\noutputs = ["y"]\n')
    put(card, 'loop.toml.example', f'''[loop]
language = "pypto"
kb_arch = "{args.arch}"
gate_window = "device"
oracle_mode = "{args.oracle}"
recon_op = ""
ref_mode = "ref"
ref_name = "Huawei CANN fused-infer-attention library (exact version in report)"
ref_shape = "{len(cases)} public {task_dir} cases"
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
dsl_source_dirs = [{json.dumps(args.pypto_source)}, {json.dumps(args.pto_isa)}]
''')
    put(card, 'README.md', f'''# {args.operator} — PyPTO on Ascend, CANN-Bench {task_dir}
Frozen public cases are judged by official golden.py, not the library reference.
`kernel.py` is student-owned and starts from a signature stub. No vendor operator
call is permitted as its implementation. The separate strong library tree is sealed
in blackbox mode. Public local scoring is not the paper's 910C / CANN 1.1.2 hosted result.
The library uses kernel_details kernel-only time. PyPTO records both that and trace_view
AI-Core E2E. Reference comparisons are cross-job median-vs-median, not paired timing.
''')
    shutil.copy2(cann / task_dir / 'desc.md', dest / 'description.md')
    put(dest, 'shapes.json', '[]\n')
    put(dest, 'task.toml', f'''[task]
language = "pypto"
arch = "{args.arch}"
[leaderboard]
suite = "cann-bench"
card = "leaderboard/card"
ref = "leaderboard/ref"
[target]
ratio = {args.target_ratio}
timing = "device"
[agent]
provider = {json.dumps(args.provider)}
model = {json.dumps(args.student_model)}
effort = "max"
[budget]
hours = {args.hours}
max_meta = {args.max_meta}
''')
    put(dest, 'experiment_protocol.json', json.dumps({
        'level': args.level.rsplit('/', 1)[-1], 'operator': args.operator, 'cases': 'public', 'judge': 'official golden.py',
        'architecture_reference': 'CANN library gqa_ref / npu_fused_infer_attention_score',
        'student': 'PyPTO authored kernel; vendor fused-attention calls forbidden',
        'oracle_mode': args.oracle, 'cann_commit': commit,
        'paper_equivalence': 'same operator family, NOT paper hardware/benchmark hosted reproduction',
    }, indent=2))
    load_task(dest, kb_root=root)
    print(dest)


if __name__ == '__main__':
    main()
