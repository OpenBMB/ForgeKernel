from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import yaml

from forgekernel.task import load_task


def builder():
    p = Path('scripts/init_pypto_cann_task.py').resolve()
    spec = importlib.util.spec_from_file_location('init_cann', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generic_level4_task_is_complete_and_sealable(tmp_path):
    cann = tmp_path / 'cann'
    task_dir = cann / 'tasks/level4/gqa'
    task_dir.mkdir(parents=True)
    case = {'case_id': 1, 'input_shape': [[1, 2, 4, 8]] * 3,
            'dtype': ['float16'] * 3, 'attrs': {'scaleValue': -1, 'is_causal': False}}
    (task_dir / 'cases.yaml').write_text(yaml.safe_dump({'cases': [case]}))
    (task_dir / 'desc.md').write_text('test operator semantics')
    (cann / 'scripts/baseline/refs').mkdir(parents=True)
    (cann / 'scripts/baseline/refs/level4.py').write_text(
        'def gqa_ref(inputs, attrs):\n    return inputs[0]\n')
    subprocess.run(['git', 'init', '-q', str(cann)], check=True)
    subprocess.run(['git', '-C', str(cann), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(cann), '-c', 'user.name=probe',
                    '-c', 'user.email=probe@local', 'commit', '-qm', 'fixture'], check=True)
    dest = tmp_path / 'tasks/gqa'
    commit = subprocess.run(['git', '-C', str(cann), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    builder().main([str(dest), '--cann-bench-dir', str(cann), '--expected-cann-ref', commit,
                    '--level', 'tasks/level4',
                    '--pypto-source', '.', '--pto-isa', '.'])
    task = load_task(dest)
    assert task.name == 'gqa' and len(task.shapes) == 1
    assert task.teacher.provider == task.student.provider == 'deepseek'
    assert task.teacher.model == task.student.model == 'deepseek-flash[1m]'
    for name in ('build.sh', 'setup.py', 'kernel.py', 'cann_bench/__init__.py'):
        assert (task.leaderboard.ref / 'src' / name).is_file(), name
    assert 'NotImplementedError' in (task.leaderboard.card / 'kernel.py').read_text()
    card = json.loads((task.leaderboard.card / 'eval/cann_task.json').read_text())
    ref = json.loads((task.leaderboard.ref / 'eval/cann_task.json').read_text())
    assert card['timing_basis'] == 'cann_trace_view'
    assert ref['timing_basis'] == 'cann_kernel_details'
    assert card['perf_metric_strategy'] == ref['perf_metric_strategy'] == 'kernel_details'
    from forgekernel.cannref import seal_tree
    seal_tree(task.leaderboard.ref, 'kernel.py::gqa')
    assert not (task.leaderboard.ref / 'src').exists()
    assert (task.leaderboard.ref / 'eval/closed_ref.py').is_file()
