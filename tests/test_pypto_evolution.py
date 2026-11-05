from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from forgekernel import launch, taskcard, sealref
from forgekernel.paths import repo_root, template_dir
from forgekernel.task import load_task
from test_self_evolve import se


def best():
    source = (repo_root() / 'tasks/swiglu-pypto/leaderboard/card/kernel.py').read_text()
    return {'sha': 'f' * 40, 'kernel': source, 'speedup': 1.1,
            'ratio_basis': 'cross-job-median-vs-median', 'requested': 3,
            'samples_ms': [1.0, 1.001, 1.002], 'ref_samples_ms': [1.1, 1.101, 1.102]}


def test_cross_job_noise_counts_both_sides():
    assert se.noise_floor({'samples_ms': [1, 1.1], 'ref_samples_ms': [1, 1.2]}) == pytest.approx(.32)
    assert se.noise_floor({'samples_ms': [float('nan')]}) == float('inf')


def test_pypto_promotion_requires_confirmed_gain_and_three_jobs():
    task = load_task(repo_root() / 'tasks/swiglu-pypto')
    assert se.promotion_problem(task, best()) is None
    assert se.promotion_problem(task, {**best(), 'samples_ms': [1]})
    assert se.promotion_problem(task, {**best(), 'speedup': 1.01})
    assert se.promotion_problem(task, {**best(), 'ratio_basis': 'stale-baseline'})
    assert se.promotion_problem(task, {**best(), 'rejected_jobs': [{'reason': 'accuracy'}]})


def test_promoted_pypto_task_is_self_contained_and_source_does_not_cross(tmp_path):
    task = load_task(repo_root() / 'tasks/swiglu-pypto')
    dest = se.next_generation(task, best(), tmp_path / 'swiglu-g2', 1.05)
    promoted = load_task(dest)
    assert promoted.target.ratio == 1.05
    assert (promoted.leaderboard.ref / 'src/kernel.py').read_text() == best()['kernel']
    stub = (promoted.leaderboard.card / 'kernel.py').read_text()
    assert 'def swi_glu(' in stub and 'NotImplementedError' in stub
    assert 'swi_glu_impl_c1' not in stub
    lc = tomllib.loads((promoted.leaderboard.card / 'loop.toml.example').read_text())['loop']
    assert lc['oracle_mode'] == 'blackbox' and lc['confirm_jobs'] == 3
    assert (dest / 'promotion.json').is_file()


def test_promotion_failure_leaves_no_partial_task(tmp_path):
    task = load_task(repo_root() / 'tasks/swiglu-pypto')
    dest = tmp_path / 'bad'
    with pytest.raises(ValueError):
        se.next_generation(task, {**best(), 'kernel': 'no_entry=1'}, dest, 1.05)
    assert not dest.exists()


def test_sealed_cann_tree_contains_no_source_and_key_is_untracked(tmp_path):
    from forgekernel.cannref import seal_tree
    task = load_task(repo_root() / 'tasks/swiglu-pypto')
    dest = se.next_generation(task, best(), tmp_path / 'next', 1.05)
    tree = dest / 'leaderboard/ref'
    seal_tree(tree, 'kernel.py::swi_glu')
    assert not (tree / 'src').exists()
    payload = (tree / sealref.PAYLOAD).read_bytes()
    assert best()['kernel'].encode() not in payload
    key = tomllib.loads((tree / 'submit.local.toml').read_text())['env'][sealref.KEY_ENV]
    loader = sealref.closed_ref(template_dir() / 'eval')
    manifest = loader.unseal_bundle(payload, key)
    assert manifest['files']['kernel.py'] == best()['kernel']
    script = (tree / 'gpu_entry.sh').read_text()
    assert 'blackbox_guard "$*"' in script and 'exec env -u FK_REF_KEY' in script
    assert '*.local.toml' in (tree / '.gitignore').read_text()


def test_final_confirmation_overrules_a_target_stop(tmp_path, monkeypatch):
    # Test the acceptance predicate separately: insufficient or below-target
    # confirmations must not satisfy a PyPTO target, even after an optimistic stop.
    task = SimpleNamespace(language='pypto', target=SimpleNamespace(ratio=1.05, ms=None))
    c = {'median_ms': 1, 'speedup': 1.0449, 'requested': 3,
         'jobs': [{}, {}, {}], 'ref_samples_ms': [1.04, 1.05, 1.04]}
    monkeypatch.setattr(launch, 'confirmation', lambda run, task: c)
    assert not launch.confirmed_at_target(None, task)
    c['speedup'] = 1.06
    assert launch.confirmed_at_target(None, task)
    c['jobs'].pop()
    assert not launch.confirmed_at_target(None, task)
