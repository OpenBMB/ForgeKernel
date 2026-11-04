from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest

from forgekernel import agents, profiling
from forgekernel.baseline_cli import parse_ref_stdout
from forgekernel.evidence import archive_job_artifacts
from forgekernel.inner.trajectory import IterationRecord, append_record, load_records, build_digest
from forgekernel.loop import driver
from forgekernel.loop.parse import parse_eval
from forgekernel.paths import repo_root
from test_pypto import _adapter, _report


def enhanced():
    report = _report()
    report['overall_score'] = 71.0
    op = report['operators'][0]
    op.update(compilation_score=20, function_score=30, performance_score=21)
    for case in op['cases']:
        case.update(baseline_perf_us=50, t_hw_us=8, speedup=0.5, perf_score=0.4,
                    memory_peak_mb=None, op_times={'trace_view': {
                        'aicore_e2e': case['elapsed_us'], 'aicore_e2e_jitter': 1.2,
                        'aicpukernel_gap': 0}})
    return report


def test_official_evidence_survives_eval_trajectory_and_baseline(tmp_path):
    result = _adapter().report_to_eval(enhanced())
    text = '=== JSON ===\n' + json.dumps(result)
    ev = parse_eval(text)
    rec = IterationRecord(0, 'D1.1', 'tile', ev.correctness, 'KEEP',
                          device_median_ms=ev.device_median_ms, profiler=ev.profiler,
                          cann_bench=ev.cann_bench, workloads=ev.workloads, n_workloads=2)
    status = tmp_path / 'status.jsonl'
    append_record(status, rec)
    loaded = load_records(status)[0]
    assert loaded.workloads[0]['cann_bench']['t_hw_us'] == 8
    assert loaded.cann_bench['scores']['function_score'] == 30
    assert loaded.cann_bench['cases'][0]['memory_peak_mb'] is None
    digest = build_digest([loaded], {'value': .1})
    assert digest['cann_bench'][0]['overall_score'] == 71
    assert digest['profiler'][0]['backend'] == 'cann_trace_view'
    baseline = parse_ref_stdout(text)
    assert baseline['ms_device'] == pytest.approx(.2)
    assert baseline['cann_bench'] == ev.cann_bench
    assert ev.ref_device_ms is None and ev.vs_ref_device is None
    assert not any(k.startswith('ref_') for k in result['performance'])


def test_failed_official_report_cannot_supply_reference_time():
    report = enhanced()
    report['operators'][0]['cases'][0]['accuracy']['passed'] = False
    result = _adapter().report_to_eval(report)
    assert not result['correctness']['pass']
    assert result['cann_bench']['cases'][0]['t_hw_us'] == 8
    assert parse_ref_stdout(json.dumps(result)) == {}


def artifact(name='cann/report.json', data=b'{"official":true}'):
    return 'FORGE-ARTIFACT: ' + json.dumps({
        'name': name, 'encoding': 'zlib+base64',
        'sha256': hashlib.sha256(data).hexdigest(),
        'data': base64.b64encode(zlib.compress(data)).decode()})


def test_artifacts_are_exact_and_checksummed(tmp_path):
    raw = b'{\n "report": [1,2], "null": null\n}\n'
    manifest = archive_job_artifacts(artifact(data=raw), tmp_path)
    assert (tmp_path / 'cann/report.json').read_bytes() == raw
    assert manifest[0]['sha256'] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'cann/../../escape', 'bad\\path'])
def test_artifact_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        archive_job_artifacts(artifact(name), tmp_path)


def test_artifact_corruption_and_duplicates_are_refused(tmp_path):
    obj = json.loads(artifact().split(': ', 1)[1])
    obj['sha256'] = 'bad'
    with pytest.raises(ValueError, match='checksum'):
        archive_job_artifacts('FORGE-ARTIFACT: ' + json.dumps(obj), tmp_path)
    with pytest.raises(ValueError):
        archive_job_artifacts(artifact() + '\n' + artifact(), tmp_path)


def test_msprof_csv_has_units_scope_and_no_fabricated_ncu():
    adapter = _adapter()
    import ascend_profiler
    result = ascend_profiler.parse_kernel_csv(
        'Name,Type,Duration(us),Block Num\nfoo,PyPTO,10,32\nfoo,PyPTO,20,32\n')
    assert result['status'] == 'available'
    assert result['kernels'][0]['launches'] == 2
    assert result['kernels'][0]['mean_us'] == 15
    assert 'not per-call' in result['scope']
    assert 'time_ns' not in result and 'tensor_pct' not in result
    assert ascend_profiler.parse_kernel_csv('wrong,column\nx,y')['status'] == 'unavailable'


def test_no_msprof_is_explicit_unavailable(tmp_path, monkeypatch):
    _adapter()
    import ascend_profiler
    monkeypatch.setattr(ascend_profiler.shutil, 'which', lambda name: None)
    evidence, files = ascend_profiler.collect(tmp_path, 'msprof')
    assert evidence['status'] == 'unavailable' and not files


def test_pypto_never_submits_ncu_even_when_kernel_passes():
    ctx = SimpleNamespace(language='pypto', profiler_backend='msprof')
    assert driver._ncu_profiler(ctx, 'main', correctness=True) == {}
    ctx = SimpleNamespace(language='cutedsl', profiler_backend='none')
    assert driver._ncu_profiler(ctx, 'main', correctness=True) == {}


def test_backend_and_channel_contract():
    assert profiling.backend_for('pypto') == 'cann_trace_view'
    assert profiling.backend_for('cutedsl', ncu_available=False) == 'none'
    with pytest.raises(ValueError):
        profiling.backend_for('pypto', 'ncu')
    assert profiling.channel_violations('[VERIFY: ncu-derived]', 'msprof')
    assert not profiling.channel_violations('[VERIFY: msprof-direct]', 'msprof')
    assert profiling.channel_violations('[VERIFY: msprof-direct]', 'cann_trace_view')


def test_teacher_env_model_routing_does_not_put_key_in_command():
    spec = SimpleNamespace(backend='claude', provider='native', model='student-model',
                           effort='', timeout_s=100)
    env = {'TEACHER_MODEL': 'k3.1pro', 'TEACHER_BASEURL': 'https://teacher.invalid',
           'TEACHER_API_KEY': 'never-log-this-key'}
    teacher = agents.agent_command(spec, 'teacher', env)
    student = agents.agent_command(spec, 'student', env)
    assert 'k3.1pro' in teacher and 'FK_PROVIDER_KEY_ENV=TEACHER_API_KEY' in teacher
    assert 'never-log-this-key' not in teacher
    assert 'k3.1pro' not in student and 'student-model' in student
    with pytest.raises(ValueError, match='together'):
        agents.agent_command(spec, 'teacher', {'TEACHER_MODEL': 'k3.1pro'})


def test_all_pypto_overlays_are_operator_agnostic():
    for name in ('student', 'distill', 'distill_blackbox', 'recon', 'diagnose', 'patch',
                 'm0_audit', 'kb_recipe', 'kb_harvest'):
        text = (repo_root() / f'meta/pypto/{name}_prompt.md').read_text()
        assert 'swi_glu' not in text and 'all eight' not in text


def test_ascend_experiment_edits_are_atomic():
    from forgekernel.skills.ascend_probe import apply_edits
    assert apply_edits('abc def', [{'old': 'abc', 'new': 'def'},
                                  {'old': 'def', 'new': 'xyz'}]) == 'def xyz'
    with pytest.raises(ValueError, match='overlap'):
        apply_edits('abcdef', [{'old': 'abc', 'new': ''}, {'old': 'bcde', 'new': ''}])
