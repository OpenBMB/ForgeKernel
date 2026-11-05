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

from forgekernel import profiling
from forgekernel.baseline_cli import parse_ref_stdout
from forgekernel.evidence import archive_job_artifacts
from forgekernel.inner.trajectory import IterationRecord, append_record, load_records, build_digest
from forgekernel.loop import driver
from forgekernel.loop.parse import parse_eval
from forgekernel.taskcard import evaluation_timeout
from forgekernel.task import load_task
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
    assert digest['profiler'][0]['backend'] == 'msprof'
    assert digest['profiler'][0]['timing_basis'] == 'cann_trace_view'
    baseline = parse_ref_stdout(text)
    assert baseline['ms_device'] == pytest.approx(.2)
    assert baseline['cann_bench'] == ev.cann_bench
    assert ev.ref_device_ms is None and ev.vs_ref_device is None
    assert not any(k.startswith('ref_') for k in result['performance'])


def test_kernel_details_and_trace_view_are_both_retained_with_explicit_selection():
    report = enhanced()
    report['operators'][0]['cases'][0]['elapsed_us'] = 123
    report['operators'][0]['cases'][0]['op_times']['device_kernels'] = {'candidate': 123}
    result = _adapter().report_to_eval(report, config={'timing_basis': 'cann_trace_view'})
    assert result['performance']['device_window_basis'] == 'cann_trace_view'
    assert result['cann_bench']['timing_basis'] == result['performance']['device_window_basis']
    assert result['performance']['workloads'][0]['device_median_ms'] == .1
    assert result['cann_bench']['cases'][0]['official_elapsed_us'] == 100
    assert result['cann_bench']['cases'][0]['op_times']['device_kernels'] == {'candidate': 123}
    vendor = _adapter().report_to_eval(report, config={'timing_basis': 'cann_kernel_details'})
    assert vendor['performance']['device_window_basis'] == 'cann_kernel_details'
    assert vendor['cann_bench']['timing_basis'] == vendor['performance']['device_window_basis']


def test_evaluator_failure_preserves_official_case_error():
    report = enhanced()
    report['operators'][0]['cases'][0].update(status='failed', error_msg='precision_mismatch')
    report['operators'][0]['cases'][0]['accuracy']['passed'] = False
    adapter = _adapter()
    result = adapter.report_to_eval(report)
    adapter.evaluator_failure(result, 4)
    assert 'precision_mismatch' in result['correctness']['error']
    assert 'exited 4' in result['correctness']['error']
    assert not result['performance']


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


def test_full_case_job_timeout_is_configurable_and_not_per_case(tmp_path):
    (tmp_path / 'loop.toml').write_text('[loop]\neval_timeout_s = 7200\n')
    task = load_task(repo_root() / 'tasks/swiglu-pypto')
    assert evaluation_timeout(tmp_path, task, 1800) == 7200
    assert evaluation_timeout(tmp_path, task, 1800) != 1740
    with pytest.raises(ValueError):
        (tmp_path / 'loop.toml').write_text('[loop]\neval_timeout_s = 0\n')
        evaluation_timeout(tmp_path, task, 1800)


def test_timeout_is_not_mislabeled_as_no_json_or_numeric_failure():
    log = '[gpu_entry.sh] timeout=1800s\n[gpu_entry.sh] KILLED: task exceeded 1800s timeout'
    ev = parse_eval(log)
    assert not ev.correctness
    assert 'did not complete' in ev.error and 'no correctness or performance claim' in ev.error


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
    assert profiling.backend_for('pypto') == profiling.PYPTO_DEFAULT_BACKEND == 'msprof'
    assert profiling.backend_for('pypto', 'cann_trace_view') == 'cann_trace_view'
    assert profiling.backend_for('cutedsl', ncu_available=False) == 'none'
    with pytest.raises(ValueError):
        profiling.backend_for('pypto', 'ncu')
    with pytest.raises(ValueError):
        profiling.backend_for('cutedsl', 'msprof')
    assert profiling.channel_violations('[VERIFY: ncu-derived]', 'msprof')
    assert not profiling.channel_violations('[VERIFY: msprof-direct]', 'msprof')
    assert profiling.channel_violations('[VERIFY: msprof-direct]', 'cann_trace_view')
    assert profiling.evidence_digest(
        {'backend': 'msprof', 'timing_basis': 'cann_kernel_details', 'status': 'available'}
    )['timing_basis'] == 'cann_kernel_details'


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


def _adapter_at(path: Path):
    spec = importlib.util.spec_from_file_location('cann_bench_card', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PYPTO_TASKS = ('gqa-pypto', 'swiglu-pypto', 'swiglu-pypto-g2')
SYNCED_CARD_FILES = (('eval/cann_bench.py', 'eval/cann_bench.py'),
                     ('eval/ascend_profiler.py', 'eval/ascend_profiler.py'),
                     ('gpu_entry.sh', 'pypto_gpu_entry.sh'))


def test_shipped_pypto_cards_are_synced_with_the_card_template():
    """Frozen card trees are what jobs execute; the template is not an overlay.

    A template-only edit leaves every installed task on the old adapter (that is how
    missing trace_view kept becoming correctness=False and the profiler fields stayed
    absent). scripts/sync_pypto_cards.py is the explicit maintenance step, and this
    invariant fails the build when a later change skips it.
    """
    template = repo_root() / 'src/forgekernel/card_template'
    for task in PYPTO_TASKS:
        for side in ('card', 'ref'):
            tree = repo_root() / f'tasks/{task}/leaderboard/{side}'
            for rel, source in SYNCED_CARD_FILES:
                assert (tree / rel).read_bytes() == (template / source).read_bytes(), (
                    f'{tree / rel} is stale; run scripts/sync_pypto_cards.py')


@pytest.mark.parametrize('task', ['gqa-pypto', 'swiglu-pypto'])
def test_shipped_card_adapter_reports_routing_and_measurement(task):
    card = repo_root() / f'tasks/{task}/leaderboard/card'
    adapter = _adapter_at(card / 'eval/cann_bench.py')
    config = json.loads((card / 'eval/cann_task.json').read_text())
    result = adapter.report_to_eval(enhanced(), config=config)
    assert result['correctness']['pass'] is True
    assert result['measurement']['status'] == 'available'
    assert result['profiler']['backend'] == config['profiler_backend'] == 'msprof'
    assert result['profiler']['timing_basis'] == result['performance']['device_window_basis']
    assert result['cann_bench']['timing_basis'] == result['performance']['device_window_basis']


def test_card_backend_and_basis_resolution_matches_the_control_plane():
    _adapter()
    import ascend_profiler
    assert ascend_profiler.DEFAULT_PROFILER_BACKEND == profiling.PYPTO_DEFAULT_BACKEND
    assert ascend_profiler.resolve_backend({}) == 'msprof'
    assert ascend_profiler.resolve_backend({'profiler_backend': 'auto'}) == 'msprof'
    assert ascend_profiler.resolve_backend(
        {'profiler_backend': 'cann_trace_view'}) == 'cann_trace_view'
    with pytest.raises(ValueError, match='ncu'):
        ascend_profiler.resolve_backend({'profiler_backend': 'ncu'})
    assert ascend_profiler.resolve_timing_basis({}) == 'cann_trace_view'
    assert ascend_profiler.resolve_timing_basis(
        {'perf_metric_strategy': 'kernel_details'}) == 'cann_kernel_details'
    assert ascend_profiler.resolve_timing_basis(
        {'perf_metric_strategy': 'kernel_details',
         'timing_basis': 'cann_trace_view'}) == 'cann_trace_view'
    with pytest.raises(ValueError, match='timing basis'):
        ascend_profiler.resolve_timing_basis({'timing_basis': 'wall_clock'})


def test_a_missing_declared_timing_window_is_measurement_unavailable_not_a_correctness_fail():
    report = enhanced()
    for case in report['operators'][0]['cases']:
        case['op_times'] = {}
    result = _adapter().report_to_eval(report, config={'profiler_backend': 'msprof'})
    # The official checker passed every case; only the declared trace_view window is
    # absent. Reporting that as correctness=False is what turned a harness defect into
    # three student REVERTs.
    assert result['correctness']['pass'] is True
    assert result['performance'] == {}
    assert result['measurement']['status'] == 'unavailable'
    assert result['measurement']['timing_basis'] == 'cann_trace_view'
    assert result['cann_bench']['timing_basis'] == 'cann_trace_view'
    assert 'trace_view.aicore_e2e' in result['measurement']['reason']
    ev = parse_eval('=== JSON ===\n' + json.dumps(result))
    assert ev.correctness is True and ev.measurement['status'] == 'unavailable'


class _Done:
    def __init__(self, stdout=''):
        self.stdout, self.stderr, self.returncode = stdout, '', 0


@pytest.mark.parametrize('config, expected_basis', [
    ({}, 'cann_trace_view'),
    ({'profiler_backend': 'msprof', 'perf_metric_strategy': 'kernel_details',
      'timing_basis': 'cann_kernel_details'}, 'cann_kernel_details'),
])
def test_evaluate_resolves_omitted_backend_and_keeps_report_metadata(
        tmp_path, monkeypatch, config, expected_basis):
    """_evaluate is the path a real job runs: resolution + collect must agree."""
    adapter = _adapter()
    import ascend_profiler
    monkeypatch.setattr(ascend_profiler.shutil, 'which', lambda name: None)

    cann = tmp_path / 'cann'
    cann.mkdir()
    workspace = tmp_path / 'ws'
    (workspace / 'eval').mkdir(parents=True)
    full = {'operator': 'SwiGlu', 'task_dir': 'tasks/level1/swi_glu', **config}
    (workspace / 'eval/cann_task.json').write_text(json.dumps(full))
    monkeypatch.setenv('CANN_BENCH_DIR', str(cann))

    monkeypatch.setattr(adapter.subprocess, 'run',
                        lambda cmd, **kw: _Done('deadbeef\n' if cmd[:2] == ['git', '-C'] else ''))

    def fake_logged(cmd, *, cwd, env, log, budget_s=None):
        reports = Path(cmd[cmd.index('--reports-dir') + 1])
        (reports / 'run_eval_1.json').write_text(json.dumps(enhanced()))
        log.write_text('evaluator ok\n')
        return adapter.RunOutcome(rc=0, interrupted=False, reason='', seconds=12.5)

    monkeypatch.setattr(adapter, '_run_logged', fake_logged)
    result = adapter._evaluate(workspace, full)
    assert result['measurement']['status'] == 'available'
    assert result['measurement']['backend'] == 'msprof'
    assert result['profiler']['backend'] == 'msprof'
    assert result['profiler']['timing_basis'] == expected_basis
    # Collection evidence merged in, report metadata not replaced by it.
    assert result['profiler']['reason'] == 'msprof executable unavailable'
    assert result['profiler']['cases'][0]['trace_view']['aicore_e2e'] == 100
    assert result['performance']['device_window_basis'] == expected_basis
    assert result['cann_bench']['timing_basis'] == result['performance']['device_window_basis']
    assert (workspace / '.cann_results').is_dir()
    assert list((workspace / '.cann_results').glob('*/forge_eval.json'))
    # Cost is written on a normal run too, not only on the failure paths.
    assert result['cann_bench']['cost']['evaluator_seconds'] == 12.5
    assert result['cann_bench']['cost']['cases_requested'] == 'all'
    assert result['cann_bench']['case_scope'] is None
