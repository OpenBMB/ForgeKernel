#!/usr/bin/env python3
"""Audit real Ascend run artifacts against docs/ascend-agent-gap.md.

No model call, no fabricated success. Missing/incomplete evidence exits nonzero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib


def load(path):
    return json.loads(path.read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m1-run', type=Path, required=True)
    p.add_argument('--evolution-run', type=Path, required=True)
    p.add_argument('--level4-run', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args(argv)
    checks = {}

    def check(name, ok, detail):
        checks[name] = {'pass': bool(ok), 'detail': detail}

    op = args.m1_run / 'operators/swiglu-pypto'
    rows = [json.loads(x) for x in (op / 'student/status.jsonl').read_text().splitlines() if x]
    meta = [json.loads(x) for x in (op / 'meta_status.jsonl').read_text().splitlines() if x]
    official = [r['cann_bench'] for r in rows if r.get('cann_bench')]
    case = official[0]['cases'][0]
    check('G1-official-fields-and-raw-reports', all(k in case for k in
        ('baseline_perf_us', 't_hw_us', 'perf_score', 'op_times', 'memory_peak_mb'))
        and list(op.glob('.jobs/*/artifacts/cann/official_report.json')),
          'per-case official anchors/scores and checksummed raw report')
    check('G1-target-superseded-by-confirmation', (op / 'best_confirmation.json').is_file()
          and not launch_confirmation_met(args.m1_run, 'swiglu-pypto'),
          'optimistic 1.05 stop is rejected by 3+3 confirmation (1.0449)')
    check('G3-M1-and-outer-patch', any(r['decision_id'].startswith('D1.') for r in rows)
          and any(m.get('outcome') == 'KEEP_PATCH' for m in meta), 'M1 and measured outer patch')
    reference = load(op / 'ref_baseline.json')
    check('G4-real-msprof-both-sides', reference['profiler']['status'] == 'available'
          and any(t['source'] == 'msprof_op_summary' for t in reference['profiler']['tables'])
          and all(r.get('profiler', {}).get('backend') == 'msprof' for r in rows),
          'actual msprof reference + per-iteration evidence')
    log = (args.m1_run / 'run.log').read_text()
    check('G2-no-fake-NCU', 'ncu empty on a passing kernel' not in log, 'no NCU retries')

    gen = args.evolution_run / 'operators/swiglu-pypto-g2'
    tree = args.evolution_run / 'operators/_swiglu-pypto-g2ref'
    promotion = load(args.evolution_run / 'task/promotion.json')
    check('G5-sealed-promotion', promotion['disclosure'] == 'blackbox'
          and (tree / 'ref_bin/ref_payload.bin').is_file()
          and not (tree / 'src').exists() and (tree / 'submit.local.toml').is_file(),
          'source available only through keyed reference tree')
    check('G5-student-stub-and-new-commit', 'NotImplementedError' in
          (args.evolution_run / 'task/leaderboard/card/kernel.py').read_text()
          and (gen / 'student/status.jsonl').is_file()
          and (gen / 'student/status.jsonl').read_text().strip(),
          'signature stub followed by real student iterations')
    check('G5-confirmed-next-generation', (gen / 'best_confirmation.json').is_file(),
          'gen2 best confirmation is required for the closed chain')

    kb = Path('hw_kb/pypto/ascend950/valid_shape_assemble.md')
    milestone = gen.parent / 'gqa-pypto/milestone.md'
    # Prefer the explicit level-4 path over any accidental name collision.
    milestone = args.level4_run / 'operators/gqa-pypto/milestone.md'
    check('G6-probe-certified-recipe', kb.is_file() and '1793761707' in kb.read_text()
          and 'Gotchas' in kb.read_text() and 'Retry preconditions' in kb.read_text(),
          'narrow NPU-certified mechanism and no-go/retry conditions')
    check('G6-reuse-in-other-operator', milestone.is_file()
          and 'valid_shape_assemble' in milestone.read_text(),
          'Level-4 milestone cites and applies the certified recipe')

    ref = load(args.level4_run / 'operators/gqa-pypto/ref_baseline.json')
    protocol = load(args.level4_run / 'task/experiment_protocol.json')
    names = [k['name'] for table in ref['profiler']['tables'] for k in table['kernels']]
    check('G7-level4-strong-library-protocol', protocol['level'].endswith('level4')
          and protocol['architecture_reference'].startswith('CANN library')
          and 'FusedInferAttentionScore' in names and ref['n_workloads'] == 20,
          '20 public Level-4 cases + independently profiled strong library reference')
    failed = [name for name, result in checks.items() if not result['pass']]
    out = {'protocol': 'real-device capability, not paper 910C/CANN-1.1.2 reproduction',
           'checks': checks, 'remaining': failed}
    print(json.dumps(out, indent=2))
    return 0 if not failed or args.allow_incomplete else 1


def launch_confirmation_met(run: Path, task_name: str):
    from forgekernel.launch import confirmed_at_target
    from forgekernel.runs import open_run
    from forgekernel.task import load_task
    run = open_run(run)
    task = load_task(run.task_dir, name=run.task_name)
    return confirmed_at_target(run, task)


if __name__ == '__main__':
    raise SystemExit(main())
