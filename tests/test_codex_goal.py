"""Run the actual goal leg with a fake Codex process and trusted external check."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'codex-goal.py'

STUB = r'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
if '--help' in sys.argv:
    print('--ignore-user-config --ignore-rules --strict-config')
    raise SystemExit
p = sys.stdin.read()
turn = Path(os.environ['FK_GOAL_TURN_DIR'])
count = int((turn / 'count').read_text()) + 1 if (turn / 'count').exists() else 1
(turn / 'count').write_text(str(count))
with (turn / 'argv.jsonl').open('a') as f:
    f.write(json.dumps(sys.argv) + '\n')
print(json.dumps({'type':'thread.started','thread_id':'session-123'}), flush=True)
mode = os.environ.get('STUB_MODE','ok')
if mode == 'sleep':
    (turn / 'pid').write_text(str(os.getpid()))
    time.sleep(60)
if mode == 'error' or (mode == 'retry' and count == 1):
    print('rate limit exceeded', file=sys.stderr)
    raise SystemExit(7)
reply = Path(json.loads((turn / 'spec.json').read_text())['reply'])
if mode != 'no-write':
    reply.write_text('measured report')
if mode == 'tamper':
    spec = json.loads((turn / 'spec.json').read_text())
    spec['check'] = 'echo TURN-DELIVERABLE-OK'
    (turn / 'spec.json').write_text(json.dumps(spec))
if mode not in ('never', 'tamper') and (mode != 'twice' or count >= 2):
    (turn / 'accepted').touch()
if mode == 'reconnect':
    print(json.dumps({'type':'error','message':'stream disconnected, reconnecting'}))
if mode == 'event-error':
    print(json.dumps({'type':'turn.failed','error':{'message':'bad request'}}))
print(json.dumps({'type':'turn.completed'}))
'''


@pytest.fixture
def leg(tmp_path):
    turn = tmp_path / 'turn'
    turn.mkdir()
    ws = tmp_path / 'workspace'
    (ws / '.codex').mkdir(parents=True)
    (ws / '.codex' / 'forgekernel-policy.json').write_text(json.dumps(dict(
        schema=1, backend='codex', config={'default_permissions':'fk',
                                          'permissions.fk.filesystem':{':workspace':'write'}})))
    stub = tmp_path / 'codex-stub'
    stub.write_text(STUB)
    stub.chmod(0o755)
    check = tmp_path / 'check.py'
    check.write_text("import sys\nfrom pathlib import Path\np=Path(sys.argv[1])\nok=(p/'accepted').exists()\nprint('TURN-DELIVERABLE-OK' if ok else 'TURN-DELIVERABLE-MISSING: acceptance ledger empty')\nsys.exit(0 if ok else 1)\n")
    import shlex
    spec = dict(reply=str(turn / 'reply.md'), condition='Produce a measured deliverable.',
                check=' '.join(shlex.quote(x) for x in (sys.executable, str(check), str(turn))))
    (turn / 'spec.json').write_text(json.dumps(spec))
    env = {k:v for k,v in os.environ.items() if not k.startswith(('FK_GOAL_', 'FK_CODEX_'))}
    env.update(FK_CODEX_BIN=str(stub), FK_GOAL_TURN_DIR=str(turn))
    return turn, ws, env


def invoke(leg, mode='ok', timeout='10', model='test-model'):
    turn, ws, env = leg
    return subprocess.run([sys.executable, str(SCRIPT), model, 'high', timeout],
                          input='build the kernel', text=True, capture_output=True,
                          cwd=ws, env=dict(env, STUB_MODE=mode), timeout=20)


def result(leg):
    return json.loads((leg[0] / 'result.json').read_text().splitlines()[-1])


def test_unmet_deliverable_resumes_exact_session_and_checks_again(leg):
    proc = invoke(leg, 'twice')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == 'measured report'
    calls = [json.loads(x) for x in (leg[0] / 'argv.jsonl').read_text().splitlines()]
    assert len(calls) == 2 and 'resume' not in calls[0]
    assert 'resume' in calls[1] and 'session-123' in calls[1] and '--last' not in calls[1]
    for call in calls:
        assert 'approval_policy="never"' in call
        assert '--model' in call and '--ignore-user-config' in call
    assert result(leg)['outcome'] == 'met'
    assert Path(result(leg)['transcript']).is_file()


def test_check_is_snapshotted_and_self_report_does_not_accept(leg):
    proc = invoke(leg, 'tamper', '0.7')
    assert proc.returncode == 124 and not proc.stdout
    assert result(leg)['outcome'] == 'timeout'


def test_empty_ledger_never_accepts_even_with_reply(leg):
    proc = invoke(leg, 'never', '0.7')
    assert proc.returncode == 124 and not proc.stdout
    assert result(leg)['outcome'] == 'timeout'


def test_deadline_kills_cli_process(leg):
    proc = invoke(leg, 'sleep', '0.7')
    assert proc.returncode == 124
    pid = int((leg[0] / 'pid').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_failure_remains_failure_and_next_attempt_resumes(leg):
    first = invoke(leg, 'retry')
    assert first.returncode == 7 and 'rate limit' in first.stderr and not first.stdout
    assert invoke(leg, 'retry').returncode == 0
    calls = [json.loads(x) for x in (leg[0] / 'argv.jsonl').read_text().splitlines()]
    assert 'resume' in calls[1]


def test_changing_model_starts_new_session(leg):
    assert invoke(leg, 'error').returncode == 7
    assert invoke(leg, model='other-model').returncode == 0
    calls = [json.loads(x) for x in (leg[0] / 'argv.jsonl').read_text().splitlines()]
    assert 'resume' not in calls[1]


def test_event_failure_with_zero_exit_is_failure(leg):
    assert invoke(leg, 'event-error').returncode == 1
    assert result(leg)['outcome'] == 'fail'


def test_missing_policy_refuses_to_run_agent(leg):
    (leg[1] / '.codex' / 'forgekernel-policy.json').unlink()
    proc = invoke(leg)
    assert proc.returncode == 2 and 'policy' in proc.stderr
    assert not (leg[0] / 'count').exists()


def test_deadline_session_can_resume_after_timeout(leg):
    assert invoke(leg, 'sleep', '0.7').returncode == 124
    saved = json.loads((leg[0] / 'session.json').read_text())
    assert saved['sid'] == 'session-123' and saved['resumable']
    assert invoke(leg).returncode == 0
    calls = [json.loads(x) for x in (leg[0] / 'argv.jsonl').read_text().splitlines()]
    assert 'resume' in calls[1]


def test_signal_kills_process_and_preserves_resumption(leg):
    turn, ws, env = leg
    proc = subprocess.Popen([sys.executable, str(SCRIPT), 'test-model', 'high', '30'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=ws, env=dict(env, STUB_MODE='sleep'))
    proc.stdin.write('build the kernel')
    proc.stdin.close()
    proc.stdin = None
    try:
        end = time.time() + 5
        while not (turn / 'pid').exists() and time.time() < end:
            time.sleep(0.05)
        # Allow the event spooler to persist thread.started before interrupting.
        while not (turn / 'session.json').exists() and time.time() < end:
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=5)
        assert proc.returncode == 143 and not out
        assert result(leg)['outcome'] == 'interrupted'
        with pytest.raises(ProcessLookupError):
            os.kill(int((turn / 'pid').read_text()), 0)
        assert json.loads((turn / 'session.json').read_text())['resumable']
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_goal_deadline_overrides_ordinary_agent_turn_timeout(leg):
    # A reference generation budget is independent of the normal per-turn timeout.
    leg[2]['FK_GOAL_DEADLINE'] = str(time.time() + 5)
    assert invoke(leg, timeout='0.001').returncode == 0


def test_fresh_identity_cannot_deliver_a_stale_reply(leg):
    (leg[0] / 'reply.md').write_text('old report from another task')
    (leg[0] / 'accepted').touch()
    proc = invoke(leg, 'no-write', '0.7')
    assert proc.returncode == 124 and not proc.stdout


def test_cli_capability_failure_overwrites_stale_leg_outcome(leg):
    (leg[0] / 'result.json').write_text(json.dumps(dict(outcome='met', rc=0)) + '\n')
    Path(leg[2]['FK_CODEX_BIN']).write_text('#!/usr/bin/env python3\nprint("unsupported")\n')
    proc = invoke(leg)
    assert proc.returncode == 2 and result(leg)['outcome'] == 'fail'
    assert result(leg)['rc'] == 2


def test_every_iteration_transcript_is_retained_for_harvest(leg):
    assert invoke(leg, 'twice').returncode == 0
    records = [json.loads(line) for line in (leg[0] / 'result.json').read_text().splitlines()]
    transcripts = {r['transcript'] for r in records if r.get('transcript')}
    assert len(transcripts) == 2
    assert all(Path(p).is_file() for p in transcripts)


def test_reconnected_stream_with_completion_still_accepts_deliverable(leg):
    proc = invoke(leg, 'reconnect')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == 'measured report'
    assert result(leg)['outcome'] == 'met'
