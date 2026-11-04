#!/usr/bin/env python3
"""Run Codex until the harness accepts its report or the deadline expires.

Usage: codex-goal.py MODEL EFFORT TIMEOUT_S
FK_GOAL_TURN_DIR must contain the harness's spec.json (check, condition, reply).
Each turn resumes an explicit session ID; completion is checked outside Codex.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

OK_LINE = 'TURN-DELIVERABLE-OK'


def runner():
    spec = importlib.util.spec_from_file_location('fk_codex_run', Path(__file__).with_name('codex-run.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    # The protected records directory keeps an agent from replacing a predictable
    # temporary path with a symlink before this host-side write.
    directory = path.parent / 'codex-records'
    directory.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                     prefix=path.name + '-', delete=False) as handle:
        temp = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before descendants; still signal the whole group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def check_deliverable(command, deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, 0)
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            shell=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=remaining)
        return subprocess.CompletedProcess(command, proc.returncode, out, err)
    except BaseException:
        stop(proc)
        raise


def codex_turn(argv, prompt, deadline, events, on_session):
    """Spool events while running, so an interrupted turn keeps its session ID."""
    errors = events.with_suffix('.stderr')
    with events.open('w', encoding='utf-8') as out, errors.open('w', encoding='utf-8') as err:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                text=True, start_new_session=True)
        box = {}
        def feed():
            try:
                proc.communicate(prompt)
            except BaseException as exc:
                box['error'] = exc
        pump = threading.Thread(target=feed, daemon=True)
        pump.start()
        position, pending, known_session = 0, '', None
        try:
            while True:
                with events.open('r', encoding='utf-8') as reader:
                    reader.seek(position)
                    pending += reader.read()
                    position = reader.tell()
                lines = pending.split('\n')
                pending = lines.pop()
                for line in lines:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('type') == 'thread.started' and event.get('thread_id'):
                        session_id = str(event['thread_id'])
                        if session_id != known_session:
                            known_session = session_id
                            on_session(session_id)
                if not pump.is_alive():
                    break
                if time.time() >= deadline:
                    raise subprocess.TimeoutExpired(argv, 0)
                pump.join(timeout=min(0.1, max(0, deadline - time.time())))
            if 'error' in box:
                raise box['error']
        except BaseException:
            stop(proc)
            pump.join(timeout=3)
            raise
    return subprocess.CompletedProcess(argv, proc.returncode,
             events.read_text(encoding='utf-8'), errors.read_text(encoding='utf-8'))


def identity(prompt, spec, model, effort):
    policy = Path.cwd() / '.codex' / 'forgekernel-policy.json'
    data = dict(prompt=prompt, spec=spec, model=model, effort=effort, cwd=str(Path.cwd()),
                policy=policy.read_text(encoding='utf-8') if policy.is_file() else '',
                account={k: os.environ.get(k, '') for k in
                         ('CODEX_HOME', 'FK_CODEX_BASE_URL',
                          'FK_CODEX_KEY_ENV', 'FK_CODEX_BIN', 'OPENAI_BASE_URL')})
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def run(model, effort, timeout_s, prompt):
    started = time.time()
    deadline = float(os.environ.get('FK_GOAL_DEADLINE', started + timeout_s))
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('FK_GOAL_DEADLINE must be finite and positive')
    directory = os.environ.get('FK_GOAL_TURN_DIR')
    if not directory:
        raise ValueError('FK_GOAL_TURN_DIR must name a harness-prepared goal directory')
    turn = Path(directory).resolve()
    turn.mkdir(parents=True, exist_ok=True)
    spec = read_json(turn / 'spec.json')
    if not isinstance(spec, dict) or not spec.get('check') or not spec.get('condition'):
        raise ValueError('goal turn needs spec.json with check and condition')
    reply = Path(spec.get('reply') or turn / 'reply.md')
    key = identity(prompt, spec, model, effort)
    previous = read_json(turn / 'session.json', {})
    sid = previous.get('sid') if previous.get('identity') == key and previous.get('resumable') else None
    (turn / 'prompt.md').write_text(prompt, encoding='utf-8')
    records = turn / 'codex-records'
    records.mkdir(exist_ok=True)
    iteration = int(previous.get('iterations', 0)) if previous.get('identity') == key else 0
    outcome, detail, rc, transcript = 'fail', '', 1, ''
    resume_count = 0
    def remember(session_id):
        nonlocal sid
        sid = session_id
        write_json(turn / 'session.json', dict(sid=sid, identity=key, resumable=True,
                                             iterations=iteration, at=time.time()))
    try:
        cli = runner()
        cli.check_cli(os.environ.get("FK_CODEX_BIN", "codex"))
        if not sid:
            reply.unlink(missing_ok=True)
        while time.time() < deadline:
            iteration += 1
            token = uuid.uuid4().hex[:10]
            events = records / f'codex-{iteration:04d}-{token}.jsonl'
            final = records / f'codex-{iteration:04d}-{token}.reply'
            message = ('# FORGEKERNEL TURN TASK\n' + prompt + '\n\n# DELIVERABLE CONDITION\n'
                       + spec['condition'] + f'\nWrite your report into {reply}.\n')
            if detail:
                message += '\nThe harness has not accepted the deliverable:\n' + detail[-6000:]
            argv = cli.build_command(model, effort, output_path=final, resume_id=sid)
            resume_count += bool(sid)
            transcript = str(events)
            result = codex_turn(argv, message, deadline, events, remember)
            with (turn / 'result.json').open('a', encoding='utf-8') as history:
                history.write(json.dumps(dict(outcome='iteration', rc=result.returncode,
                    transcript=transcript, requested_model=model, sid=sid,
                    goal_iterations=iteration)) + '\n')
            event_error = cli.event_failure(result.stdout)
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'thread.started' and event.get('thread_id'):
                    sid = str(event['thread_id'])
            write_json(turn / 'session.json', dict(sid=sid, identity=key, resumable=bool(sid),
                                                  iterations=iteration, at=time.time()))
            if result.returncode or event_error:
                detail = (result.stderr + '\n' + event_error + '\n' + result.stdout[-3000:]).strip()
                rc = result.returncode or 1
                outcome = 'fail'
                break
            # The trusted spec is snapshotted before the agent starts: edits to it cannot
            # replace this acceptance command with an echo of the success marker.
            checked = check_deliverable(spec['check'], deadline)
            detail = (checked.stdout + '\n' + checked.stderr).strip()
            try:
                report = reply.read_text(encoding='utf-8')
            except OSError:
                report = ''
            if checked.returncode == 0 and any(line.startswith(OK_LINE)
                                               for line in checked.stdout.splitlines()) and report.strip():
                rc, outcome = 0, 'met'
                sys.stdout.write(report)
                break
            if not sid:
                detail = 'Codex returned no thread.started session ID; cannot continue safely. ' + detail
                break
            time.sleep(min(0.1, max(0, deadline - time.time())))
        else:
            rc, outcome, detail = 124, 'timeout', 'goal deadline reached'
    except subprocess.TimeoutExpired:
        rc, outcome, detail = 124, 'timeout', 'goal deadline reached'
    except Interrupted as exc:
        rc, outcome, detail = 128 + exc.signum, 'interrupted', 'goal interrupted by signal'
    except (OSError, ValueError, RuntimeError) as exc:
        rc, outcome, detail = 2, 'fail', str(exc)
    finally:
        write_json(turn / 'session.json', dict(sid=sid, identity=key,
                   resumable=bool(sid) and outcome in ('fail', 'interrupted', 'timeout'),
                   iterations=iteration, at=time.time()))
        with (turn / 'result.json').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(outcome=outcome, rc=rc, detail=detail[-6000:], sid=sid,
                    backend='codex', requested_model=model, goal_iterations=iteration,
                    resume_count=resume_count, wall_s=round(time.time() - started, 3),
                    transcript=transcript)) + '\n')
    if rc:
        sys.stderr.write(f'codex-goal: {outcome}: {detail}\n')
    return rc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        sys.stderr.write('usage: codex-goal.py MODEL EFFORT TIMEOUT_S\n')
        return 2
    def interrupted(signum, frame):
        raise Interrupted(signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        timeout = float(argv[2])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        return run(argv[0], argv[1], timeout, sys.stdin.read())
    except (ValueError, OSError, RuntimeError) as exc:
        sys.stderr.write(f'codex-goal: {exc}\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
