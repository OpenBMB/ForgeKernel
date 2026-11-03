"""Codex configuration through the public launch/resume/export pipeline.

GPU measurements are stood in for; CLI subprocess and native sandbox contracts
are covered separately by the CLI contract tests and native sandbox smoke checks.
"""
from __future__ import annotations

import json
import subprocess
import sys
import shlex
import threading
import time

import pytest

from forgekernel import launch
from forgekernel.loop import cli
from forgekernel.runs import list_runs
from forgekernel.task import load_task
from test_launch import (REF_TOML, TASK_TOML, _Teacher, _git, _loop_ends,
                         _start, _task, repo)

CODEX = '\n[agent]\nbackend="codex"\nmodel="test-codex-model"\n'


@pytest.mark.parametrize("reference", [REF_TOML, REF_TOML + 'blackbox=true\n', TASK_TOML])
def test_codex_modes_resume_export_and_next_generation(repo, monkeypatch, capsys, reference):
    task_dir = _task(repo, toml=reference + CODEX)
    teacher = _Teacher()
    calls = []
    monkeypatch.setattr(cli.RealBackend, "run_agent", teacher.method)
    monkeypatch.setattr(launch, "calibrate", lambda *a: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(cli, "main", _loop_ends(calls.append))
    assert _start(task_dir) == 0
    run = list_runs()[0]
    task = load_task(run.task_dir, name=run.task_name)
    assert task.teacher.backend == task.student.backend == "codex"
    for flag in ("--agent-cmd", "--teacher-agent-cmd"):
        command = calls[0][calls[0].index(flag) + 1]
        assert "agent-codex.sh" in command and "test-codex-model" in command
    assert launch.cmd_run([str(run.dir)]) == 0
    assert len(teacher.prompts) == 1
    assert "--allow-preexisting-artifacts" in calls[-1]
    kernel = "def run(x, w):\n    return x\n"
    (run.ws / "kernel.py").write_text(kernel)
    subprocess.run(["git", "-C", str(run.ws), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qam", "best"], check=True, capture_output=True)
    (run.operators_dir / task.name / "best_confirmation.json").write_text(json.dumps({
        "sha": _git(run.ws, "rev-parse", "HEAD").strip(), "window": "device",
        "median_ms": 0.01, "median_ratio": 1.3}))
    capsys.readouterr()
    assert launch.cmd_export([str(run.dir)]) == 0
    assert capsys.readouterr().out == kernel
    # Promotion carries backend/model configuration into the next source task.
    sys.path.insert(0, str(launch.repo_root() / "scripts"))
    try:
        import self_evolve
        dest = self_evolve.next_generation(task, {"kernel": kernel},
                                           repo / "tasks" / "rmsnorm-g2", 1.05)
        promoted = load_task(dest)
        assert promoted.mode == "source" and promoted.student.backend == "codex"
        assert promoted.teacher.model == "test-codex-model"
    finally:
        sys.path.pop(0)


def test_interrupted_agent_turn_ends_tool_processes(tmp_path, monkeypatch):
    """An interruption must reach tools below the command shell, not only the shell."""
    ready, ended = tmp_path / "ready", tmp_path / "ended"
    worker = tmp_path / "worker.py"
    worker.write_text("import signal, sys, time\nfrom pathlib import Path\n"
                      f"signal.signal(signal.SIGTERM, lambda *a: (Path({str(ended)!r}).touch(), sys.exit(0)))\n"
                      f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n")
    original_join = threading.Thread.join
    interrupted = False
    def interrupt_after_start(thread, timeout=None):
        nonlocal interrupted
        if not interrupted:
            until = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < until:
                time.sleep(0.01)
            interrupted = True
            raise KeyboardInterrupt
        return original_join(thread, timeout)
    monkeypatch.setattr(threading.Thread, "join", interrupt_after_start)
    backend = object.__new__(cli.RealBackend)
    backend.workspace = tmp_path
    backend.agent_heartbeat_s = 0.1
    with pytest.raises(KeyboardInterrupt):
        backend._agent_turn(shlex.join([sys.executable, str(worker)]), "prompt", "student")
    assert ready.exists() and ended.exists()
