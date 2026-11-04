"""The DeepSeek task contract pins model AND endpoint, for both agent roles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from forgekernel import agents, provider
from forgekernel.paths import repo_root
from forgekernel.task import load_task


@pytest.mark.parametrize("task_name", ["swiglu-pypto", "swiglu-pypto-g2", "gqa-pypto"])
@pytest.mark.parametrize("role", ["teacher", "student"])
def test_pypto_roles_ignore_stale_native_teacher_overrides(task_name, role):
    task = load_task(repo_root() / "tasks" / task_name)
    spec = getattr(task, role)
    env = {"API_KEY": "deepseek-test-key", "TEACHER_API_KEY": "old-test-key",
           "TEACHER_BASEURL": "https://api.kimi.com/coding", "TEACHER_MODEL": "k3.1-pro",
           "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding"}
    assert spec.provider == "deepseek"
    assert spec.model == agents.model_for_role(spec, role, env) == "deepseek-flash[1m]"
    assert spec.effort == "max"
    words = shlex.split(agents.agent_command(spec, role, env))
    assert words[1:3] == ["ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic",
                          "FK_PROVIDER_KEY_ENV=API_KEY"]
    assert "deepseek-flash[1m]" in words and "kimi" not in " ".join(words)
    assert "deepseek-test-key" not in " ".join(words)
    # Partial stale overrides must not trigger native routing or a fallback.
    assert agents.agent_command(spec, role, {"API_KEY": "test", "TEACHER_MODEL": "old"})


def test_missing_deepseek_key_never_falls_back_to_teacher_key():
    task = load_task(repo_root() / "tasks/swiglu-pypto")
    with pytest.raises(provider.ProviderError, match="needs API_KEY"):
        agents.agent_command(task.teacher, "teacher", {"TEACHER_API_KEY": "old-test-key"})


def test_real_launcher_binds_both_auth_variables_to_the_selected_key(tmp_path):
    # Execute the actual shell launcher with a harmless local CLI stand-in: no API.
    cli = tmp_path / "mock-claude"
    cli.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                   "print(json.dumps({'host': os.environ['ANTHROPIC_BASE_URL'], "
                   "'key': os.environ['ANTHROPIC_API_KEY'], "
                   "'token': os.environ['ANTHROPIC_AUTH_TOKEN'], "
                   "'oauth': os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'), 'args': sys.argv[1:]}))\n")
    cli.chmod(0o755)
    env = {**os.environ, "API_KEY": "deepseek-test-key", "TEACHER_API_KEY": "old-test-key",
           "ANTHROPIC_API_KEY": "old-test-key", "ANTHROPIC_AUTH_TOKEN": "old-test-key",
           "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding",
           "CLAUDE_CODE_OAUTH_TOKEN": "old-subscription-token",
           "FK_CLAUDE_BIN": str(cli), "FK_AGENT_PY": os.sys.executable}
    env.pop("FK_AGENT_DRIVER", None)
    task = load_task(repo_root() / "tasks/swiglu-pypto")
    for role in ("teacher", "student"):
        proc = subprocess.run(shlex.split(agents.agent_command(getattr(task, role), role, env)),
                              env=env, input="test", capture_output=True, text=True, timeout=20,
                              check=True)
        observed = json.loads(proc.stdout)
        assert observed["host"] == "https://api.deepseek.com/anthropic"
        assert observed["key"] == observed["token"] == "deepseek-test-key"
        assert observed["oauth"] is None
        assert observed["args"][observed["args"].index("--model") + 1] == "deepseek-flash[1m]"
