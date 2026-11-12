"""Codex backend contract, exercised with a local CLI stand-in (no API calls)."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel import agents, provider
from forgekernel.task import AgentSpec, TaskError, load_task
from test_task import MINIMAL, _kb, _task

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("codex_run", REPO / "scripts/codex-run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def policy_file(tmp_path):
    p = tmp_path / ".codex/forgekernel-policy.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"schema": 1, "backend": "codex", "config": {
        "default_permissions": "forgekernel",
        "permissions.forgekernel": {"filesystem": {"/": "read", str(tmp_path): "write"}},
        "approval_policy": "never", "web_search": "disabled"}}))
    return p


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    cli = tmp_path / "codex"
    cli.write_text(f"#!{sys.executable}\n" + '''import json, os, sys, time
from pathlib import Path
if '--help' in sys.argv:
    print('--ignore-user-config --ignore-rules --strict-config')
    raise SystemExit(0)
if os.environ.get('FAKE_SLEEP'):
    time.sleep(float(os.environ['FAKE_SLEEP']))
Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'prompt': sys.stdin.read()}))
print(json.dumps({'type': 'thread.started', 'thread_id': 'thread-123'}))
if os.environ.get('RECONNECT_ERROR') == '1':
    print(json.dumps({'type': 'error', 'message': 'Reconnecting... stream disconnected'}))
print(json.dumps({'type': 'turn.completed'}))
if os.environ.get('EVENT_ERROR') == '1':
    print(json.dumps({'type': 'turn.failed', 'error': {'message': 'rate limit exceeded'}}))
if os.environ.get('TERMINAL_ERROR') == '1':
    print(json.dumps({'type': 'error', 'message': 'terminal stream error'}))
print('progress message', file=sys.stderr)
if os.environ.get('NO_FINAL') != '1':
    Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text('REC D1.1 final reply')
raise SystemExit(int(os.environ.get('FAKE_EXIT', '0')))
''')
    cli.chmod(0o755)
    monkeypatch.setenv("FK_CODEX_BIN", str(cli))
    monkeypatch.setenv("CAPTURE", str(tmp_path / "capture.json"))
    monkeypatch.chdir(tmp_path)
    policy_file(tmp_path)
    return cli


def test_roles_can_both_use_codex(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL + '[agent]\nbackend="codex"\nmodel="test-model"\n'),
                  kb_root=_kb(tmp_path))
    assert t.teacher.backend == t.student.backend == "codex"
    assert "agent-codex.sh" in agents.agent_command(t.teacher, "teacher")


def test_mixed_roles_preserve_claude_default(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL + '[agent.student]\nbackend="codex"\nmodel="test-model"\n'),
                  kb_root=_kb(tmp_path))
    assert t.teacher.backend == "claude" and t.student.backend == "codex"


def test_switching_backend_requires_explicit_model(tmp_path):
    with pytest.raises(TaskError, match="explicit Codex model"):
        load_task(_task(tmp_path, MINIMAL + '[agent]\nbackend="codex"\n'), kb_root=_kb(tmp_path))


def test_codex_provider_uses_responses_wire_and_never_embeds_key(tmp_path, monkeypatch):
    (tmp_path / "backends.toml").write_text('[gw]\nbase_url="https://gateway.example/v1"\n'
                                            'wire="responses"\nkey_env="GATEWAY_KEY"\n')
    monkeypatch.setattr(provider, "global_config_dir", lambda: tmp_path)
    spec = AgentSpec(backend="codex", model="test-model", effort="high", provider="gw")
    command = agents.agent_command(spec, "student", env={"GATEWAY_KEY": "secret-value"})
    assert "secret-value" not in command
    assert "FK_CODEX_BASE_URL=https://gateway.example/v1" in command
    assert "FK_CODEX_KEY_ENV=GATEWAY_KEY" in command
    with pytest.raises(provider.ProviderError, match="speaks"):
        provider.grant("claude", "gw", {"GATEWAY_KEY": "secret-value"})
    with pytest.raises(provider.ProviderError, match="GATEWAY_KEY"):
        provider.grant("codex", "gw", {})


def test_command_safely_quotes_model_and_effort():
    spec = AgentSpec(backend="codex", model="model with spaces", effort="high")
    args = shlex.split(agents.agent_command(spec, "student"))
    assert args[-3:] == ["model with spaces", "high", "7200"]


def test_run_turn_separates_final_reply_from_events(tmp_path, fake_cli):
    proc = runner.execute_turn("test-model", "high", "work on the kernel",
                               tmp_path / "last.md", tmp_path / "events.jsonl")
    assert proc.returncode == 0 and proc.stdout == "REC D1.1 final reply"
    assert "progress message" in proc.stderr
    assert "thread.started" in (tmp_path / "events.jsonl").read_text()
    captured = json.loads((tmp_path / "capture.json").read_text())
    assert captured["prompt"] == "work on the kernel"
    assert 'approval_policy="never"' in captured["args"]
    assert 'model_reasoning_effort="high"' in captured["args"]
    assert all(flag in captured["args"] for flag in runner.REQUIRED_FLAGS)
    assert "--dangerously-bypass-approvals-and-sandbox" not in captured["args"]


def test_resume_explicitly_names_session_and_reinstalls_policy(tmp_path, fake_cli):
    argv = runner.build_command("test-model", "", output_path=tmp_path / "reply", resume_id="thread-123")
    assert argv[1:3] == ["exec", "resume"]
    assert argv[-2:] == ["thread-123", "-"]
    assert 'default_permissions="forgekernel"' in argv


def test_provider_config_uses_environment_key_name(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("FK_CODEX_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("FK_CODEX_KEY_ENV", "GW_KEY")
    monkeypatch.setenv("GW_KEY", "private-key")
    argv = runner.build_command("test-model", "")
    assert "private-key" not in " ".join(argv)
    config = next(v for v in argv if v.startswith("model_providers.forgekernel="))
    assert '"wire_api" = "responses"' in config and '"env_key" = "GW_KEY"' in config


def test_policy_is_required_before_cli_invocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(runner.CodexError, match="policy"):
        runner.build_command("test-model", "")


def test_legacy_cli_is_rejected(tmp_path):
    cli = tmp_path / "old-codex"
    cli.write_text(f"#!{sys.executable}\nprint('old CLI')\n")
    cli.chmod(0o755)
    with pytest.raises(runner.CodexError, match="upgrade Codex"):
        runner.check_cli(str(cli))


@pytest.mark.parametrize("name", ["config.toml", "hooks.json"])
def test_foreign_project_config_cannot_weaken_policy(tmp_path, fake_cli, name):
    (tmp_path / ".codex" / name).write_text("")
    with pytest.raises(runner.CodexError, match="foreign project"):
        runner.build_command("test-model", "")


def test_ancestor_project_config_is_rejected(tmp_path, fake_cli, monkeypatch):
    (tmp_path / ".codex/config.toml").write_text('sandbox_mode = "danger-full-access"')
    child = tmp_path / "nested"
    child.mkdir()
    policy_file(child)
    monkeypatch.chdir(child)
    with pytest.raises(runner.CodexError, match="foreign project"):
        runner.build_command("test-model", "")


def test_parent_config_beyond_nested_git_root_is_rejected(tmp_path, fake_cli, monkeypatch):
    (tmp_path / ".codex/config.toml").write_text('sandbox_mode = "danger-full-access"')
    child = tmp_path / "nested"
    child.mkdir()
    (child / ".git").mkdir()
    policy_file(child)
    monkeypatch.chdir(child)
    with pytest.raises(runner.CodexError, match="foreign project"):
        runner.build_command("test-model", "")


def test_ignored_user_config_is_not_a_project_override(tmp_path, fake_cli, monkeypatch):
    (tmp_path / ".codex/config.toml").write_text('model = "personal-default"')
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    assert runner.build_command("test-model", "")


def test_event_failure_with_final_message_and_zero_exit_is_failure(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("EVENT_ERROR", "1")
    proc = runner.execute_turn("test-model", "", "prompt", tmp_path / "reply", tmp_path / "events")
    assert proc.returncode == 1 and not proc.stdout
    assert "rate limit exceeded" in proc.stderr


def test_reconnect_error_followed_by_completed_turn_succeeds(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("RECONNECT_ERROR", "1")
    proc = runner.execute_turn("test-model", "", "prompt", tmp_path / "reply", tmp_path / "events")
    assert proc.returncode == 0 and proc.stdout == "REC D1.1 final reply"


def test_unresolved_terminal_error_still_fails(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("TERMINAL_ERROR", "1")
    proc = runner.execute_turn("test-model", "", "prompt", tmp_path / "reply", tmp_path / "events")
    assert proc.returncode == 1 and not proc.stdout
    assert "terminal stream error" in proc.stderr


def test_turn_failed_is_terminal_even_if_a_completion_is_printed():
    events = '\n'.join(json.dumps({'type': kind}) for kind in ('turn.failed', 'turn.completed'))
    assert 'turn.failed' in runner.event_failure(events)


@pytest.mark.parametrize("exit_code,no_final,expected", [(17, False, 17), (0, True, 1)])
def test_failed_or_empty_turn_is_not_success(tmp_path, fake_cli, monkeypatch, exit_code, no_final, expected):
    monkeypatch.setenv("FAKE_EXIT", str(exit_code))
    monkeypatch.setenv("NO_FINAL", "1" if no_final else "0")
    proc = runner.execute_turn("test-model", "", "prompt", tmp_path / "reply", tmp_path / "events")
    assert proc.returncode == expected


def test_script_preserves_stdout_contract(tmp_path, fake_cli):
    proc = subprocess.run(["bash", str(REPO / "scripts/agent-codex.sh"), "test-model", "", "10"],
                          input="prompt", capture_output=True, text=True,
                          env={**os.environ, "FK_AGENT_PY": sys.executable})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "REC D1.1 final reply"
    assert "thread.started" in proc.stderr


def test_launcher_bounds_the_agent_turn(tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "30")
    proc = subprocess.run(["bash", str(REPO / "scripts/agent-codex.sh"), "test-model", "", "0.05"],
                          input="prompt", capture_output=True, text=True, timeout=10,
                          env={**os.environ, "FK_AGENT_PY": sys.executable})
    assert proc.returncode == 124


def test_override_values_are_toml_not_json():
    import tomllib
    value = {"filesystem": {"/path with spaces": "deny"}, "network": {"enabled": False}}
    assert tomllib.loads("value = " + runner.toml_value(value))["value"] == value
