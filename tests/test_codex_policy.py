"""Codex's role policy must constrain shell access as well as model tool access."""
import importlib.util
import json
from pathlib import Path

from forgekernel import skills
from forgekernel.loop.agent_policy import AgentPolicy, backend_for_command

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("codex_hook", ROOT / "scripts/codex-policy-hook.py")
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def policy(tmp_path, mode="blackbox", **kw):
    return AgentPolicy(oracle_mode=mode, repo_root=tmp_path / "repo",
                       ref_source_paths=(tmp_path / "ref",),
                       ref_source_urls=("github.com/vendor/Reference",),
                       teacher_trees=(tmp_path / "teacher",), **kw)


def test_student_native_permissions(tmp_path):
    ws = tmp_path / "ws"
    p = policy(tmp_path)
    cfg = p.codex_config("student", ws)
    fs = cfg["permissions.forgekernel"]["filesystem"]
    assert fs[str(tmp_path / "ref")] == "deny"
    assert fs[str(tmp_path / "teacher")] == "deny"
    assert fs[str(ws)] == "write"
    assert fs[str(ws / ".codex")] == "read"
    assert fs.get(str(p.repo_root)) != "write"
    assert not cfg["permissions.forgekernel"]["network"]["enabled"]
    assert cfg["web_search"] == "disabled"
    assert cfg["approval_policy"] == "never"
    assert "sandbox_mode" not in cfg


def test_teacher_blackbox_and_source_permissions(tmp_path):
    for mode, expected in (("source", None), ("blackbox", "deny")):
        cfg = policy(tmp_path, mode).codex_config("teacher", tmp_path / "ws")
        profile = cfg["permissions.forgekernel"]
        assert profile["filesystem"].get(str(tmp_path / "ref")) == expected
        assert profile["filesystem"][str(tmp_path / "repo")] == "write"
        assert profile["network"]["enabled"]
        assert cfg["features.network_proxy"]
        if mode == "blackbox":
            assert profile["network"]["domains"]["**.github.com"] == "deny"
            assert cfg["web_search"] == "disabled"


def test_goal_online_keeps_read_only_records(tmp_path):
    ledger = tmp_path / "ws" / "ledger"
    cfg = policy(tmp_path, student_online=True, deny_write=(ledger,)).codex_config("student", tmp_path / "ws")
    assert cfg["permissions.forgekernel"]["network"]["enabled"]
    assert cfg["permissions.forgekernel"]["filesystem"][str(ledger)] == "read"


def test_install_and_role_switch_mirrors_skills(tmp_path):
    ws = tmp_path / "ws"
    p = policy(tmp_path)
    path = p.install("student", ws, backend="codex")
    assert path == ws / ".codex/forgekernel-policy.json"
    assert json.loads(path.read_text())["offline"]
    assert {d.name for d in (ws / ".agents/skills").iterdir()} == {d.name for d in skills.skill_dirs()}
    for mirrored in (ws / ".agents/skills").iterdir():
        body = (mirrored / "SKILL.md").read_text()
        assert body.startswith("---\nname:")
        assert "\ndescription:" in body
    p.install("teacher", ws, backend="codex")
    assert not json.loads(path.read_text())["offline"]


def test_backend_detection():
    assert backend_for_command("bash '/a path/agent-codex.sh' model high 20") == "codex"
    assert backend_for_command("bash /scripts/agent-claude.sh codex-model high 20") == "claude"


def test_hook_rejects_cluster_and_repository_channels(tmp_path):
    doc = policy(tmp_path, student_online=True).policy_document("student")
    assert hook.decide(doc, {"tool_name": "exec_command", "tool_input": {"cmd": "ssh gpu nvidia-smi"}})
    assert hook.decide(doc, {"tool_name": "mcp__remote__fetch", "tool_input": {}})
    assert hook.decide(doc, {"tool_name": "web_search", "tool_input": {}})
    assert hook.decide(doc, {"tool_name": "Bash", "tool_input": {"command": "cat kernel.py"}}) is None
    doc = policy(tmp_path).policy_document("teacher")
    assert hook.decide(doc, {"tool_name": "Bash", "tool_input": {"command": "curl https://raw.githubusercontent.com/vendor/Reference/main/kernel.py"}})


def test_bare_or_foreign_codex_cannot_skip_native_policy(tmp_path):
    from forgekernel.loop.agent_policy import command_findings
    assert command_findings(["codex exec -"], {}, tmp_path)
    assert command_findings(["bash /untrusted/agent-codex.sh model high 30"], {}, tmp_path)
    assert command_findings([f"bash {ROOT / 'scripts/agent-codex.sh'} model high 30"], {}, tmp_path) == []
