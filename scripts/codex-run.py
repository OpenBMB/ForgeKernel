#!/usr/bin/env python3
"""Codex CLI adapter: enforce the installed policy and return only the final reply.

This module is stdlib-only. codex-goal.py shares build_command to ensure resumed
sessions receive exactly the same policy as ordinary turns.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

POLICY_RELPATH = Path(".codex/forgekernel-policy.json")
REQUIRED_FLAGS = ("--ignore-user-config", "--ignore-rules", "--strict-config")


class CodexError(ValueError):
    pass


def toml_value(value) -> str:
    """Encode config overrides as TOML, including maps with literal path keys."""
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(str(k)) + " = " + toml_value(v)
                                for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, (str, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise CodexError(f"unsupported policy config value: {type(value).__name__}")


def check_cli(binary: str) -> None:
    try:
        proc = subprocess.run([binary, "exec", "--help"], capture_output=True,
                              text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodexError(f"cannot run Codex CLI {binary!r}: {exc}") from exc
    missing = [flag for flag in REQUIRED_FLAGS if flag not in proc.stdout]
    if proc.returncode or missing:
        raise CodexError("Codex CLI lacks the permission isolation options required by "
                         "ForgeKernel; upgrade Codex (0.166 or later with "
                         + ", ".join(REQUIRED_FLAGS) + ")")


def check_project_config(workspace: Path) -> None:
    """Reject project config/hook layers that can weaken a named permission profile.

    --ignore-user-config isolates the user's config, but project config is a
    separate layer. The harness owns its policy JSON; it never generates a
    config.toml or hooks.json. User config is explicitly ignored by the CLI;
    every ancestor project layer is checked, including parents of a nested repo.
    """
    user_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser().resolve()
    for parent in (workspace.resolve(), *workspace.resolve().parents):
        for name in ("config.toml", "hooks.json"):
            candidate = parent / ".codex" / name
            if name == "config.toml" and candidate.parent.resolve() == user_home:
                continue  # --ignore-user-config excludes this layer.
            if candidate.exists() or candidate.is_symlink():
                raise CodexError(f"foreign project Codex configuration {candidate} could "
                                 "override the ForgeKernel policy; move it outside this project")


def build_command(model: str, effort: str, *, output_path: Path | None = None,
                  resume_id: str | None = None) -> list[str]:
    check_project_config(Path.cwd())
    path = Path.cwd() / POLICY_RELPATH
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CodexError(f"missing or invalid ForgeKernel Codex policy: {path}") from exc
    if (policy.get("schema") != 1 or policy.get("backend") != "codex"
            or not isinstance(policy.get("config"), dict)):
        raise CodexError(f"invalid ForgeKernel Codex policy schema: {path}")
    config = dict(policy["config"])
    if not config.get("default_permissions") or not any(
            k.startswith("permissions.") for k in config):
        raise CodexError("Codex policy must declare an enforced permission profile")
    config["approval_policy"] = "never"
    if effort:
        config["model_reasoning_effort"] = effort
    base_url = os.environ.get("FK_CODEX_BASE_URL")
    key_env = os.environ.get("FK_CODEX_KEY_ENV")
    if base_url or key_env:
        if not base_url or not key_env or not os.environ.get(key_env):
            raise CodexError("Codex provider needs FK_CODEX_BASE_URL and a populated "
                             "FK_CODEX_KEY_ENV variable")
        config.update({"model_provider": "forgekernel", "model_providers.forgekernel": {
            "name": "ForgeKernel", "base_url": base_url, "env_key": key_env,
            "wire_api": "responses"}})
    binary = os.environ.get("FK_CODEX_BIN", "codex")
    argv = [binary, "exec"]
    if resume_id:
        argv += ["resume"]
    argv += list(REQUIRED_FLAGS)
    argv += ["--model", model]
    # These are CLI overrides, so project or user files cannot relax the policy.
    for key, value in config.items():
        argv += ["-c", f"{key}={toml_value(value)}"]
    if config.get("hooks") or any(k.startswith("hooks.") for k in config):
        argv += ["--dangerously-bypass-hook-trust"]
    argv += ["--json"]
    if output_path is not None:
        argv += ["--output-last-message", str(output_path)]
    if resume_id:
        argv += [resume_id]
    argv += ["-"]
    return argv


def event_failure(events: str) -> str:
    """Return an unresolved event error, allowing a completed reconnect to recover.

    Codex emits type=error during automatic stream reconnects too. A subsequent
    turn.completed clears those diagnostics; turn.failed is a terminal failure.
    The process exit status remains authoritative independently of these events.
    """
    pending, failed = [], []
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "error":
            pending.append(line)
        elif kind == "turn.failed":
            failed.append(line)
        elif kind == "turn.completed":
            pending.clear()
    return "\n".join(failed + pending)


def execute_turn(model: str, effort: str, prompt: str, output_path: Path,
                 events_path: Path) -> subprocess.CompletedProcess:
    check_cli(os.environ.get("FK_CODEX_BIN", "codex"))
    output_path.unlink(missing_ok=True)
    argv = build_command(model, effort, output_path=output_path)
    with events_path.open("w", encoding="utf-8") as events:
        proc = subprocess.run(argv, input=prompt, stdout=events, stderr=subprocess.PIPE,
                              text=True)
    final = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
    # Some CLI versions can report an API/turn error as an event while exiting
    # zero. A partially written final message must not turn that into success.
    error = event_failure(events_path.read_text(encoding="utf-8"))
    if proc.returncode == 0 and error:
        return subprocess.CompletedProcess(argv, 1, "", proc.stderr + "\n" + error)
    if proc.returncode == 0 and not final.strip():
        return subprocess.CompletedProcess(argv, 1, "", "Codex returned no final reply\n" + proc.stderr)
    return subprocess.CompletedProcess(argv, proc.returncode, final, proc.stderr)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        sys.stderr.write("usage: codex-run.py MODEL [EFFORT]\n")
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="fk-codex-") as tmp:
            result = execute_turn(args[0], args[1] if len(args) > 1 else "",
                                  sys.stdin.read(), Path(tmp) / "reply.md",
                                  Path(tmp) / "events.jsonl")
            # JSON events stay out of the driver's final-response parser.
            events = (Path(tmp) / "events.jsonl").read_text(encoding="utf-8")
            if events:
                sys.stderr.write(events)
            sys.stderr.write(result.stderr or "")
            if result.returncode == 0:
                sys.stdout.write(result.stdout)
            return result.returncode
    except (CodexError, OSError) as exc:
        sys.stderr.write(f"codex-run: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
