#!/usr/bin/env python3
"""Translate Codex PreToolUse events into the harness's shared policy tripwire.

Native permission profiles provide filesystem and network isolation. This hook
blocks named repository fetches and direct cluster commands, as in the Claude
backend. It refuses unknown tool channels rather than allowing MCP/tool escapes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location("fk_policy_hook", Path(__file__).with_name("agent_policy_hook.py"))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


def decide(policy: dict, event: dict) -> str | None:
    tool = event.get("tool_name", "")
    args = event.get("tool_input") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return "BLOCKED: malformed Codex tool input"
    # Canonical Codex hook names are Bash/apply_patch; tolerate raw shell events
    # as well, including the exec_command cmd spelling.
    if tool in ("Bash", "exec_command", "shell", "shell_command"):
        cmd = args.get("command", args.get("cmd", ""))
        if isinstance(cmd, list):
            cmd = " ".join(str(x) for x in cmd)
        return _shared.decide(policy, {"tool_name": "Bash", "tool_input": {"command": cmd}})
    if tool in ("WebSearch", "WebFetch", "web_search", "web.run"):
        if policy.get("offline") or policy.get("deny_urls"):
            return "BLOCKED: web tools are unavailable under this ForgeKernel role policy"
        return None
    if tool in ("apply_patch", "Read", "Edit", "Write", "Grep", "Glob", "update_plan",
                "view_image", "write_stdin", "list_mcp_resources", "list_mcp_resource_templates"):
        return None
    return f"BLOCKED: tool channel {tool!r} is outside the ForgeKernel policy"


def main() -> int:
    try:
        policy = json.loads(Path(sys.argv[1]).read_text())
        event = json.loads(sys.stdin.read())
        msg = decide(policy, event)
    except (OSError, ValueError, IndexError, TypeError) as e:
        msg = f"BLOCKED: unreadable ForgeKernel policy event ({e})"
    if msg:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
              "permissionDecision": "deny", "permissionDecisionReason": msg}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
