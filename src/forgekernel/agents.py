"""Agent commands from a task's ``[agent]`` (merged per role: forgekernel.task).

The harness builds the command line itself (scripts/agent-claude.sh): the task names the
backend, the provider, the model, the effort and a turn's timeout, and the user never writes
a CLI invocation. One script serves both a ``claude -p`` turn and, when the caller asks for
one (FK_AGENT_DRIVER=goal), a /goal session.
"""

from __future__ import annotations

import shlex
import os

from forgekernel import provider as provider_mod
from forgekernel.paths import repo_root

AGENT_SCRIPT = "scripts/agent-claude.sh"


def model_for_role(spec, role: str, env: dict | None = None) -> str:
    if spec.provider == "native" and role == "teacher":
        environment = os.environ if env is None else env
        if environment.get("TEACHER_MODEL"):
            return environment["TEACHER_MODEL"]
    return spec.model


def agent_command(spec, role: str, env: dict | None = None) -> str:
    """The shell command the loop runs for one ``role`` turn (prompt on stdin). A provider
    other than ``native`` rides as environment assignments in front of it: the endpoint and
    the name of the variable that holds the key, never the key."""
    if spec.backend != "claude":
        raise ValueError(f"backend {spec.backend!r} is not supported (claude only)")
    grant = provider_mod.grant(spec.backend, spec.provider, env)
    environment = os.environ if env is None else env
    model = spec.model
    # Role-scoped overrides never expose the key in argv or saved task metadata.
    # Explicit named providers continue to take precedence over environment defaults.
    if spec.provider == "native":
        prefix = "TEACHER_" if role == "teacher" else ""
        names = [prefix + key for key in ("MODEL", "BASEURL", "API_KEY")]
        if role == "teacher" and any(environment.get(k) for k in names):
            if not all(environment.get(k) for k in names):
                raise ValueError("teacher overrides require TEACHER_MODEL, TEACHER_BASEURL, "
                                 "and TEACHER_API_KEY together")
            model = model_for_role(spec, role, environment)
            grant = {provider_mod.BASE_URL_ENV: environment[names[1]],
                     provider_mod.KEY_ENV_ENV: names[2]}
    argv = ["bash", str(repo_root() / AGENT_SCRIPT), model, spec.effort or "",
            str(spec.timeout_s)]
    prefix = ["env", *(f"{k}={v}" for k, v in grant.items())] if grant else []
    return " ".join(shlex.quote(a) for a in [*prefix, *argv])
