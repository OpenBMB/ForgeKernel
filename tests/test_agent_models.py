"""Teacher and student are configured independently by their task tables."""
from __future__ import annotations

import shlex

from forgekernel import agents, provider
from forgekernel.task import load_task
from test_task import _kb, _task

TASK = '''[task]
language = "cutedsl"
arch = "sm100"
[agent]
provider = "teacher-gw"
model = "teacher-model"
effort = "high"
[agent.student]
provider = "student-gw"
model = "student-model"
effort = "low"
'''


def test_teacher_and_student_can_use_different_models_and_providers(tmp_path, monkeypatch):
    gc = tmp_path / "gc"
    gc.mkdir()
    (gc / "backends.toml").write_text(
        '[teacher-gw]\nbase_url = "https://teacher.example/anthropic"\n'
        'wire = "anthropic"\nkey_env = "TEACHER_GW_KEY"\n'
        '[student-gw]\nbase_url = "https://student.example/anthropic"\n'
        'wire = "anthropic"\nkey_env = "STUDENT_GW_KEY"\n')
    monkeypatch.setattr(provider, "global_config_dir", lambda: gc)
    t = load_task(_task(tmp_path, TASK), kb_root=_kb(tmp_path))
    env = {"TEACHER_GW_KEY": "teacher-secret", "STUDENT_GW_KEY": "student-secret"}
    teacher = shlex.split(agents.agent_command(t.teacher, "teacher", env))
    student = shlex.split(agents.agent_command(t.student, "student", env))
    assert teacher[1] == "ANTHROPIC_BASE_URL=https://teacher.example/anthropic"
    assert teacher[2] == "FK_PROVIDER_KEY_ENV=TEACHER_GW_KEY"
    assert teacher[-3] == "teacher-model" and teacher[-2] == "high"
    assert student[1] == "ANTHROPIC_BASE_URL=https://student.example/anthropic"
    assert student[2] == "FK_PROVIDER_KEY_ENV=STUDENT_GW_KEY"
    assert student[-3] == "student-model" and student[-2] == "low"


def test_no_hidden_teacher_environment_override(tmp_path):
    t = load_task(_task(tmp_path, '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[agent]\nmodel = "configured-model"\n'),
                  kb_root=_kb(tmp_path))
    env = {"TEACHER_MODEL": "legacy-teacher", "TEACHER_BASEURL": "https://legacy.invalid",
           "TEACHER_API_KEY": "old-secret", "ANTHROPIC_BASE_URL": "https://legacy.invalid"}
    command = agents.agent_command(t.teacher, "teacher", env)
    assert "configured-model" in command
    assert "legacy-teacher" not in command and "legacy.invalid" not in command
    assert "old-secret" not in command
