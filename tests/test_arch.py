"""Ascend architecture aliases and generic KB fallback."""
from __future__ import annotations

from pathlib import Path

from forgekernel import arch
from forgekernel.task import load_task
from test_task import _task


def _kb_root(tmp_path: Path, *arches: str, generic: bool = False) -> Path:
    root = tmp_path / "repo"
    for name in arches:
        d = root / "hw_kb/pypto" / name
        d.mkdir(parents=True)
        (d / "index.md").write_text("# index\n")
    if generic:
        d = root / "hw_kb/pypto/ascend"
        d.mkdir(parents=True)
        (d / "index.md").write_text("# generic\n")
    return root


def test_supported_ascend_variants_have_stable_labels():
    assert arch.canonical_arch("ascend910b") == "ascend910b"
    assert arch.canonical_arch("ascend910c") == "ascend910c"
    assert arch.canonical_arch("ascend950pr") == "ascend950pr"
    assert arch.canonical_arch("ascend950dt") == "ascend950dt"
    assert arch.canonical_arch("ascend950") == "ascend950pr"
    assert arch.hardware_label("ascend910b") == "Ascend910B"
    assert arch.hardware_label("ascend950dt") == "Ascend950DT"


def test_generic_ascend_kb_is_a_fallback_not_an_exact_certification(tmp_path):
    root = _kb_root(tmp_path, generic=True)
    exact = root / "hw_kb/pypto/ascend950dt"
    exact.mkdir()
    (exact / "index.md").write_text("# exact\n")
    assert arch.resolve_kb_dir(root / "hw_kb/pypto", "pypto", "ascend950dt") == exact
    assert (arch.resolve_kb_dir(root / "hw_kb/pypto", "pypto", "ascend910c")
            == root / "hw_kb/pypto/ascend")


def test_pypto_task_accepts_all_requested_variants(tmp_path):
    for name, label in (("ascend910b", "Ascend910B"), ("ascend910c", "Ascend910C"),
                        ("ascend950pr", "Ascend950PR"), ("ascend950dt", "Ascend950DT")):
        root = _kb_root(tmp_path / name, name)
        d = _task(tmp_path / name,
                  f'[task]\nlanguage = "pypto"\narch = "{name}"\n')
        task = load_task(d, kb_root=root)
        assert task.arch == name
        assert arch.hardware_label(task.arch) == label


def test_unknown_ascend_variant_can_use_generic_kb(tmp_path):
    root = _kb_root(tmp_path, generic=True)
    d = _task(tmp_path / "future",
              '[task]\nlanguage = "pypto"\narch = "ascend980"\n')
    task = load_task(d, kb_root=root)
    assert task.arch == "ascend980"
    assert arch.resolve_kb_dir(root / "hw_kb/pypto", "pypto", task.arch) == (
        root / "hw_kb/pypto/ascend")
