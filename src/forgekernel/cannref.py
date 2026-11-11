"""Seal a CANN source package in a teacher-only execution tree.

The frozen task keeps portable source, denied to agents. A run's reference tree
contains only a keyed payload; its job materializes the build package remotely.
No payload or key rides a student job. This uses the same policy/key mechanism
as the existing source-reference loader, not a claim of a hostile-code sandbox.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from forgekernel import binref, sealref
from forgekernel.paths import template_dir


def seal_tree(tree: Path, entry: str) -> None:
    config_path = tree / "eval/cann_task.json"
    config = json.loads(config_path.read_text())
    source = (tree / config.get("source_dir", "src")).resolve()
    if source == tree.resolve() or not source.is_relative_to(tree.resolve()):
        raise ValueError("sealed CANN source must be a subdirectory of the reference tree")
    loader = sealref.closed_ref(template_dir() / "eval")
    files = {str(p.relative_to(source)): p.read_text()
             for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    manifest = loader.bundle_manifest(files, entry)
    key = loader.generate_key()
    payload = tree / sealref.PAYLOAD
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(loader.seal_bundle(manifest, key))
    sealref.write_key(tree, key)
    shutil.copy2(template_dir() / "eval/closed_ref.py", tree / "eval/closed_ref.py")
    config["sealed_source"] = sealref.PAYLOAD
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    shutil.rmtree(source)
    script = tree / "gpu_entry.sh"
    text = sealref._with_key_preamble(script.read_text(), sealref.KEY_ENV)
    text = text.replace('        run_with_timeout bash -c "$*"',
                        '        blackbox_guard "$*"; run_with_timeout bash -c "$*"')
    guards = sealref.REACH + sealref.REF_TREE_WORDS + [r"\.sealed_source", r"cann_bench/kernel_impl"]
    boundary = text.rindex('case "$TASK" in\n')
    script.write_text(text[:boundary] + binref._guard_block(
        guards, guards, "Ascend sealed reference", extra_files=binref.CARD_CPP_NOTES)
        + "\n" + text[boundary:])
    with (tree / ".gitignore").open("a") as f:
        f.write("\n*.local.toml\nref_bin/*.key\n.sealed_source/\n")
