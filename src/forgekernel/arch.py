"""Hardware architecture names and knowledge-base lookup.

PyPTO should not require a code change when a task moves between Ascend
variants.  The task's ``arch`` remains the exact target (it is recorded in the
run and prompts), while knowledge-base lookup prefers an exact tree and then a
generic ``ascend`` tree.  That fallback is deliberately narrow: it never
pretends a 950PR recipe was measured on a 950DT, and the generic index must say
so.
"""
from __future__ import annotations

import re
from pathlib import Path

ASCEND_ARCH_RE = re.compile(r"^ascend\d{3,4}[a-z]{0,2}$")
# Older cards used the family name for a 950PR run.
_ARCH_ALIASES = {"ascend950": "ascend950pr"}


def canonical_arch(arch: str) -> str:
    """The canonical spelling for a supported Ascend architecture."""
    arch = str(arch or "").strip().lower()
    return _ARCH_ALIASES.get(arch, arch)


def is_ascend_arch(arch: str | None) -> bool:
    return bool(arch and ASCEND_ARCH_RE.fullmatch(canonical_arch(arch)))


def hardware_label(arch: str) -> str:
    """A human-facing label derived from the architecture name.

    ``ascend910b`` -> ``Ascend910B``; ``ascend950pr`` -> ``Ascend950PR``.
    This is only a label; measured facts still belong in the exact KB tree.
    """
    arch = canonical_arch(arch)
    m = re.fullmatch(r"ascend(\d{3,4})([a-z]{0,2})", arch)
    if not m:
        return arch
    return "Ascend" + m.group(1) + m.group(2).upper()


def variants() -> str:
    """A compact spelling of the supported families for user-facing messages."""
    return "ascend910b/c, ascend950pr/dt"


def kb_candidates(kb_root: Path, language: str, arch: str) -> list[Path]:
    """Exact architecture tree, then the language's generic Ascend tree."""
    root = Path(kb_root)
    canonical = canonical_arch(arch)
    out = [root / canonical]
    if language == "pypto" and is_ascend_arch(canonical):
        generic = root / "ascend"
        if generic not in out:
            out.append(generic)
    return out


def resolve_kb_dir(kb_root: Path, language: str, arch: str) -> Path:
    """The first existing KB tree for this language/architecture."""
    candidates = kb_candidates(kb_root, language, arch)
    return next((p for p in candidates if p.is_dir()), candidates[0])
