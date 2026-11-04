"""Receive bounded, checksummed job artifacts through local/SSH stdout.

Only relative regular files are written under the job's artifact directory. A
remote path is never used as a controller-side destination. Corrupt/truncated
payloads are errors, not evidence. The original job log remains the audit trail.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import zlib

MARKER = "FORGE-ARTIFACT: "
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024


def archive_job_artifacts(log: str, dest: Path) -> list[dict]:
    manifest = []
    total = 0
    seen = set()
    for line in log.splitlines():
        if not line.startswith(MARKER):
            continue
        obj = json.loads(line[len(MARKER):])
        name = obj["name"]
        rel = PurePosixPath(name)
        if (rel.is_absolute() or ".." in rel.parts or not rel.parts or "\\" in name
                or name in seen or obj.get("encoding") != "zlib+base64"):
            raise ValueError(f"invalid artifact name/encoding: {name!r}")
        seen.add(name)
        target = dest / str(rel)
        if not target.resolve().is_relative_to(dest.resolve()):
            raise ValueError("artifact escapes destination")
        compressed = base64.b64decode(obj["data"], validate=True)
        decoder = zlib.decompressobj()
        data = decoder.decompress(compressed, MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError("oversized or truncated artifact")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("job artifact budget exceeded")
        digest = hashlib.sha256(data).hexdigest()
        if digest != obj["sha256"]:
            raise ValueError("artifact checksum mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({"name": name, "sha256": digest, "bytes": len(data)})
    if manifest:
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
