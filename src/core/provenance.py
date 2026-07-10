"""Source-tree provenance captured in every experiment artifact."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Sequence


def _git_output(repo_root: Path, args: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return b""
    return completed.stdout if completed.returncode == 0 else b""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def collect_source_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Return Git/diff/lock identity without mutating repository state.

    The dirty hash includes both the tracked binary diff and content hashes for
    untracked, non-ignored files. This distinguishes remediation runs made from
    different working trees even when they share the same HEAD revision.
    """

    root = repo_root or Path(__file__).resolve().parents[2]
    revision = _git_output(root, ["rev-parse", "HEAD"]).decode(
        "utf-8", errors="replace"
    ).strip()
    status = _git_output(
        root, ["status", "--porcelain=v1", "--untracked-files=all"]
    )
    dirty = bool(status.strip())

    diff_hash: str | None = None
    untracked_count = 0
    if dirty:
        digest = hashlib.sha256()
        digest.update(b"tracked-diff\0")
        digest.update(
            _git_output(root, ["diff", "--binary", "--no-ext-diff", "HEAD", "--", "."])
        )
        untracked_blob = _git_output(
            root, ["ls-files", "--others", "--exclude-standard", "-z"]
        )
        for raw_name in sorted(name for name in untracked_blob.split(b"\0") if name):
            untracked_count += 1
            relative = raw_name.decode("utf-8", errors="surrogateescape")
            digest.update(b"untracked\0")
            digest.update(raw_name)
            digest.update(b"\0")
            digest.update(_sha256_file(root / relative).encode("ascii"))
            digest.update(b"\0")
        diff_hash = digest.hexdigest()

    lock_path = root / "uv.lock"
    return {
        "git_revision": revision or "unknown",
        "git_dirty": dirty,
        "git_diff_sha256": diff_hash,
        "git_untracked_file_count": untracked_count,
        "uv_lock_sha256": _sha256_file(lock_path) if lock_path.exists() else "",
    }
