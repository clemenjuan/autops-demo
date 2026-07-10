"""Result source-provenance regression tests."""
from __future__ import annotations

from pathlib import Path

from src.core import provenance


def test_dirty_source_hash_covers_tracked_diff_untracked_content_and_lock(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "uv.lock").write_text("lock-v1", encoding="utf-8")
    (tmp_path / "new.py").write_text("value = 1\n", encoding="utf-8")

    def fake_git_output(root: Path, args: list[str]) -> bytes:
        assert root == tmp_path
        if args[:2] == ["rev-parse", "HEAD"]:
            return b"d8c7cae00000000000000000000000000000000\n"
        if args and args[0] == "status":
            return b" M tracked.py\n?? new.py\n"
        if args and args[0] == "diff":
            return b"tracked-diff-v1"
        if args and args[0] == "ls-files":
            return b"new.py\0"
        raise AssertionError(args)

    monkeypatch.setattr(provenance, "_git_output", fake_git_output)

    first = provenance.collect_source_provenance(tmp_path)
    (tmp_path / "new.py").write_text("value = 2\n", encoding="utf-8")
    second = provenance.collect_source_provenance(tmp_path)

    assert first["git_revision"].startswith("d8c7cae")
    assert first["git_dirty"] is True
    assert len(first["git_diff_sha256"]) == 64
    assert first["git_untracked_file_count"] == 1
    assert len(first["uv_lock_sha256"]) == 64
    assert first["git_diff_sha256"] != second["git_diff_sha256"]


def test_clean_source_has_no_diff_hash(tmp_path: Path, monkeypatch) -> None:
    def fake_git_output(root: Path, args: list[str]) -> bytes:
        if args[:2] == ["rev-parse", "HEAD"]:
            return b"abc123\n"
        return b""

    monkeypatch.setattr(provenance, "_git_output", fake_git_output)
    result = provenance.collect_source_provenance(tmp_path)
    assert result["git_revision"] == "abc123"
    assert result["git_dirty"] is False
    assert result["git_diff_sha256"] is None
    assert result["git_untracked_file_count"] == 0
