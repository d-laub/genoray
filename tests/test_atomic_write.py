from __future__ import annotations

import os
from pathlib import Path

import pytest

import genoray._utils as U
from genoray._utils import atomic_write_dir, atomic_write_path


def test_atomic_write_path_replaces_and_is_sibling(tmp_path: Path):
    dest = tmp_path / "index.gvi"
    with atomic_write_path(dest) as tmp:
        assert tmp.parent == dest.parent  # sibling => same filesystem
        assert tmp != dest
        tmp.write_bytes(b"NEW")
    assert dest.read_bytes() == b"NEW"
    # no leftover temp siblings
    assert [p.name for p in tmp_path.iterdir()] == ["index.gvi"]


def test_atomic_write_path_failure_preserves_dest(tmp_path: Path):
    dest = tmp_path / "index.gvi"
    dest.write_bytes(b"OLD")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write_path(dest) as tmp:
            tmp.write_bytes(b"PARTIAL")
            raise RuntimeError("boom")
    assert dest.read_bytes() == b"OLD"  # untouched
    assert [p.name for p in tmp_path.iterdir()] == ["index.gvi"]  # tmp cleaned


def test_atomic_write_dir_swaps_new(tmp_path: Path):
    dest = tmp_path / "out.svar"
    with atomic_write_dir(dest) as staging:
        assert staging.parent == dest.parent
        assert staging.is_dir()
        (staging / "data.bin").write_bytes(b"X")
    assert (dest / "data.bin").read_bytes() == b"X"
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]


def test_atomic_write_dir_overwrite_replaces(tmp_path: Path):
    dest = tmp_path / "out.svar"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"OLD")
    with atomic_write_dir(dest) as staging:
        (staging / "new.bin").write_bytes(b"NEW")
    assert (dest / "new.bin").read_bytes() == b"NEW"
    assert not (dest / "old.bin").exists()  # fully replaced, not merged
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]


def test_atomic_write_dir_failure_preserves_existing(tmp_path: Path):
    dest = tmp_path / "out.svar"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"OLD")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write_dir(dest) as staging:
            (staging / "partial.bin").write_bytes(b"PARTIAL")
            raise RuntimeError("boom")
    assert (dest / "old.bin").read_bytes() == b"OLD"  # untouched
    assert not (dest / "partial.bin").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]  # staging cleaned


def test_atomic_write_dir_rollback_on_swap_failure(tmp_path: Path, monkeypatch):
    dest = tmp_path / "out.svar"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"OLD")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # 1=dest->backup, 2=staging->dest (fail), 3=rollback
            raise OSError("swap boom")
        return real_replace(src, dst)

    monkeypatch.setattr(U.os, "replace", flaky_replace)

    with pytest.raises(OSError, match="swap boom"):
        with atomic_write_dir(dest) as staging:
            (staging / "new.bin").write_bytes(b"NEW")

    assert (dest / "old.bin").read_bytes() == b"OLD"  # rolled back
    assert not (dest / "new.bin").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out.svar"]
