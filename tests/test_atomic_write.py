from __future__ import annotations

import os
import shutil
from pathlib import Path

import polars as pl
import pytest

import genoray._utils as U
from genoray import SparseVar, VCF
from genoray._utils import atomic_write_dir, atomic_write_path

_DDIR = Path(__file__).parent / "data"


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


def _raise_boom(*args, **kwargs):
    raise RuntimeError("write boom")


def _corrupt_then_boom(self, path, *args, **kwargs):
    from pathlib import Path as _P

    _P(path).write_bytes(b"CORRUPT")
    raise RuntimeError("write boom")


def test_vcf_gvi_write_atomic_no_leftover(tmp_path: Path):
    for ext in (".vcf.gz", ".vcf.gz.csi"):
        shutil.copy(_DDIR / f"biallelic{ext}", tmp_path / f"biallelic{ext}")
    vcf = VCF(tmp_path / "biallelic.vcf.gz")
    vcf._write_gvi_index(overwrite=True)
    gvi = vcf._index_path()
    assert gvi.exists()
    pl.read_ipc(gvi)  # loads cleanly
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_vcf_gvi_write_preserves_on_failure(tmp_path: Path, monkeypatch):
    for ext in (".vcf.gz", ".vcf.gz.csi"):
        shutil.copy(_DDIR / f"biallelic{ext}", tmp_path / f"biallelic{ext}")
    vcf = VCF(tmp_path / "biallelic.vcf.gz")
    vcf._write_gvi_index(overwrite=True)
    gvi = vcf._index_path()
    good = gvi.read_bytes()
    monkeypatch.setattr(pl.DataFrame, "write_ipc", _corrupt_then_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        vcf._write_gvi_index(overwrite=True)
    assert gvi.read_bytes() == good  # prior index intact
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def test_pgen_write_index_preserves_on_failure(tmp_path: Path, monkeypatch):
    from genoray._pgen import _write_index

    for ext in (".pgen", ".pvar", ".psam"):
        shutil.copy(_DDIR / f"biallelic{ext}", tmp_path / f"biallelic{ext}")
    idx = tmp_path / "biallelic.pvar.gvi"
    _write_index(idx)
    good = idx.read_bytes()
    monkeypatch.setattr(pl.LazyFrame, "sink_ipc", _corrupt_then_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        _write_index(idx)
    assert idx.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []


def _dir_digest(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_from_vcf_atomic_no_leftover(tmp_path: Path):
    out = tmp_path / "x.svar"
    SparseVar.from_vcf(out, VCF(_DDIR / "biallelic.vcf.gz"), max_mem="1g")
    assert (out / "metadata.json").exists()
    SparseVar(out)  # loads cleanly
    # only the final output remains in the parent dir
    assert [p.name for p in tmp_path.iterdir() if p.name != "x.svar"] == []


def test_from_vcf_atomic_preserves_existing_on_failure(tmp_path: Path, monkeypatch):
    import genoray._svar as S

    out = tmp_path / "x.svar"
    SparseVar.from_vcf(out, VCF(_DDIR / "biallelic.vcf.gz"), max_mem="1g")
    before = _dir_digest(out)

    monkeypatch.setattr(S, "_concat_data", _raise_boom)
    with pytest.raises(RuntimeError, match="write boom"):
        SparseVar.from_vcf(
            out, VCF(_DDIR / "biallelic.vcf.gz"), max_mem="1g", overwrite=True
        )

    assert _dir_digest(out) == before  # old output intact
    assert [p.name for p in tmp_path.iterdir() if p.name != "x.svar"] == []
