from __future__ import annotations

import subprocess
import sys
from pathlib import Path


from genoray import SparseVar


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "genoray._cli", *argv],
        check=False,
        capture_output=True,
        text=True,
    )


def test_view_single_region_single_sample(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        [
            "view",
            str(tiny_svar),
            str(out),
            "-r",
            "chr1:1-100",
            "-s",
            "A",
        ]
    )
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    assert list(sub.available_samples) == ["A"]
    assert sub.n_variants >= 1  # A has at least one non-ref call in chr1:1-100


def test_view_bed_and_sample_file(tmp_path: Path, tiny_svar: Path):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t0\t100\n")
    samples_f = tmp_path / "s.txt"
    samples_f.write_text("A\nB\n")
    out = tmp_path / "view.svar"
    r = _run(
        [
            "view",
            str(tiny_svar),
            str(out),
            "-R",
            str(bed),
            "-S",
            str(samples_f),
        ]
    )
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    assert sorted(sub.available_samples) == ["A", "B"]


def test_view_regions_comma_list(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        [
            "view",
            str(tiny_svar),
            str(out),
            "-r",
            "chr1:1-15,chr1:25-35",
            "-s",
            "A,B,C",
        ]
    )
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    # genoray's .index stores POS as 1-based (matches the source VCF POS),
    # so VCF POS=10/20/30/40 appear unchanged in sub.index["POS"].
    # POS 10 falls in 1-15, POS 30 in 25-35, POS 20 outside both, POS 40 outside both.
    positions = sub.index["POS"].to_list()
    assert 10 in positions
    assert 30 in positions
    assert 20 not in positions
    assert 40 not in positions  # outside both ranges


def test_view_no_args_errors(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(["view", str(tiny_svar), str(out)])
    assert r.returncode != 0
    assert "at least one of" in (r.stderr + r.stdout).lower()


def test_view_regions_only_uses_all_samples(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        [
            "view",
            str(tiny_svar),
            str(out),
            "-r",
            "chr1:1-100",
        ]
    )
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    src = SparseVar(tiny_svar)
    assert sorted(sub.available_samples) == sorted(src.available_samples)
    assert sub.n_variants == src.n_variants  # all variants kept since region covers all


def test_view_samples_only_uses_all_variants(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        [
            "view",
            str(tiny_svar),
            str(out),
            "-s",
            "A,B,C",
        ]
    )
    assert r.returncode == 0, r.stderr
    sub = SparseVar(out)
    src = SparseVar(tiny_svar)
    # All variants kept since all samples kept.
    assert sub.n_variants == src.n_variants
    assert sorted(sub.available_samples) == sorted(src.available_samples)
