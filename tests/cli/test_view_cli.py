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
            "svar1",
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
            "svar1",
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
            "svar1",
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
    r = _run(["view", "svar1", str(tiny_svar), str(out)])
    assert r.returncode != 0
    assert "at least one of" in (r.stderr + r.stdout).lower()


def test_view_regions_only_uses_all_samples(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        [
            "view",
            "svar1",
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
            "svar1",
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


def test_view_missing_source_errors(tmp_path: Path):
    out = tmp_path / "view.svar"
    r = _run(["view", str(tmp_path / "nope.svar"), str(out), "-s", "A"])
    combined = r.stderr + r.stdout
    assert r.returncode != 0
    # The validator fires at parse time (before the function body runs), so
    # cyclopts emits a structured "Invalid value" error — NOT a Python traceback.
    # This assertion is the true gate: it fails when the validators.Path(...)
    # annotation on `source` is absent (runtime SparseVar.__init__ raises a
    # FileNotFoundError with a traceback instead).
    assert "Invalid value" in combined, (
        "Expected a cyclopts parse-time validation error ('Invalid value …'), "
        "got a runtime error instead — the source validator may be missing.\n"
        f"stderr: {r.stderr}\nstdout: {r.stdout}"
    )
    assert "Traceback" not in combined, (
        "Got a Python traceback — error is being raised at runtime, not at "
        "parse time. The source validator may be missing.\n"
        f"stderr: {r.stderr}\nstdout: {r.stdout}"
    )
    assert "does not exist" in combined.lower()


def test_view_missing_samples_file_errors(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        ["view", "svar1", str(tiny_svar), str(out), "-S", str(tmp_path / "nope.txt")]
    )
    assert r.returncode != 0
    assert "does not exist" in (r.stderr + r.stdout).lower()


def test_view_missing_regions_file_errors(tmp_path: Path, tiny_svar: Path):
    out = tmp_path / "view.svar"
    r = _run(
        ["view", "svar1", str(tiny_svar), str(out), "-R", str(tmp_path / "nope.bed")]
    )
    assert r.returncode != 0
    assert "does not exist" in (r.stderr + r.stdout).lower()


def _dir_digest(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_view_progress_flag_byte_identical(tmp_path: Path, tiny_svar: Path):
    out_a = tmp_path / "a.svar"
    out_b = tmp_path / "b.svar"
    base = ["view", "svar1", str(tiny_svar)]
    region = ["-r", "chr1:1-100", "-s", "A"]
    r1 = _run([*base, str(out_a), *region])
    r2 = _run([*base, str(out_b), *region, "--progress"])
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    # --progress must not change the written output.
    assert _dir_digest(out_a) == _dir_digest(out_b)
