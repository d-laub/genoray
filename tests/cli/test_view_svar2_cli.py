import subprocess
import sys

from genoray import SparseVar2


def _run(argv):
    return subprocess.run(
        [sys.executable, "-m", "genoray._cli", *argv], capture_output=True, text=True
    )


def test_cli_split_then_concat(tiny_svar2, tmp_path):
    r = _run(["split", str(tiny_svar2), str(tmp_path / "parts")])
    assert r.returncode == 0, r.stderr
    parts = sorted((tmp_path / "parts").glob("*.svar2"))
    assert len(parts) == 2
    r = _run(["concat", str(tmp_path / "m.svar2"), *map(str, parts)])
    assert r.returncode == 0, r.stderr
    assert set(SparseVar2(tmp_path / "m.svar2").contigs) == set(
        SparseVar2(tiny_svar2).contigs
    )


def test_cli_split_with_contigs(tiny_svar2, tmp_path):
    out = tmp_path / "chr1_only.svar2"
    r = _run(["split", str(tiny_svar2), str(out), "--contigs", "chr1"])
    assert r.returncode == 0, r.stderr
    contigs = SparseVar2(out).contigs
    assert contigs == ["chr1"]
    assert "chr2" not in contigs


def test_cli_view_svar2_region_subset(tiny_svar2, tmp_path):
    # tiny_svar2 is a two-contig (chr1, chr2) store; chr1's variants sit at
    # POS 1 and 3 (1-based). Restricting to chr1:1-40 should keep only chr1.
    out = tmp_path / "v.svar2"
    r = _run(["view", str(tiny_svar2), str(out), "-r", "chr1:1-40"])
    assert r.returncode == 0, r.stderr
    assert SparseVar2(out).contigs == ["chr1"]


def test_cli_view_svar1_still_works(tiny_svar, tmp_path):
    out = tmp_path / "v.svar"
    r = _run(["view", "svar1", str(tiny_svar), str(out), "-r", "chr1:1-100"])
    assert r.returncode == 0, r.stderr


def test_cli_view_svar2_no_reroute_succeeds(tiny_svar2, tmp_path):
    """`--no-reroute` (reroute=False, the representation-preserving direct
    slice) is now implemented and produces a readable store."""
    out = tmp_path / "v.svar2"
    r = _run(
        [
            "view",
            str(tiny_svar2),
            str(out),
            "-r",
            "chr1:1-40",
            "--no-reroute",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert SparseVar2(out).contigs == ["chr1"]
