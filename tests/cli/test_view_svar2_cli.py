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
