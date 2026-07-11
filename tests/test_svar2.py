import json
from pathlib import Path

from genoray import SparseVar2, _core


def _write_fixture(root: Path) -> None:
    """A finished-store fixture: meta.json + empty per-contig dirs. Empty
    sub-streams are tolerated by ContigReader::open, so no pipeline run is
    needed to exercise the skeleton."""
    meta = {
        "format_version": 1,
        "samples": ["S0", "S1"],
        "contigs": ["chr1", "chr2"],
        "ploidy": 2,
    }
    (root / "meta.json").write_text(json.dumps(meta))
    for contig in meta["contigs"]:
        (root / contig).mkdir()


def test_sparsevar2_reads_meta(tmp_path):
    _write_fixture(tmp_path)
    sv = SparseVar2(tmp_path)
    assert sv.available_samples == ["S0", "S1"]
    assert sv.contigs == ["chr1", "chr2"]
    assert sv.ploidy == 2
    assert sv.n_samples == 2
    assert sv.format_version == 1


def test_sparsevar2_opens_a_reader_per_contig(tmp_path):
    _write_fixture(tmp_path)
    sv = SparseVar2(tmp_path)
    assert set(sv._readers) == {"chr1", "chr2"}
    assert all(isinstance(r, _core.PyContigReader) for r in sv._readers.values())


def test_svar2_available_samples(svar2_store):
    sv = SparseVar2(str(svar2_store))
    assert isinstance(sv.available_samples, list)
    assert sv.n_samples == len(sv.available_samples)
    assert not hasattr(sv, "samples")  # renamed in 3.0.0


def test_svar2_raw_methods_are_private(svar2_store):
    sv = SparseVar2(str(svar2_store))
    for public in ("overlap_batch", "find_ranges", "gather_ranges"):
        assert not hasattr(sv, public), f"{public} should be privatized in 3.0.0"
    for private in ("_overlap_batch", "_find_ranges", "_gather_ranges"):
        assert callable(getattr(sv, private))
