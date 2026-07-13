import hashlib
from pathlib import Path

import pytest

from genoray import SparseVar2


def _dir_digest(root: Path) -> dict[str, str]:
    """Map every non-meta file under a store to a hash of its bytes."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "meta.json":
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture
def two_contig_store(tmp_path: Path) -> SparseVar2:
    """Reuse the session svar2 fixture's builder but with two contigs (chr1, chr2)."""
    # Build via SparseVar2.from_vcf on a 2-contig inline VCF; see conftest svar2_store.
    from tests.conftest import build_two_contig_svar2  # helper added in Step 3

    return build_two_contig_svar2(tmp_path)


def test_subset_contigs_narrows_meta_and_copies_bytes(two_contig_store, tmp_path):
    out = tmp_path / "chr1only.svar2"
    two_contig_store.subset_contigs(out, "chr1", overwrite=True)
    sub = SparseVar2(out)
    assert sub.contigs == ["chr1"]
    assert sub.available_samples == two_contig_store.available_samples
    assert sub.ploidy == two_contig_store.ploidy
    # per-contig bytes identical to source
    src_c1 = _dir_digest(two_contig_store.path / "chr1")
    out_c1 = _dir_digest(out / "chr1")
    assert src_c1 == out_c1


def test_subset_contigs_rejects_unknown(two_contig_store, tmp_path):
    with pytest.raises(ValueError, match="not in store"):
        two_contig_store.subset_contigs(tmp_path / "x.svar2", ["chrZ"])


def test_subset_contigs_refuses_in_place(two_contig_store):
    with pytest.raises(ValueError, match="in place"):
        two_contig_store.subset_contigs(two_contig_store.path, ["chr1"], overwrite=True)


def test_split_by_contig_explodes(two_contig_store, tmp_path):
    paths = two_contig_store.split_by_contig(tmp_path / "split", overwrite=True)
    assert [p.name for p in paths] == ["chr1.svar2", "chr2.svar2"]
    assert SparseVar2(paths[0]).contigs == ["chr1"]
    assert SparseVar2(paths[1]).contigs == ["chr2"]
