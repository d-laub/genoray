from __future__ import annotations

import numpy as np
import polars as pl
import pysam
import pytest

from genoray import SparseVar
from genoray._reference import Reference


@pytest.fixture
def annotated_svar(tmp_path):
    """Build a tiny SVAR by hand + a matching reference, then annotate it."""
    # Reference chr1: A C G T A C G T A C  (0..9)
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    svar_dir = tmp_path / "tiny.svar"
    _build_tiny_svar(svar_dir)
    svar = SparseVar(svar_dir)
    svar.annotate_mutations(Reference.from_path(fa), write_back=True)
    return svar_dir


def _build_tiny_svar(path):
    """Write a minimal valid SVAR directory with 2 samples, ploidy 1, 3 SNVs."""
    from genoray._svar import SparseVarMetadata, _write_genos
    from seqpro.rag import Ragged

    path.mkdir(parents=True)
    # 3 variants on chr1 at POS 1,2,8 (0-based); all SNVs
    # Note: _load_index adds 'index' as a row-index column; the on-disk file
    # must NOT include it.
    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1"],
            "POS": np.array([1, 2, 8], dtype=np.int32),
            "REF": ["C", "G", "A"],
            "ALT": [["A"], ["T"], ["C"]],
            "ILEN": np.array([0, 0, 0], dtype=np.int32),
        }
    )
    index.write_ipc(path / "index.arrow")

    # sample 0 carries variants 0 and 1 (adjacent -> DBS); sample 1 carries variant 2
    data = np.array([0, 1, 2], dtype=np.int32)
    offsets = np.array([0, 2, 3], dtype=np.int64)  # (n_samples*ploidy + 1) = 3
    genos = Ragged.from_offsets(data, (2, 1, None), offsets)
    _write_genos(path, genos)

    with open(path / "metadata.json", "w") as f:
        f.write(
            SparseVarMetadata(
                version=1, samples=["s0", "s1"], ploidy=1, contigs=["chr1"]
            ).model_dump_json()
        )


def test_annotate_writes_mutcat_field(annotated_svar):
    assert (annotated_svar / "mutcat.npy").exists()
    # re-open and confirm metadata records it
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    assert "mutcat" in svar.available_fields
    assert svar.available_fields["mutcat"] == np.dtype("int16")


def test_annotate_dbs_partner_present(annotated_svar):
    from genoray._mutcat import SENTINELS

    mut = np.memmap(annotated_svar / "mutcat.npy", dtype=np.int16, mode="r")
    # sample 0's two adjacent SNVs -> [DBS code, DBS_PARTNER]
    assert mut[1] == SENTINELS["DBS_PARTNER"]
