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


def test_mutation_matrix_shapes_and_samples(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    sbs = svar.mutation_matrix("SBS96", count="allele")
    assert sbs.columns[0] == "MutationType"
    assert sbs.height == 96
    assert set(svar.available_samples).issubset(set(sbs.columns))

    dbs = svar.mutation_matrix("DBS78")
    assert dbs.height == 78
    # sample s0 has exactly one DBS event
    assert dbs["s0"].sum() == 1


def test_mutation_matrix_requires_annotation(tmp_path):
    # build an un-annotated svar
    d = tmp_path / "x.svar"
    _build_tiny_svar(d)
    svar = SparseVar(d)
    with pytest.raises(ValueError, match="mutcat"):
        svar.mutation_matrix("SBS96")


def test_count_unit_allele_vs_sample(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    a = svar.mutation_matrix("SBS96", count="allele")
    s = svar.mutation_matrix("SBS96", count="sample")
    # totals: allele >= sample (hom collapses to 1 under "sample")
    assert (
        a.select(svar.available_samples).sum().sum_horizontal().item()
        >= s.select(svar.available_samples).sum().sum_horizontal().item()
    )


@pytest.fixture
def annotated_svar_ploidy2(tmp_path):
    """Build a ploidy-2 SVAR with a homozygous SNV carrier + matching reference."""
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    svar_dir = tmp_path / "tiny2.svar"
    _build_ploidy2_svar(svar_dir)
    svar = SparseVar(svar_dir)
    svar.annotate_mutations(Reference.from_path(fa), write_back=True)
    return svar_dir


def _build_ploidy2_svar(path):
    """Write a minimal SVAR with 2 samples, ploidy 2.

    One variant: chr1 POS=7 (0-based), A>C (isolated SNV -> SBS).
    Sample 0 is HOMOZYGOUS: hap0=[0], hap1=[0].
    Sample 1 is REF: hap0=[], hap1=[].
    offsets length = n_samples*ploidy + 1 = 2*2 + 1 = 5.
    offsets = [0, 1, 2, 2, 2]
    """
    from genoray._svar import SparseVarMetadata, _write_genos
    from seqpro.rag import Ragged

    path.mkdir(parents=True)
    # Reference chr1: A C G T A C G T A C (0-based indices)
    # POS=7 -> ref base T, but we use an A context:
    # Let's place the SNV at POS=7 (T>C) — flanks at 6=G and 8=A.
    # Pyrimidine ref T -> label: G[T>C]A
    index = pl.DataFrame(
        {
            "CHROM": ["chr1"],
            "POS": np.array([7], dtype=np.int32),
            "REF": ["T"],
            "ALT": [["C"]],
            "ILEN": np.array([0], dtype=np.int32),
        }
    )
    index.write_ipc(path / "index.arrow")

    # sample 0 hom: both haps carry variant 0
    # sample 1: both haps empty
    data = np.array([0, 0], dtype=np.int32)  # 2 entries total
    offsets = np.array([0, 1, 2, 2, 2], dtype=np.int64)
    genos = Ragged.from_offsets(data, (2, 2, None), offsets)
    _write_genos(path, genos)

    with open(path / "metadata.json", "w") as f:
        f.write(
            SparseVarMetadata(
                version=1, samples=["s0", "s1"], ploidy=2, contigs=["chr1"]
            ).model_dump_json()
        )


def test_count_unit_allele_vs_sample_strict(annotated_svar_ploidy2):
    """Homozygous sample 0 -> allele counts the SNV twice, sample counts it once."""
    svar = SparseVar(annotated_svar_ploidy2, fields=["mutcat"])
    a = svar.mutation_matrix("SBS96", count="allele")
    s = svar.mutation_matrix("SBS96", count="sample")

    a_total = a.select(svar.available_samples).sum().sum_horizontal().item()
    s_total = s.select(svar.available_samples).sum().sum_horizontal().item()

    # allele=2 (both haps), sample=1 (s0 counted once) -> strictly greater
    assert a_total > s_total, (
        f"Expected allele total ({a_total}) > sample total ({s_total}); "
        "homozygous s0 should count as 2 alleles but 1 sample."
    )
    # Also verify the per-sample allele count for s0 is exactly 2
    assert a["s0"].sum() == 2, f"Expected s0 allele sum=2, got {a['s0'].sum()}"
    assert s["s0"].sum() == 1, f"Expected s0 sample sum=1, got {s['s0'].sum()}"


def test_annotate_dbs_partner_present(annotated_svar):
    from genoray._mutcat import SENTINELS, code_ranges

    mut = np.memmap(annotated_svar / "mutcat.npy", dtype=np.int16, mode="r")
    # sample 0's two adjacent SNVs -> [DBS code, DBS_PARTNER]
    lo, hi = code_ranges()["DBS78"]
    assert lo <= mut[0] < hi
    assert mut[1] == SENTINELS["DBS_PARTNER"]
