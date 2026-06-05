"""Filter-inheritance tests for VCF/PGEN -> SVAR.

Replaces the old skip_symbolic_alts flag tests: symbolic filtering is now just
`pl_filter=~exprs.is_symbolic` (+ paired cyvcf2 `filter`), and SparseVar inherits
the source's filter.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from vcfixture import Number, Seq, Sym, Type, VcfBuilder, VcfVersion

from genoray import PGEN, VCF, SparseVar
from genoray import exprs as gexprs


def _not_symbolic(rec) -> bool:
    return not any(a.startswith("<") for a in rec.ALT)


def _mixed_vcf(tmp_path: Path) -> Path:
    """chr1: SNV A>T@100, <DEL>@200, <INS>@300, ins G>GAT@400."""
    b = (
        VcfBuilder(
            samples=["s1", "s2"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .info("SVLEN")
        .info("SVCLAIM")
        .info("END")
        .fmt("GT")
    )
    b.record("chr1", 100, ref="A", alt=[Seq("T")], gt=["0|1", "1|1"])
    b.record(
        "chr1",
        200,
        ref="A",
        alt=[Sym.deletion()],
        gt=["0|1", "0|0"],
        info={"SVLEN": [50], "SVCLAIM": ["D"], "END": [250]},
    )
    b.record(
        "chr1",
        300,
        ref="C",
        alt=[Sym.insertion()],
        gt=["0|0", "0|1"],
        info={"SVLEN": [60]},
    )
    b.record("chr1", 400, ref="G", alt=[Seq("GAT")], gt=["1|1", "0|1"])
    return b.write(tmp_path / "mixed.vcf.gz", bgzip=True, index=True)


def test_vcf_load_index_list_typed_filter(tmp_path):
    """A list-typed pl_filter (is_symbolic) must evaluate on the VCF path."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path, filter=_not_symbolic, pl_filter=~gexprs.is_symbolic)
    v._write_gvi_index()
    v._load_index()
    assert v._index is not None
    assert v._index.height == 2
    assert v._index["POS"].to_list() == [100, 400]


def test_from_vcf_inherits_symbolic_filter(tmp_path):
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path, filter=_not_symbolic, pl_filter=~gexprs.is_symbolic)
    out = tmp_path / "out.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)

    sv = SparseVar(out)
    assert sv.n_variants == 2
    assert sv.index["POS"].to_list() == [100, 400]

    # Genotype scan and index must agree: the sparse data stores global variant
    # indices (0 = POS100, 1 = POS400).  If chunk() ignored self._filter the
    # worker would scan all 4 variants and assign indices 0-3, so POS400 would
    # land at index 3 instead of 1 — a value-level mismatch caught below.
    #
    # Fixture genotypes after filtering to {POS100, POS400}:
    #   POS100 (A>T):   s1=0|1, s2=1|1  -> 3 alt haplotypes -> 3 × var_idx 0
    #   POS400 (G>GAT): s1=1|1, s2=0|1  -> 3 alt haplotypes -> 3 × var_idx 1
    genos = sv.read_ranges("chr1", 0, 1_000_000)
    # read_ranges returns Ragged[V_IDX_TYPE]; .data is the flat array of
    # variant indices for every alt-allele call across all ranges/samples/ploidy.
    assert sorted(genos.data.tolist()) == [0, 0, 0, 1, 1, 1]


def test_from_vcf_inherits_general_filter(tmp_path):
    """A non-symbolic filter (is_snp) is also honored, proving the path is general."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(
        vcf_path,
        filter=lambda rec: len(rec.REF) == 1 and all(len(a) == 1 for a in rec.ALT),
        pl_filter=gexprs.is_snp,
    )
    out = tmp_path / "snp.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    # Only the SNV A>T@100 is a pure SNP (200/300 symbolic, 400 indel).
    assert sv.n_variants == 1
    assert sv.index["POS"].to_list() == [100]
    # POS100 (A>T): s1=0|1, s2=1|1 -> 3 alt calls, all variant index 0
    genos = sv.read_ranges("chr1", 0, 1_000_000)
    assert sorted(genos.data.tolist()) == [0, 0, 0]


def test_from_vcf_no_filter_keeps_all(tmp_path):
    """Back-compat: no filter -> all records written."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path)
    out = tmp_path / "all.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    assert sv.n_variants == 4


def _mixed_pgen(tmp_path: Path) -> Path:
    """Convert the mixed VCF to PGEN via plink2 (symbolic alleles carried verbatim)."""
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")
    vcf_path = _mixed_vcf(tmp_path)
    prefix = tmp_path / "mixed"
    subprocess.run(
        [
            "plink2",
            "--vcf",
            str(vcf_path),
            "--make-pgen",
            "--out",
            str(prefix),
            "--allow-extra-chr",
        ],
        check=True,
        capture_output=True,
    )
    return prefix.with_suffix(".pgen")


def test_from_pgen_inherits_symbolic_filter(tmp_path):
    pgen_path = _mixed_pgen(tmp_path)
    pgen = PGEN(pgen_path, filter=~gexprs.is_symbolic)
    out = tmp_path / "pg.svar"
    SparseVar.from_pgen(out, pgen, max_mem="1g", overwrite=True)

    sv = SparseVar(out)
    assert sv.n_variants == 2
    # symbolic POS 200/300 dropped; precise 100/400 kept
    assert set(sv.index["POS"].to_list()) == {100, 400}

    # Alignment guard: sparse data must only reference output indices {0, 1}.
    # Under the old bug (no filter applied during from_pgen) the worker would
    # scan all 4 physical variants and assign output indices 0-3, so POS400
    # would land at index 3 instead of 1 — caught by asserting no index ≥ 2.
    # Also assert the full multiset: 3 alt-calls per kept variant (matching the
    # VCF fixture), confirming plink2 did not swap alleles for these records.
    #   POS100 (A>T):   s1=0|1, s2=1|1  -> 3 alt haplotypes -> 3 × var_idx 0
    #   POS400 (G>GAT): s1=1|1, s2=0|1  -> 3 alt haplotypes -> 3 × var_idx 1
    genos = sv.read_ranges("chr1", 0, 1_000_000)
    assert sorted(genos.data.tolist()) == [0, 0, 0, 1, 1, 1]


def test_from_pgen_no_filter_keeps_all(tmp_path):
    pgen_path = _mixed_pgen(tmp_path)
    pgen = PGEN(pgen_path)
    out = tmp_path / "pg_all.svar"
    SparseVar.from_pgen(out, pgen, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    assert sv.n_variants == 4


def test_cli_write_skip_symbolic_vcf(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    vcf_path = _mixed_vcf(tmp_path)
    out = tmp_path / "cli.svar"
    cli_write(vcf_path, out, max_mem="1g", overwrite=True, no_symbolic=True)
    sv = SparseVar(out)
    assert sv.n_variants == 2
    assert sv.index["POS"].to_list() == [100, 400]


def test_cli_write_skip_symbolic_pgen(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    pgen_path = _mixed_pgen(tmp_path)
    out = tmp_path / "cli_pg.svar"
    cli_write(pgen_path, out, max_mem="1g", overwrite=True, no_symbolic=True)
    sv = SparseVar(out)
    assert sv.n_variants == 2
    assert set(sv.index["POS"].to_list()) == {100, 400}


# ---------------------------------------------------------------------------
# Multi-contig filtered tests (Fix 2)
# ---------------------------------------------------------------------------


def _flat_var_idxs(ragged) -> list[int]:
    """Extract the actual per-slot variant indices from a Ragged result.

    In a multi-contig SVAR, ``ragged.data`` is the full backing store (all
    contigs).  Reading per-slot via the Ragged offsets gives only the elements
    that fall within the queried contig's global index range.
    """
    starts, ends = ragged.offsets[0].tolist(), ragged.offsets[1].tolist()
    result = []
    for s, e in zip(starts, ends):
        if e > s:
            result.extend(ragged.data[s:e].tolist())
    return result


def _two_contig_vcf(tmp_path: Path) -> Path:
    """Two contigs, one symbolic record dropped per contig.

    chr1: SNV A>T@100, <DEL>@200
    chr2: SNV C>G@500, <INS>@600

    After symbolic filter: chr1@100 (idx 0), chr2@500 (idx 1).
    """
    b = (
        VcfBuilder(
            samples=["s1", "s2"],
            contigs=[("chr1", 1_000_000), ("chr2", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .info("SVLEN")
        .info("SVCLAIM")
        .info("END")
        .fmt("GT")
    )
    # chr1
    b.record("chr1", 100, ref="A", alt=[Seq("T")], gt=["0|1", "1|1"])
    b.record(
        "chr1",
        200,
        ref="A",
        alt=[Sym.deletion()],
        gt=["0|1", "0|0"],
        info={"SVLEN": [50], "SVCLAIM": ["D"], "END": [250]},
    )
    # chr2
    b.record("chr2", 500, ref="C", alt=[Seq("G")], gt=["1|0", "0|1"])
    b.record(
        "chr2",
        600,
        ref="T",
        alt=[Sym.insertion()],
        gt=["0|0", "1|0"],
        info={"SVLEN": [30]},
    )
    return b.write(tmp_path / "two_contig.vcf.gz", bgzip=True, index=True)


def test_from_vcf_multi_contig_filtered(tmp_path):
    """from_vcf with symbolic filter across 2 contigs: correct n_variants, per-contig
    POS, and genotype/index alignment."""
    vcf_path = _two_contig_vcf(tmp_path)
    v = VCF(vcf_path, filter=_not_symbolic, pl_filter=~gexprs.is_symbolic)
    out = tmp_path / "two_contig.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True)

    sv = SparseVar(out)
    assert sv.n_variants == 2
    # Per-contig POS check via index
    idx = sv.index
    assert idx.filter(pl.col("CHROM") == "chr1")["POS"].to_list() == [100]
    assert idx.filter(pl.col("CHROM") == "chr2")["POS"].to_list() == [500]

    # chr1: s1=0|1, s2=1|1 → 3 alt calls, all var_idx 0
    g1 = sv.read_ranges("chr1", 0, 1_000_000)
    assert sorted(_flat_var_idxs(g1)) == [0, 0, 0]

    # chr2: s1=1|0, s2=0|1 → 2 alt calls, both var_idx 1
    g2 = sv.read_ranges("chr2", 0, 1_000_000)
    assert sorted(_flat_var_idxs(g2)) == [1, 1]


def test_from_pgen_multi_contig_filtered(tmp_path):
    """from_pgen with symbolic filter across 2 contigs: exercises the compacted
    chunk_idx path (Fix 1 footgun class) and asserts cross-contig ordering."""
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")

    vcf_path = _two_contig_vcf(tmp_path)
    prefix = tmp_path / "two_contig"
    subprocess.run(
        [
            "plink2",
            "--vcf",
            str(vcf_path),
            "--make-pgen",
            "--out",
            str(prefix),
            "--allow-extra-chr",
        ],
        check=True,
        capture_output=True,
    )
    pgen_path = prefix.with_suffix(".pgen")
    pgen = PGEN(pgen_path, filter=~gexprs.is_symbolic)
    out = tmp_path / "two_contig_pg.svar"
    SparseVar.from_pgen(out, pgen, max_mem="1g", overwrite=True)

    sv = SparseVar(out)
    assert sv.n_variants == 2
    # plink2 strips the "chr" prefix, so contigs may be "1"/"2" or "chr1"/"chr2".
    # Assert per-contig POS using ContigNormalizer-aware queries via read_ranges.
    assert set(sv.index["POS"].to_list()) == {100, 500}
    # Contig-0 (POS=100) should map to global variant index 0.
    contig0, contig1 = sv.contigs[0], sv.contigs[1]

    # Only output indices {0, 1} should appear; no stale 2/3 from unfiltered scan.
    g1 = sv.read_ranges(contig0, 0, 1_000_000)
    assert sorted(_flat_var_idxs(g1)) == [0, 0, 0]
    g2 = sv.read_ranges(contig1, 0, 1_000_000)
    assert sorted(_flat_var_idxs(g2)) == [1, 1]


# ---------------------------------------------------------------------------
# Dosage filter test (Fix 3)
# ---------------------------------------------------------------------------


def _dosage_vcf(tmp_path: Path) -> Path:
    """chr1: SNV A>T@100 with DS, <DEL>@200 (symbolic, will be filtered).

    DS values:
      s1 (hom-alt A>T):  DS=[1.9]
      s2 (het A>T):      DS=[0.8]
    """
    b = (
        VcfBuilder(
            samples=["s1", "s2"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .info("SVLEN")
        .info("SVCLAIM")
        .info("END")
        .fmt("GT")
        .fmt("DS", Number.A, Type.FLOAT)
    )
    b.record(
        "chr1",
        100,
        ref="A",
        alt=[Seq("T")],
        gt=["1|1", "0|1"],
        DS=[[1.9], [0.8]],
    )
    b.record(
        "chr1",
        200,
        ref="A",
        alt=[Sym.deletion()],
        gt=["0|1", "0|0"],
        DS=[[0.0], [0.0]],
        info={"SVLEN": [50], "SVCLAIM": ["D"], "END": [250]},
    )
    return b.write(tmp_path / "dosage.vcf.gz", bgzip=True, index=True)


def test_from_vcf_filtered_with_dosages(tmp_path):
    """filtered from_vcf with with_dosages=True: only the kept variant's
    dosages appear in the SVAR; the symbolic record's dosages are absent."""
    vcf_path = _dosage_vcf(tmp_path)
    v = VCF(
        vcf_path,
        filter=_not_symbolic,
        pl_filter=~gexprs.is_symbolic,
        dosage_field="DS",
    )
    out = tmp_path / "dosage.svar"
    SparseVar.from_vcf(out, v, max_mem="1g", overwrite=True, with_dosages=True)

    sv = SparseVar(out)
    assert sv.n_variants == 1
    assert sv.index["POS"].to_list() == [100]

    # Read back via with_fields so dosages Ragged is populated.
    sv_fields = sv.with_fields(["dosages"])
    result = sv_fields.read_ranges("chr1", 0, 1_000_000)
    # result is an awkward record array; .dosages is Ragged[float32]
    # Flat data: one dosage per alt-call (s1 hom-alt → 2 calls, s2 het → 1 call).
    dosages_flat = result["dosages"].data.tolist()
    assert len(dosages_flat) == 3
    # All non-zero (the kept variant has real dosages; symbolic zeros never written)
    assert all(d > 0.0 for d in dosages_flat)
    # s1 has DS=1.9 for both haplotypes (two calls); s2 has DS=0.8 (one call)
    assert sorted(round(d, 1) for d in dosages_flat) == [0.8, 1.9, 1.9]


def test_record_predicates_mirror_exprs():
    from genoray.exprs import _record_is_breakend, _record_is_symbolic

    # symbolic: any ALT starting with "<"
    assert _record_is_symbolic(["<DEL>"]) is True
    assert _record_is_symbolic(["A", "<INS>"]) is True
    assert _record_is_symbolic(["A", "T"]) is False

    # breakend: any ALT in mate-pair or single-breakend notation
    assert _record_is_breakend(["G[chr1:500000["]) is True
    assert _record_is_breakend(["]chr2:321]G"]) is True
    assert _record_is_breakend([".TGCA"]) is True
    assert _record_is_breakend(["TGCA."]) is True
    assert _record_is_breakend(["A", "T"]) is False
    # symbolic alleles are NOT breakends (distinct ALT class)
    assert _record_is_breakend(["<DEL>"]) is False
