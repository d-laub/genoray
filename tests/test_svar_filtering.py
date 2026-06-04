"""Filter-inheritance tests for VCF/PGEN -> SVAR.

Replaces the old skip_symbolic_alts flag tests: symbolic filtering is now just
`pl_filter=~exprs.is_symbolic` (+ paired cyvcf2 `filter`), and SparseVar inherits
the source's filter.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from vcfixture import Seq, Sym, VcfBuilder, VcfVersion

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


def test_from_pgen_no_filter_keeps_all(tmp_path):
    pgen_path = _mixed_pgen(tmp_path)
    pgen = PGEN(pgen_path)
    out = tmp_path / "pg_all.svar"
    SparseVar.from_pgen(out, pgen, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    assert sv.n_variants == 4
