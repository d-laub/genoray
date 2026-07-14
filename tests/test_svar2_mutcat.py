from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl
import pysam
import pytest

from genoray import SparseVar, SparseVar2
from genoray._reference import Reference
from tests.test_svar2_from_vcf import _write_ref, _write_vcf


def test_mutation_matrix_shape_and_labels(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store.svar2"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)

    sv2 = SparseVar2(out)
    sv2.annotate_mutations(ref)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["mutcat_contigs"] == ["chr1"]

    mm = sv2.mutation_matrix("SBS96", count="allele")
    assert mm.columns[0] == "MutationType"
    assert mm.height == 96
    assert set(mm.columns[1:]) == set(sv2.available_samples)
    assert mm.select(pl.exclude("MutationType")).to_numpy().dtype.kind in "iu"


def test_mutation_matrix_requires_annotation(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store_unannotated.svar2"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)

    sv2 = SparseVar2(out)
    with pytest.raises(ValueError, match="not annotated"):
        sv2.mutation_matrix("SBS96")


# ---------------------------------------------------------------------------
# Task 14: v1 <-> SVAR2 parity + write-time/post-hoc round-trip
# ---------------------------------------------------------------------------

# 80bp non-repetitive reference so indel regions have a single unambiguous
# left-aligned representation (no homopolymer/repeat context to shift).
_PARITY_SEQ = (
    "AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGGCAACGACATGTGCGGCGACCCTTGCGACAGTGACGCTT"
)


def _write_parity_ref(d: Path) -> Path:
    ref = d / "parity_ref.fa"
    ref.write_text(f">chr1\n{_PARITY_SEQ}\n")
    pysam.faidx(str(ref))
    return ref


def _write_parity_vcf(d: Path) -> Path:
    """Build a shared VCF exercising SBS96, ID83, and DBS78 for both v1 and v2.

    Variants (1-based POS, all against ``_PARITY_SEQ``):

    - POS=10  A>G   isolated SNV on S0 (het)                       -> SBS96
    - POS=30  A>C   \\ adjacent SNV pair on S1's hap0 (phased 1|0,  -> DBS78
    - POS=31  T>A   /  1|0) -- the critical "count once" case
    - POS=45  A>G   isolated SNV on S0 (het, other haplotype)      -> SBS96
    - POS=50  A>AG  pure insertion on S0 (het)                     -> ID83
    - POS=65  TG>T  pure deletion on S1 (het)                      -> ID83

    All positions are spaced >=5bp apart except the intentional POS=30/31
    doublet, so no other variant accidentally collapses into a DBS pair.
    """
    vcf_path = d / "parity_in.vcf"
    h = pysam.VariantHeader()
    h.add_line(f"##contig=<ID=chr1,length={len(_PARITY_SEQ)}>")
    h.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    h.add_sample("S0")
    h.add_sample("S1")

    # (0-based start, REF, ALT, {sample: (hap0, hap1)})
    records = [
        (9, "A", "G", {"S0": (0, 1), "S1": (0, 0)}),  # POS=10 isolated SNV
        (29, "A", "C", {"S0": (0, 0), "S1": (1, 0)}),  # POS=30 doublet part 1
        (30, "T", "A", {"S0": (0, 0), "S1": (1, 0)}),  # POS=31 doublet part 2
        (44, "A", "G", {"S0": (1, 0), "S1": (0, 0)}),  # POS=45 isolated SNV
        (49, "A", "AG", {"S0": (0, 1), "S1": (0, 0)}),  # POS=50 insertion
        (64, "TG", "T", {"S0": (0, 0), "S1": (0, 1)}),  # POS=65 deletion
    ]
    with pysam.VariantFile(str(vcf_path), "w", header=h) as vf:
        for start, ref, alt, gts in records:
            r = h.new_record(contig="chr1", start=start, alleles=(ref, alt))
            for s, gt in gts.items():
                r.samples[s]["GT"] = gt
                r.samples[s].phased = True
            vf.write(r)

    vcf_gz = d / "parity_in.vcf.gz"
    with open(vcf_gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf_path)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)
    return vcf_gz


def test_matrix_parity_with_v1(tmp_path: Path):
    """v1 ``SparseVar`` and ``SparseVar2`` must classify+count identically.

    Converts one shared VCF+FASTA both ways, annotates both, and checks
    ``mutation_matrix`` for every kind x count combination is bit-for-bit
    equal. The fixture includes an isolated SNV pair (SBS96 case), a pure
    insertion and a pure deletion (ID83 case), and one ISOLATED adjacent-SNV
    doublet on the same haplotype (DBS78 case) -- this last one is the
    critical check that both stores count a doublet exactly ONCE (v1 via the
    DBS_PARTNER sentinel, SVAR2 via its equivalent fix).
    """
    fa = _write_parity_ref(tmp_path)
    vcf = _write_parity_vcf(tmp_path)

    # v1: SparseVar.from_vcf takes a genoray.VCF wrapper (not a bare path) and
    # a required max_mem; it has no reference= kwarg (v1 does not normalize
    # against a reference at write time -- only annotate_mutations does).
    from genoray import VCF as _V1VCF

    v1_out = tmp_path / "v1.svar"
    SparseVar.from_vcf(v1_out, _V1VCF(str(vcf)), max_mem="10m", overwrite=True)
    v1 = SparseVar(v1_out)
    v1.annotate_mutations(Reference.from_path(fa), write_back=True)

    # v2
    v2_out = tmp_path / "v2.svar2"
    SparseVar2.from_vcf(v2_out, vcf, fa, overwrite=True, threads=1)
    v2 = SparseVar2(v2_out)
    v2.annotate_mutations(fa)

    dbs_nonzero = False
    for kind in ("SBS96", "ID83", "DBS78"):
        for count in ("allele", "sample"):
            a = v1.mutation_matrix(kind, count=count)
            b = v2.mutation_matrix(kind, count=count)
            cols = ["MutationType", *v2.available_samples]
            a = a.select(cols)
            b = b.select(cols)
            assert a.equals(b), f"{kind}/{count} mismatch:\nv1:\n{a}\nv2:\n{b}"

            total = a.select(v2.available_samples).sum().sum_horizontal().item()
            assert total > 0, f"{kind}/{count} matrix is unexpectedly all-zero"
            if kind == "DBS78":
                dbs_nonzero = True

    assert dbs_nonzero, "DBS78 comparison did not run"
    # Explicitly confirm the doublet was counted exactly ONCE (not twice, not
    # dropped) in both stores' allele-count matrices.
    d1 = v1.mutation_matrix("DBS78", count="allele")
    d2 = v2.mutation_matrix("DBS78", count="allele")
    assert d1.select("S1").sum().item() == 1
    assert d2.select("S1").sum().item() == 1


def _write_parity_ref_softmasked(d: Path) -> Path:
    """Same sequence as ``_write_parity_ref`` but entirely lower-case, simulating
    a soft-masked reference (e.g. UCSC hg38, ~50% lower-case repeat-mask)."""
    ref = d / "parity_ref_softmasked.fa"
    ref.write_text(f">chr1\n{_PARITY_SEQ.lower()}\n")
    pysam.faidx(str(ref))
    return ref


def test_annotate_handles_softmasked_reference(tmp_path: Path):
    """Post-hoc ``annotate_mutations`` must classify identically whether the
    reference FASTA is upper- or lower-case (soft-masked).

    Regression for a bug where write-time ``from_vcf(signatures=True)``
    uppercases the reference it loads (``load_contig_seq``) but post-hoc
    ``annotate_mutations`` (via ``Reference._load_contig``, which only does
    ``seq.encode("ascii")``) did not. The shared Rust classifiers'
    ``base_index`` matches uppercase A/C/G/T only, so a soft-masked
    reference would silently classify every SNV/indel in a lower-case region
    as UNCLASSIFIED post-hoc while write-time counted them -- corrupting the
    "post-hoc == write-time" invariant on any real (soft-masked) genome.

    Builds two identical stores from the same VCF (using the upper-case
    reference for conversion, so REF/ALT validation is unaffected), then
    annotates one post-hoc with the upper-case reference and the other with
    an all-lower-case version of the exact same sequence. Without the fix
    the lower-case run returns all-UNCLASSIFIED/zero matrices; with the fix
    the two matrices are bit-for-bit identical.
    """
    fa_upper = _write_parity_ref(tmp_path)
    fa_soft = _write_parity_ref_softmasked(tmp_path)
    vcf = _write_parity_vcf(tmp_path)

    out_upper = tmp_path / "softmask_upper.svar2"
    SparseVar2.from_vcf(out_upper, vcf, fa_upper, overwrite=True, threads=1)
    sv_upper = SparseVar2(out_upper)
    sv_upper.annotate_mutations(fa_upper)

    out_soft = tmp_path / "softmask_lower.svar2"
    SparseVar2.from_vcf(out_soft, vcf, fa_upper, overwrite=True, threads=1)
    sv_soft = SparseVar2(out_soft)
    sv_soft.annotate_mutations(fa_soft)

    for kind in ("SBS96", "ID83"):
        m_upper = sv_upper.mutation_matrix(kind)
        m_soft = sv_soft.mutation_matrix(kind)
        assert m_upper.equals(m_soft), (
            f"{kind} mismatch between upper-case and soft-masked reference:\n"
            f"upper:\n{m_upper}\nsoft-masked:\n{m_soft}"
        )
        total = m_upper.select(sv_upper.available_samples).sum().sum_horizontal().item()
        assert total > 0, f"{kind} matrix is unexpectedly all-zero"


def test_write_time_signatures_match_posthoc(tmp_path: Path):
    """Write-time (`signatures=True`) and post-hoc `annotate_mutations` must
    produce identical mutation matrices.

    The two stores may differ in var_key/dense *routing* since write-time
    signature classification is factored into the cost model that decides
    routing, but `mutation_matrix` counts over the classified codes
    irrespective of routing, so the matrices must match exactly.
    """
    fa = _write_parity_ref(tmp_path)
    vcf = _write_parity_vcf(tmp_path)

    a_out = tmp_path / "wtime.svar2"
    SparseVar2.from_vcf(a_out, vcf, fa, signatures=True, overwrite=True, threads=1)
    b_out = tmp_path / "posthoc.svar2"
    SparseVar2.from_vcf(b_out, vcf, fa, overwrite=True, threads=1)
    sb = SparseVar2(b_out)
    sb.annotate_mutations(fa)
    sa = SparseVar2(a_out)

    for kind in ("SBS96", "ID83"):
        ma = sa.mutation_matrix(kind)
        mb = sb.mutation_matrix(kind)
        assert ma.equals(mb), f"{kind} mismatch:\nwrite-time:\n{ma}\npost-hoc:\n{mb}"
        total = ma.select(sa.available_samples).sum().sum_horizontal().item()
        assert total > 0, f"{kind} matrix is unexpectedly all-zero"


def test_annotate_accepts_strand_arrays_smoke(tmp_path: Path):
    import numpy as np

    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store_strand_smoke.svar2"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)
    sv2 = SparseVar2(out)
    reader = sv2._readers["chr1"]
    seq = Reference.from_path(ref).contig_array("chr1").astype(np.uint8, copy=False)
    # Empty strand partition (all N) — just proves the 6-arg binding is callable.
    reader.annotate_mutations(
        str(out),
        "chr1",
        seq,
        np.empty(0, np.int32),
        np.empty(0, np.int32),
        np.empty(0, np.uint8),
    )
    assert (out / "chr1" / "mutcat" / "var_key_snp" / "strand.bin").exists()
