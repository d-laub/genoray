from __future__ import annotations

import json

import numpy as np
import polars as pl
import pysam
import pytest
from loguru import logger

import genoray
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
    # 3 variants on chr1 at 1-based POS 2,3,9 (VCF convention); all SNVs
    # Reference chr1: A C G T A C G T A C (0-based 0..9)
    # POS=2 -> 0-based 1 -> REF=C; POS=3 -> 0-based 2 -> REF=G; POS=9 -> 0-based 8 -> REF=A
    # Note: _load_index adds 'index' as a row-index column; the on-disk file
    # must NOT include it.
    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1"],
            "POS": np.array([2, 3, 9], dtype=np.int32),
            "REF": ["C", "G", "A"],
            "ALT": [["A"], ["T"], ["C"]],
            "ILEN": pl.Series([[0], [0], [0]], dtype=pl.List(pl.Int32)),
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

    One variant: chr1 POS=8 (1-based), T>C (isolated SNV -> SBS).
    Sample 0 is HOMOZYGOUS: hap0=[0], hap1=[0].
    Sample 1 is REF: hap0=[], hap1=[].
    offsets length = n_samples*ploidy + 1 = 2*2 + 1 = 5.
    offsets = [0, 1, 2, 2, 2]
    """
    from genoray._svar import SparseVarMetadata, _write_genos
    from seqpro.rag import Ragged

    path.mkdir(parents=True)
    # Reference chr1: A C G T A C G T A C (0-based indices)
    # 1-based POS=8 -> 0-based index 7 -> ref base T
    # Flanks: 0-based 6=G (5') and 8=A (3').
    # Pyrimidine ref T -> label: G[T>C]A
    index = pl.DataFrame(
        {
            "CHROM": ["chr1"],
            "POS": np.array([8], dtype=np.int32),
            "REF": ["T"],
            "ALT": [["C"]],
            "ILEN": pl.Series([[0]], dtype=pl.List(pl.Int32)),
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


def test_public_reference_export():
    assert hasattr(genoray, "Reference")
    from genoray import Reference as R

    assert R is genoray.Reference


def test_write_view_mutcat_explicit_without_reference_raises(annotated_svar, tmp_path):
    """write_view with fields=["mutcat"] and no reference must raise ValueError."""
    svar = SparseVar(annotated_svar)
    assert "mutcat" in svar.available_fields, (
        "fixture must have mutcat to test the error path"
    )
    out = tmp_path / "view.svar"
    with pytest.raises(ValueError, match="mutcat"):
        svar.write_view(
            regions=("chr1", 0, 100),
            samples=svar.available_samples,
            output=out,
            fields=["mutcat"],
            # reference=None (default) — should raise
        )


def test_mutcat_staleness_warning(tmp_path):
    # --- build a tiny reference + single-SNV VCF ---

    seq = "ACGTACGTACGTACGT"
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    pysam.faidx(str(fa))

    vcf = tmp_path / "t.vcf"
    h = pysam.VariantHeader()
    h.add_line("##contig=<ID=chr1,length=16>")
    h.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">')
    h.add_sample("S1")
    with pysam.VariantFile(str(vcf), "w", header=h) as vf:
        r = h.new_record(contig="chr1", start=4, alleles=("A", "C"))  # 1-based POS=5
        r.samples["S1"]["GT"] = (0, 1)
        vf.write(r)
    pysam.tabix_index(str(vcf), preset="vcf", force=True)

    svp = tmp_path / "sv.svar"
    genoray.SparseVar.from_vcf(
        svp, genoray.VCF(str(vcf) + ".gz"), max_mem="1g", overwrite=True
    )
    sv = SparseVar(svp)
    sv.annotate_mutations(Reference.from_path(fa), write_back=True)

    # corrupt the persisted version to look stale
    meta_path = svp / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["mutcat_version"] = 0
    meta_path.write_text(json.dumps(meta))

    # capture loguru output
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        SparseVar(svp, fields=["mutcat"])
    finally:
        logger.remove(sink_id)

    assert any("older version" in m for m in messages), messages


def test_write_view_recomputes_mutcat_with_reference(annotated_svar, tmp_path):
    """write_view with reference= recomputes mutcat on the subset view."""
    svar = SparseVar(annotated_svar)
    assert "mutcat" in svar.available_fields, (
        "fixture must have mutcat so the source is annotated"
    )
    # The fixture writes ref.fa alongside the svar dir (same tmp_path).
    ref_fa = annotated_svar.parent / "ref.fa"
    assert ref_fa.exists(), f"Reference FASTA not found at {ref_fa}"

    out = tmp_path / "view.svar"
    # Keep all samples and all variants (chr1 POS 1,2,8 → region 0..100).
    svar.write_view(
        regions=("chr1", 0, 100),
        samples=svar.available_samples,
        output=out,
        reference=ref_fa,  # triggers recomputation
    )

    sv2 = SparseVar(out)
    # mutcat should be present on the output (recomputed, not stale)
    assert "mutcat" in sv2.available_fields, (
        "write_view with reference= must produce a mutcat field on the view"
    )
    # mutation_matrix must work and give sane counts
    sbs = sv2.mutation_matrix("SBS96")
    assert sbs.height == 96
    assert set(sv2.available_samples).issubset(set(sbs.columns))
    # The subset has the same variants as the source, so totals should be positive
    total = sbs.select(sv2.available_samples).sum().sum_horizontal().item()
    assert total > 0, "Recomputed mutation matrix should have non-zero counts"

    # DBS: sample s0 carries the two adjacent SNVs -> one DBS event
    dbs = sv2.mutation_matrix("DBS78")
    assert dbs.height == 78
    assert dbs["s0"].sum() == 1, (
        f"Expected s0 to have 1 DBS event in the view, got {dbs['s0'].sum()}"
    )


def test_write_view_recompute_breaks_dbs_when_partner_dropped(annotated_svar, tmp_path):
    """write_view with reference= RECOMPUTES mutcat; dropping the DBS 3' partner
    reclassifies the surviving 5' SNV as SBS rather than leaving a stale DBS code.

    Variant layout (1-based POS): s0 carries POS=2 (DBS 5') and POS=3 (DBS 3').
    Region [1, 2) keeps POS=2 only — the partner at POS=3 is excluded.
    After recompute, the isolated SNV must count as SBS96=1, DBS78=0.
    A positional copy of the old mutcat would wrongly leave DBS78=1, SBS96=0.
    """
    svar = SparseVar(annotated_svar)
    ref_fa = annotated_svar.parent / "ref.fa"
    assert ref_fa.exists(), f"Reference FASTA not found at {ref_fa}"

    out = tmp_path / "view_no_partner.svar"
    # Region [1, 2) is 0-based half-open: covers POS=1 only, NOT POS=2.
    svar.write_view(
        regions=("chr1", 1, 2),
        samples=["s0"],
        output=out,
        reference=ref_fa,
    )

    sv2 = SparseVar(out)
    assert "mutcat" in sv2.available_fields, (
        "write_view with reference= must produce a mutcat field on the view"
    )

    dbs = sv2.mutation_matrix("DBS78")
    sbs = sv2.mutation_matrix("SBS96")

    dbs_sum = dbs["s0"].sum()
    sbs_sum = sbs["s0"].sum()

    assert dbs_sum == 0, (
        f"Expected DBS78 count=0 for s0 after partner dropped, got {dbs_sum}. "
        "A stale positional copy would incorrectly leave DBS=1."
    )
    assert sbs_sum == 1, (
        f"Expected SBS96 count=1 for s0 (isolated SNV reclassified), got {sbs_sum}."
    )


def test_write_view_drops_mutcat_by_default(annotated_svar, tmp_path):
    """write_view with fields=None must NOT carry the derived mutcat field.

    mutcat codes encode cross-variant DBS adjacency that is only valid for the
    full variant set.  Subsetting may drop a DBS partner, leaving a stale 5'
    code that mutation_matrix would miscount.
    """
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    assert "mutcat" in svar.available_fields, (
        "fixture must have mutcat to test the drop"
    )

    out = tmp_path / "view.svar"
    # Keep all samples and a broad region (all 3 variants are on chr1 at POS 1,2,8).
    svar.write_view(
        regions=("chr1", 0, 100),
        samples=svar.available_samples,
        output=out,
        # fields=None is the default — mutcat should be excluded
    )
    sv2 = SparseVar(out)
    assert "mutcat" not in sv2.available_fields, (
        "write_view with default fields=None should NOT carry mutcat; "
        "re-run annotate_mutations on the view instead"
    )


def _toy_sbs_reference():
    # cover all 96 SBS rows with two signatures so any catalogue aligns
    from genoray._mutcat import labels

    rows = labels("SBS96")
    n = len(rows)
    return pl.DataFrame(
        {
            "MutationType": rows,
            "SBS_A": np.linspace(1, 2, n) / np.linspace(1, 2, n).sum(),
            "SBS_B": np.linspace(2, 1, n) / np.linspace(2, 1, n).sum(),
        }
    )


def test_assign_signatures_with_explicit_reference(annotated_svar):
    svar = SparseVar(annotated_svar, fields=["mutcat"])
    ref = _toy_sbs_reference()
    act = svar.assign_signatures("SBS96", reference=ref)
    assert act.columns[0] == "Sample"
    assert "cosine_similarity" in act.columns
    assert set(act["Sample"].to_list()).issubset(set(svar.available_samples))
    # activities are nonnegative
    assert (act.select(["SBS_A", "SBS_B"]).to_numpy() >= 0).all()


def test_issue59_deletion_no_longer_crashes(tmp_path):
    from genoray._mutcat import ID83_OFFSET, N_CODES

    seq = "ACGTTGCAACGTTGCAAGGCCTTAGCATCGTACGATCGTTAGCCATGACTGACATGCATGC"
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    pysam.faidx(str(fa))

    anchor0 = 19
    REF, ALT = seq[anchor0 : anchor0 + 6], seq[anchor0]  # "CCTTAG" -> "C", 5bp del

    vcf = tmp_path / "t.vcf"
    h = pysam.VariantHeader()
    h.add_line("##contig=<ID=chr1,length=60>")
    h.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">')
    h.add_sample("S1")
    with pysam.VariantFile(str(vcf), "w", header=h) as vf:
        r = h.new_record(contig="chr1", start=anchor0, alleles=(REF, ALT))
        r.samples["S1"]["GT"] = (0, 1)
        vf.write(r)
    pysam.tabix_index(str(vcf), preset="vcf", force=True)

    svp = tmp_path / "sv.svar"
    genoray.SparseVar.from_vcf(
        svp, genoray.VCF(str(vcf) + ".gz"), max_mem="1g", overwrite=True
    )
    sv = SparseVar(svp)
    assert sv.index["POS"].to_list() == [20]  # 1-based preserved

    sv.annotate_mutations(Reference.from_path(fa), write_back=False)  # must NOT raise
    codes = sv.fields["mutcat"].data
    # the deletion entry must be a valid ID-83 code, not UNCLASSIFIED/mismatch
    id83_codes = [c for c in codes if ID83_OFFSET <= c < N_CODES]
    assert len(id83_codes) == 1


def test_mutation_matrix_thread_invariant(tmp_path):
    import numba as nb

    # Inline the same construction used by the annotated_svar fixture.
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    svar_dir = tmp_path / "tiny.svar"
    _build_tiny_svar(svar_dir)
    ref = Reference.from_path(fa)

    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        svar = SparseVar(svar_dir)
        svar.annotate_mutations(ref, write_back=True)
        a = SparseVar(svar_dir, fields=["mutcat"]).mutation_matrix("SBS96")

        nb.set_num_threads(max(2, prev))
        svar2 = SparseVar(svar_dir)
        svar2.annotate_mutations(ref, write_back=True)
        b = SparseVar(svar_dir, fields=["mutcat"]).mutation_matrix("SBS96")
    finally:
        nb.set_num_threads(prev)

    assert a.equals(b), "mutation_matrix('SBS96') differs between 1 and >=2 threads"


def test_classify_variants_snv_context_uses_pos_minus_one(tmp_path):
    import polars as pl

    from genoray._mutcat import classify_sbs96, classify_variants

    # Reference where the base at 0-based index differs left vs right of the variant,
    # so a +1 shift would pick a different trinucleotide context.
    seq = "AACGTTGCA"
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    # Variant at 1-based POS=4 => 0-based index 3, REF = seq[3] = "G".
    p0 = 3
    assert seq[p0] == "G"
    index = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [p0 + 1], "REF": ["G"], "ALT": [["A"]]}
    )
    codes = classify_variants(index, ref)

    five = seq[p0 - 1].encode()
    three = seq[p0 + 1].encode()
    expected = classify_sbs96(five, b"G", b"A", three)
    assert int(codes[0]) == expected


def test_metadata_has_mutcat_contigs_default():
    from genoray._svar import SparseVarMetadata

    m = SparseVarMetadata(samples=["s0"], ploidy=1, contigs=["chr1"])
    assert m.mutcat_contigs is None
    # round-trips through JSON
    m2 = SparseVarMetadata.model_validate_json(
        SparseVarMetadata(
            samples=["s0"], ploidy=1, contigs=["chr1"], mutcat_contigs=["chr1"]
        ).model_dump_json()
    )
    assert m2.mutcat_contigs == ["chr1"]
