from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from genoray import SparseVar2

# 40 bp reference. 1-based POS 3 = 'A', 7 = 'C', 12..14 = 'GTA'.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

# Phased, no half-calls, no symbolics: plink2's VCF import is lossless here, so
# from_pgen and from_vcf must agree exactly. (A half-call like './1' would NOT
# round-trip: gen_from_vcf.sh passes --vcf-half-call r, which rewrites it.)
_VCF_BODY = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=40>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
    "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
    "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    "chr1\t12\t.\tGTA\tG,GT\t.\t.\t.\tGT\t1|2\t0|1\n"
    "chr1\t20\t.\tT\tA\t.\t.\t.\tGT\t.|.\t1|0\n"
)


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> tuple[Path, Path, Path]:
    """(reference fasta, bgzipped+indexed vcf, pgen) for the same variants."""
    d = tmp_path_factory.mktemp("frompgen")

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    plain = d / "in.vcf"
    plain.write_text(_VCF_BODY)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    return ref, gz, d / "in.pgen"


def _assert_ragged_equal(ragged_pgen, ragged_vcf) -> None:
    """Compare two decoded record Raggeds field-by-field.

    `.data` on a record Ragged (multiple named fields sharing one ragged axis)
    returns a dict of field -> ndarray, not a single array -- compare per field.
    """
    assert ragged_pgen.offsets.tolist() == ragged_vcf.offsets.tolist()
    assert ragged_pgen.data.keys() == ragged_vcf.data.keys()
    for field in ragged_vcf.data:
        assert ragged_pgen.data[field].tolist() == ragged_vcf.data[field].tolist()


def test_from_pgen_matches_from_vcf(sources, tmp_path):
    ref, vcf, pgen = sources
    from_vcf = tmp_path / "vcf.svar2"
    from_pgen = tmp_path / "pgen.svar2"

    SparseVar2.from_vcf(from_vcf, vcf, ref)
    SparseVar2.from_pgen(from_pgen, pgen, ref)

    a = SparseVar2(from_vcf)
    b = SparseVar2(from_pgen)
    assert a.n_samples == b.n_samples == 2

    regions = [(0, len(_REF))]
    ragged_vcf = a.decode("chr1", regions)
    ragged_pgen = b.decode("chr1", regions)
    _assert_ragged_equal(ragged_pgen, ragged_vcf)


def test_from_pgen_requires_exactly_one_of_reference_or_no_reference(sources, tmp_path):
    _, _, pgen = sources
    with pytest.raises(ValueError, match="exactly one"):
        SparseVar2.from_pgen(tmp_path / "a.svar2", pgen)
    with pytest.raises(ValueError, match="exactly one"):
        SparseVar2.from_pgen(tmp_path / "b.svar2", pgen, "ref.fa", no_reference=True)


def test_from_pgen_refuses_to_overwrite(sources, tmp_path):
    ref, _, pgen = sources
    out = tmp_path / "exists.svar2"
    SparseVar2.from_pgen(out, pgen, ref)
    with pytest.raises(FileExistsError):
        SparseVar2.from_pgen(out, pgen, ref)
    SparseVar2.from_pgen(out, pgen, ref, overwrite=True)  # no raise


def test_from_pgen_no_reference_matches_from_vcf(sources, tmp_path):
    """The no_reference path (no REF validation, no left-alignment) must also agree."""
    _, vcf, pgen = sources
    from_vcf = tmp_path / "vcf.svar2"
    from_pgen = tmp_path / "pgen.svar2"

    SparseVar2.from_vcf(from_vcf, vcf, no_reference=True)
    SparseVar2.from_pgen(from_pgen, pgen, no_reference=True)

    a = SparseVar2(from_vcf)
    b = SparseVar2(from_pgen)
    regions = [(0, len(_REF))]
    _assert_ragged_equal(b.decode("chr1", regions), a.decode("chr1", regions))


def test_from_pgen_reads_zstd_pvar(sources, tmp_path):
    """plink2 `vzs` writes a .pvar.zst; the Rust streamer must handle it."""
    ref, vcf, _ = sources
    d = tmp_path / "zst"
    d.mkdir()
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "vzs",
            "--output-chr",
            "chrM",
            "--vcf",
            str(vcf),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    assert (d / "in.pvar.zst").exists()

    out_zst = tmp_path / "zst.svar2"
    out_ref = tmp_path / "plain.svar2"
    SparseVar2.from_pgen(out_zst, d / "in.pgen", ref)
    SparseVar2.from_vcf(out_ref, vcf, ref)

    regions = [(0, len(_REF))]
    a = SparseVar2(out_ref).decode("chr1", regions)
    b = SparseVar2(out_zst).decode("chr1", regions)
    _assert_ragged_equal(b, a)


def test_from_pgen_multi_contig(tmp_path):
    """Contigs are converted from disjoint .pvar index ranges; a two-contig file
    exercises the range computation and the per-contig PgenReader."""
    d = tmp_path / "multi"
    d.mkdir()

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n>chr2\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        "##contig=<ID=chr2,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr2\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )

    from_vcf = tmp_path / "mv.svar2"
    from_pgen = tmp_path / "mp.svar2"
    SparseVar2.from_vcf(from_vcf, gz, ref)
    SparseVar2.from_pgen(from_pgen, d / "in.pgen", ref)

    a, b = SparseVar2(from_vcf), SparseVar2(from_pgen)
    regions = [(0, len(_REF))]
    for contig in ("chr1", "chr2"):
        _assert_ragged_equal(b.decode(contig, regions), a.decode(contig, regions))


# Same variants as `_VCF_BODY`, but with a monomorphic site (REF present, ALT
# `.` -- no alternate allele) inserted *before* the real variants. plink2
# writes such sites as a bare `.` ALT in the .pvar; the .pvar's ALT-derived
# `allele_idx_offsets` must count this as 0 alt alleles for this variant
# without corrupting every variant that follows it.
_VCF_BODY_MONO = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=40>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
    "chr1\t1\t.\tA\t.\t.\t.\t.\tGT\t0|0\t0|0\n"
    "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
    "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    "chr1\t12\t.\tGTA\tG,GT\t.\t.\t.\tGT\t1|2\t0|1\n"
    "chr1\t20\t.\tT\tA\t.\t.\t.\tGT\t.|.\t1|0\n"
)


@pytest.fixture(scope="module")
def sources_mono(tmp_path_factory) -> tuple[Path, Path, Path]:
    """Same shape as `sources`, but the .pvar's first variant is monomorphic
    (ALT '.') -- regression coverage for the allele_idx_offsets corruption bug."""
    d = tmp_path_factory.mktemp("frompgen_mono")

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    plain = d / "in.vcf"
    plain.write_text(_VCF_BODY_MONO)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    return ref, gz, d / "in.pgen"


def test_from_pgen_monomorphic_site_matches_from_vcf(sources_mono, tmp_path):
    """A plink2-emitted monomorphic .pvar row (ALT '.') must decode to zero alt
    alleles, and must not corrupt allele_idx_offsets for the real variants that
    follow it in the file."""
    ref, vcf, pgen = sources_mono
    from_vcf = tmp_path / "vcf.svar2"
    from_pgen = tmp_path / "pgen.svar2"

    SparseVar2.from_vcf(from_vcf, vcf, ref)
    SparseVar2.from_pgen(from_pgen, pgen, ref)

    a = SparseVar2(from_vcf)
    b = SparseVar2(from_pgen)
    assert a.n_samples == b.n_samples == 2

    regions = [(0, len(_REF))]
    ragged_vcf = a.decode("chr1", regions)
    ragged_pgen = b.decode("chr1", regions)
    _assert_ragged_equal(ragged_pgen, ragged_vcf)


def test_from_pgen_missing_pvar_is_a_clear_error(sources, tmp_path):
    _, _, pgen = sources
    lonely = tmp_path / "lonely.pgen"
    lonely.write_bytes(pgen.read_bytes())
    with pytest.raises(FileNotFoundError, match="No .pvar or .pvar.zst"):
        SparseVar2.from_pgen(tmp_path / "x.svar2", lonely, no_reference=True)


def test_from_pgen_check_ref_accepts_x(sources, tmp_path):
    ref, _, pgen = sources
    out = tmp_path / "check_ref_x.svar2"
    SparseVar2.from_pgen(out, pgen, ref, check_ref="x")
    assert (out / "meta.json").exists()


def test_from_pgen_check_ref_invalid_raises(sources, tmp_path):
    ref, _, pgen = sources
    with pytest.raises(ValueError, match="check_ref"):
        SparseVar2.from_pgen(tmp_path / "bad.svar2", pgen, ref, check_ref="z")


def test_from_pgen_regions_restrict(sources, tmp_path):
    """`regions="chr1:1-4"` (0-based `[0, 4)`) covers only the POS-3 SNP."""
    ref, _, pgen = sources
    out = tmp_path / "pg_regions"
    SparseVar2.from_pgen(out, pgen, ref, regions="chr1:1-4")

    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]
    counts = sv.region_counts("chr1", [(0, len(_REF))])
    assert int(counts.sum()) >= 1
    rag = sv.decode("chr1", [(0, len(_REF))])
    assert sorted(set(np.asarray(rag["pos"].data).tolist())) == [2]


def test_from_pgen_regions_overlap_variant_rejects_multiple_regions_per_contig(
    sources, tmp_path
):
    """`regions_overlap="variant"` with 2+ disjoint regions on the same
    contig is unsound (see `genoray._svar2._reject_multiregion_variant`) and
    must raise rather than silently drop variants."""
    ref, _, pgen = sources
    with pytest.raises(ValueError, match="at most one region per contig"):
        SparseVar2.from_pgen(
            tmp_path / "pg_multi_variant",
            pgen,
            ref,
            regions=["chr1:1-4", "chr1:7-12"],
            regions_overlap="variant",
        )


def _cell_pos(rag, sample: int, ploid: int) -> list[int]:
    """The `pos` values decoded for one `(region 0, sample, ploid)` cell.

    `Ragged.data` returns the WHOLE underlying flat buffer regardless of how
    the `Ragged` was indexed -- only `.offsets` narrows -- so slicing a cell
    out requires combining an indexed sub-`Ragged`'s offsets with the
    top-level `.data` array.
    """
    cell = rag[0, sample, ploid]
    flat_offsets = cell.offsets.reshape(-1)
    lo, hi = int(flat_offsets[0]), int(flat_offsets[-1])
    return rag.data["pos"][lo:hi].tolist()


def test_from_pgen_samples_preserve_caller_order_and_reorders_genotypes(
    sources, tmp_path
):
    """`samples=["S1", "S0"]` must both reorder `available_samples` AND put
    S1's actual decoded genotypes under output column 0 -- a real reorder,
    not just a relabeled name list."""
    ref, _, pgen = sources
    full = tmp_path / "pg_full"
    reordered = tmp_path / "pg_reordered"
    SparseVar2.from_pgen(full, pgen, ref)
    SparseVar2.from_pgen(reordered, pgen, ref, samples=["S1", "S0"])

    sv_full = SparseVar2(full)
    sv_re = SparseVar2(reordered)
    assert sv_re.available_samples == ["S1", "S0"]

    regions = [(0, len(_REF))]
    rag_full = sv_full.decode("chr1", regions)
    rag_re = sv_re.decode("chr1", regions)

    # S1 is full-cohort column 1; it must land at reordered column 0.
    for ploid in range(2):
        assert _cell_pos(rag_re, 0, ploid) == _cell_pos(rag_full, 1, ploid)
    # S0 is full-cohort column 0; it must land at reordered column 1.
    for ploid in range(2):
        assert _cell_pos(rag_re, 1, ploid) == _cell_pos(rag_full, 0, ploid)


def test_from_pgen_unknown_sample_raises(sources, tmp_path):
    ref, _, pgen = sources
    with pytest.raises(ValueError, match="not found"):
        SparseVar2.from_pgen(tmp_path / "x", pgen, ref, samples=["NOPE"])


def _write_pgen_with_spanning_deletion(d: Path) -> tuple[Path, Path]:
    """(reference, pgen) with a deletion whose POS sits outside
    ``chr1:7-12`` but whose anchor-trimmed extent reaches into it -- the same
    fixture as `test_svar2_from_vcf.py`'s
    `_write_vcf_with_spanning_deletion`, built through plink2 instead of read
    directly by htslib. ``chr1:5 TACA>T`` (0-based POS 4, deleting the
    anchor-trimmed extent ``[5, 8)``) round-trips through plink2 with POS and
    REF/ALT unchanged (verified empirically -- plink2 does not
    left-align/renormalize this VCF), so the POS-outside/extent-inside
    property holds identically to the VCF fixture.
    """
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t5\t.\tTACA\tT\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = d / "del_in.vcf"
    plain.write_text(body)
    gz = d / "del_in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(d / "del_in"),
        ],
        check=True,
    )
    pvar_text = (d / "del_in.pvar").read_text()
    assert "\t5\t" in pvar_text, (
        "plink2 renormalized the spanning deletion's POS; the "
        "POS-outside/extent-inside property this test locks no longer "
        f"holds. Actual .pvar:\n{pvar_text}"
    )
    return ref, d / "del_in.pgen"


def test_from_pgen_variant_mode_covering_range_keeps_spanning_deletion(tmp_path):
    """Locks the `_pvar_covering_ranges` lower-bound handling for
    `regions_overlap="variant"`: it must NOT narrow the covering range's
    lower bound below the contig's original start, or a spanning deletion
    whose POS precedes the region (but whose extent reaches into it) would
    never even reach the per-record Rust filter."""
    d = tmp_path / "span"
    d.mkdir()
    ref, pgen = _write_pgen_with_spanning_deletion(d)

    out_v = tmp_path / "variant_mode"
    out_p = tmp_path / "pos_mode"
    SparseVar2.from_pgen(
        out_v, pgen, ref, regions="chr1:7-12", regions_overlap="variant", threads=1
    )
    SparseVar2.from_pgen(
        out_p, pgen, ref, regions="chr1:7-12", regions_overlap="pos", threads=1
    )

    rag_v = SparseVar2(out_v).decode("chr1", [(0, len(_REF))])
    rag_p = SparseVar2(out_p).decode("chr1", [(0, len(_REF))])
    # POS 4 (0-based) is the spanning deletion; its extent overlaps the
    # region even though its POS does not.
    assert 4 in np.asarray(rag_v["pos"].data).tolist()
    assert 4 not in np.asarray(rag_p["pos"].data).tolist()
