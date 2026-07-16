from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

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


def _hash_store(store: Path) -> bytes:
    """SHA256 over every file's (relative path, bytes) under `store`, sorted --
    order-independent and content-exact, so two stores hash equal iff their
    on-disk layout and every byte match exactly."""
    h = hashlib.sha256()
    for p in sorted(store.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store).as_posix().encode())
            h.update(p.read_bytes())
    return h.digest()


def test_pgen_thread_count_independent(sources, tmp_path):
    """PGEN conversion output must be byte-identical regardless of `threads`.

    PGEN sub-contig sharding is currently **disabled** (`from_pgen` pins the
    shard budget `P = 1`): single-reader PGEN conversion is already fast and
    bound by the executor/writer + reference I/O, not pgenlib decode, so
    sub-contig sharding measures *slower* (concurrent readers add coordination
    and, on 0.91.x, GIL-serialize the decode). Bumping to a GIL-releasing
    pgenlib (>=0.94.x, internal `prange`) does not help -- decode isn't the
    bottleneck. See memory `pgenlib-holds-gil-sharded-reads` and the decision
    record `docs/roadmap/svar2-conversion-decision-2026-07-15.md`. The Rust
    sharding machinery is retained (correct + tested via the `pgen_shard` unit
    tests) for re-enablement only if the bottleneck ever shifts onto decode.

    So with sharding off, `threads` only scales HTSlib-style decode threads for
    the single PGEN reader; it must never change a single output byte. This
    test locks that invariant: `threads=1` (serial) vs `threads=24` (max
    decode threads) must produce byte-identical stores. `sources` has 4
    variants (chr1, positions 2/6/11/19 0-based, including a co-located
    multiallelic pair), exercising the single-reader position-ownership path.

    When PGEN sharding is re-enabled (P>1), this test doubles as the
    byte-identity gate for the sharded path.
    """
    ref, _, pgen = sources
    serial_out = tmp_path / "serial.svar2"
    parallel_out = tmp_path / "parallel.svar2"

    SparseVar2.from_pgen(serial_out, pgen, ref, threads=1)
    SparseVar2.from_pgen(parallel_out, pgen, ref, threads=24)

    assert _hash_store(serial_out) == _hash_store(parallel_out)


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
