"""Scheduling must not change output bytes.

concurrent_chroms, reader_workers, and contig dispatch order all move under
the tuned planner. Each is an opportunity to perturb chunk ordinals, per-chunk
ledgers, or long-allele bank offsets. If this test fails, nothing else in the
tuned-load-balancing change matters.
"""

from __future__ import annotations

import pytest

from genoray import SparseVar2

from tests import _oracle

# (concurrent_chroms, reader_workers) -- spans the corners the planner can now
# reach: one contig at a time with many readers, and many contigs with few.
SCHEDULES = [(1, 1), (1, 12), (4, 3), (8, 2)]

# Small enough that the largest contig (chr8, 32 records) spans multiple
# chunks (4) while the smallest (chr1, 4 records) still fits in one. Chunk
# ordinals must survive reordering/concurrency, not just a degenerate
# one-chunk-per-contig case -- and it also gives the reader-rate probe
# (PROBE_CHUNKS = 2, src/tune.rs) its full two chunks on the probed (largest)
# contig instead of breaking after one.
CHUNK_SIZE = 8

# > MAX_INLINE_ALT_LEN (13, svar2-codec/src/lib.rs) so these records spill
# into the long-allele bank instead of packing inline. Without at least one
# bank write, an offset-scrambling bug in the bank would produce a
# byte-identical (empty) result under every schedule and this gate would
# never catch it.
_LONG_ALT = "ACGTACGTACGTACGTACGT"  # 20 bases


@pytest.fixture(scope="module")
def multi_contig_vcf(tmp_path_factory):
    """Eight contigs with DIFFERENT record counts, some with an indel long
    enough to spill into the long-allele bank.

    Unequal counts are the point: with equal contigs, longest-first ordering
    is a no-op and the invariance test proves nothing about reordering. The
    long-ALT record is planted at the MIDPOINT of more than one contig (never
    the first or last record) so bank offsets have a real chance to
    interleave differently across schedules, instead of only ever landing at
    a chunk boundary.
    """
    import subprocess

    d = tmp_path_factory.mktemp("sched")
    contigs = {f"chr{i}": 4 * i for i in range(1, 9)}  # 4, 8, ... 32 records
    length = 4 * max(contigs.values()) + 10
    # Every other contig gets one long-ALT record; the rest stay all-short so
    # both the inline and bank paths are exercised in the same store.
    long_alt_contigs = {"chr2", "chr4", "chr6", "chr8"}

    header = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="">',
        *[f"##contig=<ID={c},length={length}>" for c in contigs],
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1",
    ]
    rows = []
    for c, n in contigs.items():
        long_idx = n // 2 if c in long_alt_contigs else None
        for j in range(n):
            alt = _LONG_ALT if j == long_idx else "G"
            rows.append(f"{c}\t{4 * j + 1}\t.\tA\t{alt}\t.\t.\t.\tGT\t0|1\t1|1")
    vcf = d / "sched.vcf"
    vcf.write_text("\n".join(header + rows) + "\n")

    vcf_gz = d / "sched.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)
    return vcf_gz


def _convert(vcf, out, cc, w, monkeypatch):
    # The bench hooks are read in-process by the Rust orchestrator
    # (src/orchestrator.rs:80 and :448), so they must be set on os.environ --
    # a subprocess env would never reach this process's pipeline.
    # `monkeypatch.setenv` mutates `os.environ` in place (satisfying that
    # requirement) but restores whatever value was there before the test at
    # teardown, instead of unconditionally deleting the key the way a bare
    # `os.environ.pop` would.
    monkeypatch.setenv("GENORAY_CONCURRENT_CHROMS", str(cc))
    monkeypatch.setenv("GENORAY_READER_WORKERS", str(w))
    SparseVar2.from_vcf(out, vcf, no_reference=True, chunk_size=CHUNK_SIZE)
    return _oracle.store_digest(out)


def test_digest_is_invariant_across_schedules(multi_contig_vcf, tmp_path, monkeypatch):
    digests = {}
    outs = {}
    for cc, w in SCHEDULES:
        out = tmp_path / f"cc{cc}_w{w}.svar"
        digests[(cc, w)] = _convert(multi_contig_vcf, out, cc, w, monkeypatch)
        outs[(cc, w)] = out
    assert len(set(digests.values())) == 1, f"schedule changed output: {digests}"

    # Digest-invariance alone cannot tell "correctly non-empty" from
    # "incorrectly empty" -- a future edit shortening `_LONG_ALT` below
    # MAX_INLINE_ALT_LEN would silently empty the long-allele bank and every
    # digest above would still agree (on nothing). Assert the bank a planted
    # long ALT actually lands in (chr8, per `multi_contig_vcf`) is non-empty,
    # on one representative store -- the digests already proved every
    # schedule produced byte-identical output.
    any_out = next(iter(outs.values()))
    long_alleles = any_out / "chr8" / "indel" / "long_alleles.bin"
    assert long_alleles.exists() and long_alleles.stat().st_size > 0, (
        "long-allele bank is empty -- the digest-invariance gate above would "
        "pass green even if the bank write path silently broke"
    )


def test_max_mem_too_small_raises_rather_than_writing_an_empty_store(
    multi_contig_vcf, tmp_path
):
    out = tmp_path / "tiny.svar"
    with pytest.raises(Exception, match="max_mem"):
        SparseVar2.from_vcf(
            out,
            multi_contig_vcf,
            no_reference=True,
            chunk_size=CHUNK_SIZE,
            max_mem="1M",
        )
    assert not out.exists(), "a rejected max_mem budget must not create the store dir"


def test_tune_does_not_change_output(multi_contig_vcf, tmp_path):
    a = tmp_path / "untuned.svar"
    b = tmp_path / "tuned.svar"
    SparseVar2.from_vcf(a, multi_contig_vcf, no_reference=True, chunk_size=CHUNK_SIZE)
    SparseVar2.from_vcf(
        b, multi_contig_vcf, no_reference=True, chunk_size=CHUNK_SIZE, tune=True
    )
    assert _oracle.store_digest(a) == _oracle.store_digest(b)
