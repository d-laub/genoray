"""Scheduling must not change output bytes.

concurrent_chroms, reader_workers, and contig dispatch order all move under
the planner. Each is an opportunity to perturb chunk ordinals, per-chunk
ledgers, or long-allele bank offsets. If this test fails, nothing else in the
load-balancing change matters.
"""

from __future__ import annotations

import pytest

from genoray import SparseVar2

from tests import _oracle

# (concurrent_chroms, reader_workers) -- spans the corners the planner can
# reach: one contig at a time with many readers, and many contigs with few.
# The wide-w rows matter more since #169: `reader_workers` is now derived from
# cores rather than pinned at 3, so a schedule the planner never used to
# produce is now the default on a large host.
SCHEDULES = [(1, 1), (1, 12), (1, 32), (4, 3), (8, 2)]

# Small enough that the largest contig (chr8, 32 records) spans multiple
# chunks (4) while the smallest (chr1, 4 records) still fits in one. Chunk
# ordinals must survive reordering/concurrency, not just a degenerate
# one-chunk-per-contig case.
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


def test_digest_is_invariant_across_frontier_granularities(
    multi_contig_vcf, tmp_path, monkeypatch
):
    """Unit granularity must not move a single output byte.

    Since #169 the work-unit count comes from the contig's RECORD count
    (`shard::plan_unit_count`), not from `workers * OVERSHARD_FACTOR`, and the
    reorder backlog is bounded so non-head readers park mid-stream. Both
    change WHEN a chunk reaches the collector; neither may change what is
    written. The `GENORAY_OVERSHARD` values below drive the no-exact-counts
    fallback path, which this small fixture takes.
    """
    digests = {}
    for overshard in (1, 4, 40):
        monkeypatch.setenv("GENORAY_OVERSHARD", str(overshard))
        out = tmp_path / f"ov{overshard}.svar"
        digests[overshard] = _convert(multi_contig_vcf, out, 1, 8, monkeypatch)
    assert len(set(digests.values())) == 1, (
        f"unit granularity changed output: {digests}"
    )


def test_explicit_reader_workers_matches_the_derived_default(
    multi_contig_vcf, tmp_path, monkeypatch
):
    """The public knob and the planner's own choice must agree byte-for-byte.

    `_convert` sets GENORAY_READER_WORKERS; this asserts the public
    `reader_workers=` argument lands on the same code path and produces the
    same store.
    """
    env_out = tmp_path / "via_env.svar"
    env_digest = _convert(multi_contig_vcf, env_out, 1, 6, monkeypatch)

    monkeypatch.delenv("GENORAY_READER_WORKERS", raising=False)
    monkeypatch.delenv("GENORAY_CONCURRENT_CHROMS", raising=False)
    arg_out = tmp_path / "via_arg.svar"
    SparseVar2.from_vcf(
        arg_out,
        multi_contig_vcf,
        no_reference=True,
        chunk_size=CHUNK_SIZE,
        reader_workers=6,
    )
    assert _oracle.store_digest(arg_out) == env_digest
