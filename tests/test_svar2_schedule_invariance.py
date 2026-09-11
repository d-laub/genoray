"""Scheduling must not change output bytes.

concurrent_chroms, reader_workers, and contig dispatch order all move under
the planner. Each is an opportunity to perturb chunk ordinals, per-chunk
ledgers, or long-allele bank offsets. If this test fails, nothing else in the
load-balancing change matters.
"""

from __future__ import annotations

import pytest

from genoray import SparseVar2, Tuning

from tests import _oracle
from tests._log_parsing import log_field_int


def _pipeline_schedule(captured_text: str) -> tuple[int, int]:
    """The `(concurrent_chroms, reader_workers)` pair the planner actually
    chose, read off the "pipeline config" tracing event."""
    return (
        log_field_int(captured_text, "concurrent_chroms"),
        log_field_int(captured_text, "reader_workers"),
    )


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


def _convert(vcf, out, cc, w):
    SparseVar2.from_vcf(
        out,
        vcf,
        no_reference=True,
        chunk_size=CHUNK_SIZE,
        tuning=Tuning(concurrent_chroms=cc, reader_workers=w),
    )
    return _oracle.store_digest(out)


def test_digest_is_invariant_across_schedules(multi_contig_vcf, tmp_path, capfd):
    digests = {}
    outs = {}
    schedules_seen = set()
    for cc, w in SCHEDULES:
        out = tmp_path / f"cc{cc}_w{w}.svar"
        digests[(cc, w)] = _convert(multi_contig_vcf, out, cc, w)
        captured = capfd.readouterr()
        schedules_seen.add(_pipeline_schedule(captured.out + captured.err))
        outs[(cc, w)] = out
    assert len(schedules_seen) >= 2, (
        "concurrent_chroms/reader_workers never varied across SCHEDULES rows "
        f"-- this test proves nothing about schedule invariance: {schedules_seen}"
    )
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
    multi_contig_vcf, tmp_path, capfd
):
    """Unit granularity must not move a single output byte.

    Since #169 the work-unit count comes from the contig's RECORD count
    (`shard::plan_unit_count`), and the reorder backlog is bounded so
    non-head readers park mid-stream. Both change WHEN a chunk reaches the
    collector; neither may change what is written.

    `Tuning(overshard=)` only feeds `plan_unit_count` on the no-exact-counts
    (header-length fallback) tier -- this fixture is indexed, so
    `exact_counts` is always true and that knob moves nothing here.
    Drive granularity via `chunk_size` instead, at `reader_workers=1` so the
    `.max(workers)` floor in `plan_unit_count` can't paper over the effect:
    on the largest contig (chr8, 32 records),
    `per_unit = UNITS_TARGET_CHUNKS(4) * chunk_size` gives
    `planned_units = ceil(32 / per_unit)`, i.e. 4 units at `chunk_size=2` and
    1 unit at `chunk_size=8` -- confirmed directly against
    `shard::plan_unit_count` rather than assumed. Parse the actual
    `planned_units` off the "pipeline config" log line (via `capfd`,
    via the shared `log_field_int` in tests/_log_parsing.py) so
    the test fails loudly if a future change makes the granularity axis
    inert again, instead of passing vacuously.
    """
    digests = {}
    planned_units_seen = {}
    for chunk_size in (2, 4, 8):
        out = tmp_path / f"cs{chunk_size}.svar"
        SparseVar2.from_vcf(
            out,
            multi_contig_vcf,
            no_reference=True,
            chunk_size=chunk_size,
            tuning=Tuning(concurrent_chroms=1, reader_workers=1),
            log_level="info",
        )
        captured = capfd.readouterr()
        planned_units_seen[chunk_size] = log_field_int(
            captured.out + captured.err, "planned_units"
        )
        digests[chunk_size] = _oracle.store_digest(out)

    assert len(set(planned_units_seen.values())) >= 2, (
        "chunk_size never moved planned_units -- this test proves nothing "
        f"about granularity invariance: {planned_units_seen}"
    )
    assert len(set(digests.values())) == 1, (
        f"unit granularity changed output: {digests} (planned_units={planned_units_seen})"
    )
