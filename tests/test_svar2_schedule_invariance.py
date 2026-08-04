"""Scheduling must not change output bytes.

concurrent_chroms, reader_workers, and contig dispatch order all move under
the tuned planner. Each is an opportunity to perturb chunk ordinals, per-chunk
ledgers, or long-allele bank offsets. If this test fails, nothing else in the
tuned-load-balancing change matters.
"""

from __future__ import annotations

import os

import pytest

from genoray import SparseVar2

from tests import _oracle

# (concurrent_chroms, reader_workers) -- spans the corners the planner can now
# reach: one contig at a time with many readers, and many contigs with few.
SCHEDULES = [(1, 1), (1, 12), (4, 3), (8, 2)]


@pytest.fixture(scope="module")
def multi_contig_vcf(tmp_path_factory):
    """Eight contigs with DIFFERENT record counts.

    Unequal counts are the point: with equal contigs, longest-first ordering
    is a no-op and the invariance test proves nothing about reordering.
    """
    import subprocess

    d = tmp_path_factory.mktemp("sched")
    contigs = {f"chr{i}": 4 * i for i in range(1, 9)}  # 4, 8, ... 32 records
    length = 4 * max(contigs.values()) + 10

    header = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="">',
        *[f"##contig=<ID={c},length={length}>" for c in contigs],
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1",
    ]
    rows = [
        f"{c}\t{4 * j + 1}\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|1"
        for c, n in contigs.items()
        for j in range(n)
    ]
    vcf = d / "sched.vcf"
    vcf.write_text("\n".join(header + rows) + "\n")

    vcf_gz = d / "sched.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)
    return vcf_gz


def _convert(vcf, out, cc, w):
    # The bench hooks are read in-process by the Rust orchestrator
    # (src/orchestrator.rs:80 and :448), so they must be set on os.environ --
    # a subprocess env would never reach this process's pipeline.
    os.environ.update(GENORAY_CONCURRENT_CHROMS=str(cc), GENORAY_READER_WORKERS=str(w))
    try:
        SparseVar2.from_vcf(out, vcf, no_reference=True, chunk_size=64)
    finally:
        os.environ.pop("GENORAY_CONCURRENT_CHROMS", None)
        os.environ.pop("GENORAY_READER_WORKERS", None)
    return _oracle.store_digest(out)


def test_digest_is_invariant_across_schedules(multi_contig_vcf, tmp_path):
    digests = {}
    for cc, w in SCHEDULES:
        out = tmp_path / f"cc{cc}_w{w}.svar"
        digests[(cc, w)] = _convert(multi_contig_vcf, out, cc, w)
    assert len(set(digests.values())) == 1, f"schedule changed output: {digests}"


def test_max_mem_too_small_raises_rather_than_writing_an_empty_store(
    multi_contig_vcf, tmp_path
):
    out = tmp_path / "tiny.svar"
    with pytest.raises(Exception, match="max_mem"):
        SparseVar2.from_vcf(
            out, multi_contig_vcf, no_reference=True, chunk_size=64, max_mem="1M"
        )


def test_tune_does_not_change_output(multi_contig_vcf, tmp_path):
    a = tmp_path / "untuned.svar"
    b = tmp_path / "tuned.svar"
    SparseVar2.from_vcf(a, multi_contig_vcf, no_reference=True, chunk_size=64)
    SparseVar2.from_vcf(
        b, multi_contig_vcf, no_reference=True, chunk_size=64, tune=True
    )
    assert _oracle.store_digest(a) == _oracle.store_digest(b)
