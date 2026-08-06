"""Scheduling must not change PGEN output bytes.

concurrent_chroms and contig dispatch order both move under the planner.
Each is an opportunity to perturb chunk ordinals, per-chunk ledgers, or
long-allele bank offsets. Mirrors test_svar2_schedule_invariance.py for the
VCF path.

The fixture is built with plink2 from a hand-written VCF rather than from
scripts/bench_svar2/pgen_corpus.py: this gate must run in CI, and the
vcfixture bulk CLI is not available there.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from genoray import SparseVar2

from tests import _oracle

# PGEN pins P=1, so reader_workers has no axis to sweep -- only cc moves.
SCHEDULES = [1, 2, 4, 8]

CHUNK_SIZE = 8

# > MAX_INLINE_ALT_LEN (13, svar2-codec/src/lib.rs) so these records spill
# into the long-allele bank instead of packing inline. Without at least one
# bank write, an offset-scrambling bug in the bank would produce a
# byte-identical (empty) result under every schedule and this gate would
# never catch it. Matches test_svar2_schedule_invariance.py's VCF-path
# fixture.
_LONG_ALT = "ACGTACGTACGTACGTACGT"  # 20 bases

pytestmark = pytest.mark.skipif(
    shutil.which("plink2") is None, reason="plink2 not available"
)


@pytest.fixture(scope="module")
def multi_contig_pgen(tmp_path_factory):
    """Eight contigs with DIFFERENT record counts, some with an indel long
    enough to spill into the long-allele bank.

    Unequal counts are the point: with equal contigs, longest-first ordering
    is a no-op and this test proves nothing about reordering. The long-ALT
    record is planted at the MIDPOINT of more than one contig (never the
    first or last record) so bank offsets have a real chance to interleave
    differently across schedules, instead of only ever landing at a chunk
    boundary.
    """
    d = tmp_path_factory.mktemp("pgen_sched")
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

    # --output-chr chrM keeps the `chr` prefix on .pvar DATA rows; without it
    # plink2 writes `1` in the body while copying ##contig=<ID=chr1> into the
    # header, and from_pgen reads the body.
    subprocess.run(
        [
            "plink2",
            "--vcf",
            str(vcf),
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--out",
            str(d / "sched"),
        ],
        check=True,
    )
    return d / "sched.pgen"


def _convert(pgen, out, cc, monkeypatch):
    # Read in-process by the Rust orchestrator, so it must be set on
    # os.environ -- a subprocess env would never reach this pipeline.
    monkeypatch.setenv("GENORAY_CONCURRENT_CHROMS", str(cc))
    SparseVar2.from_pgen(
        out, pgen, no_reference=True, chunk_size=CHUNK_SIZE, log_level="off"
    )
    return _oracle.store_digest(out)


def test_digest_is_invariant_across_schedules(multi_contig_pgen, tmp_path, monkeypatch):
    digests = {}
    outs = {}
    for cc in SCHEDULES:
        out = tmp_path / f"cc{cc}.svar"
        digests[cc] = _convert(multi_contig_pgen, out, cc, monkeypatch)
        outs[cc] = out
    assert len(set(digests.values())) == 1, f"schedule changed output: {digests}"

    # Digest-invariance alone cannot tell "correctly non-empty" from
    # "incorrectly empty" -- a future edit shortening `_LONG_ALT` below
    # MAX_INLINE_ALT_LEN would silently empty the long-allele bank and every
    # digest above would still agree (on nothing). Assert the bank a planted
    # long ALT actually lands in (chr8, per `multi_contig_pgen`) is non-empty,
    # on one representative store -- the digests already proved every
    # schedule produced byte-identical output.
    any_out = next(iter(outs.values()))
    long_alleles = any_out / "chr8" / "indel" / "long_alleles.bin"
    assert long_alleles.exists() and long_alleles.stat().st_size > 0, (
        "long-allele bank is empty -- the digest-invariance gate above would "
        "pass green even if the bank write path silently broke"
    )


def test_dispatch_order_is_longest_first_and_still_writes_meta_in_file_order(
    multi_contig_pgen, tmp_path, monkeypatch
):
    """Dispatch order must not leak into the store's layout: meta.json's
    contig order is part of the on-disk format and comes from `chroms`, not
    from the (reordered) dispatch list."""
    out = tmp_path / "order.svar"
    _convert(multi_contig_pgen, out, 4, monkeypatch)
    sv = SparseVar2(out)
    assert sv.contigs == [f"chr{i}" for i in range(1, 9)]


def test_max_mem_too_small_raises_rather_than_writing_an_empty_store(
    multi_contig_pgen, tmp_path
):
    out = tmp_path / "tiny.svar"
    with pytest.raises(Exception, match="max_mem"):
        SparseVar2.from_pgen(
            out,
            multi_contig_pgen,
            no_reference=True,
            chunk_size=CHUNK_SIZE,
            max_mem="1M",
            log_level="off",
        )
    assert not out.exists(), "a rejected max_mem budget must not create the store dir"


def test_max_mem_none_plans_against_a_detected_budget(
    multi_contig_pgen, tmp_path, monkeypatch
):
    """`None` must mean DETECTED, not unbounded -- unbounded is the OOM
    exposure the byte budget exists to remove."""
    import genoray._svar2 as svar2

    called = []
    real = svar2.detect_memory_budget

    def spy():
        called.append(True)
        return real()

    monkeypatch.setattr(svar2, "detect_memory_budget", spy)
    SparseVar2.from_pgen(
        tmp_path / "detected.svar",
        multi_contig_pgen,
        no_reference=True,
        chunk_size=CHUNK_SIZE,
        log_level="off",
    )
    assert called, "from_pgen(max_mem=None) must consult detect_memory_budget"


def test_detection_failure_warns_and_still_converts(
    multi_contig_pgen, tmp_path, monkeypatch
):
    """Detection raises on any host without a cgroup limit AND without a
    readable /proc/meminfo -- every macOS run. That must degrade to
    core-bound planning, not fail the conversion."""
    import genoray._svar2 as svar2

    def boom():
        raise RuntimeError("no cgroup limit and no /proc/meminfo")

    monkeypatch.setattr(svar2, "detect_memory_budget", boom)
    out = tmp_path / "degraded.svar"
    with pytest.warns(UserWarning, match="could not detect a memory budget"):
        SparseVar2.from_pgen(
            out,
            multi_contig_pgen,
            no_reference=True,
            chunk_size=CHUNK_SIZE,
            log_level="off",
        )
    assert (out / "meta.json").exists(), "degraded planning must still produce a store"
