"""Byte-identity across the presence-mask migration (issue #155).

The reader stopped retaining `Calls::Dense` and started retaining presence
bitsets. That is a representation change with no semantic content, so every
source must still produce exactly the store it produced before. Digests below
were captured at commit fab677f, the last commit before the migration, using
the SAME fixture idiom reused here (imported from the other test modules,
not reinvented): `tests/test_svar2_from_pgen.py`'s (ref, vcf, pgen) trio for
`vcf`/`pgen`, `tests/test_svar2_from_svar1.py`'s `_build_svar1` for `svar1`,
`tests/test_svar2_from_vcf_list.py`'s `_ss` single-sample-file helper for
`vcf_list`, and `vcf_wide` (50 samples, defined in this module) for the
multi-word/carry path.

What this gate does and does not exercise: `pgen`/`vcf`/`svar1`/`vcf_list`
are all 2-sample (4-column) fixtures, so `words_per_mask == 1` and every
presence-mask row lives inside a single u64 -- they exercise composition
(which `from_*` entry point, which `Calls` variant) and the single-word
packing path only. `vcf_wide` uses 50 samples (100 columns,
`words_per_mask == 2`) specifically so `or_mask_into`'s carry branch in
`src/chunk_assembler.rs` (`words[hi] |= m >> (64 - s)`) actually runs; see
the comment above `_WIDE_N_SAMPLES` for the bit-span arithmetic proving it.
None of these fixtures are large enough to exercise `pack_presence_par`
(needs >= `PARALLEL_MIN_CELLS` = 524,288 cells -- deliberately out of scope
for a unit test) or the dividing branch of either byte budget (both clamp to
their max at these sizes). The bit-level packing logic itself -- including
the carry path and the parallel/sequential equivalence -- is covered by the
Rust proptests `or_mask_into_matches_the_allele_scan` and
`test_par_packing_matches_seq` in `src/chunk_assembler.rs`; this module only
gates the higher-level claim that the full conversion pipeline still
produces the same store bytes.

Re-pin policy: a digest mismatch here means the reader's output changed and
must be FIXED, not re-baselined. The only legitimate reason to update
`EXPECTED` is a deliberate store-format change. Re-capture with the
`git archive fab677f` recipe (never `git checkout fab677f` in a worktree
that has to come back): extract `git archive fab677f` into a scratch dir,
`pixi run maturin build --release` there, drop the resulting `_core*.so`
into `<dir>/python/genoray/`, then run the same fixture builders against
that tree and against HEAD and compare.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2

from tests import _oracle
from tests.test_svar2_from_pgen import _REF, _VCF_BODY
from tests.test_svar2_from_svar1 import _build_svar1
from tests.test_svar2_from_vcf_list import _ss

# Filled in from a standalone capture script run at fab677f and at HEAD
# (both gave identical results). One entry per source; do not relax these to
# "digests agree with each other" -- that would pass if every source broke
# identically.
EXPECTED = {
    "pgen": "b3276500a1d417a0",
    "vcf": "b3276500a1d417a0",
    "vcf_wide": "3755e596876a3c72",
    "svar1": "e416cb93b68d30b3",
    "vcf_list": "d4a17fab269e02ef",
}


def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _write_vcf(d: Path) -> Path:
    plain = d / "in.vcf"
    plain.write_text(_VCF_BODY)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def _write_pgen(d: Path, vcf_gz: Path) -> Path:
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(vcf_gz),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    return d / "in.pgen"


def _build_vcf_store(tmp_path: Path) -> Path:
    ref = _write_ref(tmp_path)
    vcf_gz = _write_vcf(tmp_path)
    out = tmp_path / "out.svar2"
    SparseVar2.from_vcf(out, vcf_gz, ref, threads=1)
    return out


def _build_pgen_store(tmp_path: Path) -> Path:
    ref = _write_ref(tmp_path)
    vcf_gz = _write_vcf(tmp_path)
    pgen = _write_pgen(tmp_path, vcf_gz)
    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(out, pgen, ref, threads=1)
    return out


def _build_svar1_store(tmp_path: Path) -> Path:
    # from_svar1 supports only biallelic SVAR1 stores; reuse the test
    # suite's own biallelic fixture rather than inventing a new one. The
    # underlying 40bp reference (this module's own _write_ref) is the same
    # one _build_svar1's VCF is built against.
    ref = _write_ref(tmp_path)
    src = _build_svar1(tmp_path)
    out = tmp_path / "out.svar2"
    SparseVar2.from_svar1(out, src, ref, threads=1)
    return out


def _build_vcf_list_store(tmp_path: Path) -> Path:
    # The control: from_vcf_list produces Calls::Sparse, which this
    # migration deliberately did not touch.
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "out.svar2"
    SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    return out


# 50 samples => 100 columns (ploidy 2). columns=100 => words_per_mask =
# ceil(100/64) = 2, so a presence-mask row no longer fits in one u64 the way
# the 2-sample (4-column) fixtures above always do. Concretely: variant
# index vi=1 (the second record below, the insertion at POS 7) has bit span
# [100*1, 100*1+100) = [100, 200). 100 is not a multiple of 64 (100 % 64 ==
# 36), so this row starts mid-word; its words run from word 100//64==1
# through word 199//64==3, crossing the 64-bit boundaries at 128 and 192.
# That forces `or_mask_into` (src/chunk_assembler.rs) to take its carry
# branch (`words[hi] |= m >> (64 - s)`), which every other source in this
# file -- all one word per row -- never reaches.
_WIDE_N_SAMPLES = 50


def _wide_gt(i: int, n_alt: int) -> str:
    """Deterministic GT string from the sample index -- no RNG, ever: the
    digest below is pinned, so any nondeterminism here makes the test flaky.

    Cycles every 6 samples through hom-ref, phased het, phased hom-alt,
    unphased het, a missing call, and (for the multiallelic record,
    `n_alt=2`) a second-alt combination -- so every wide record's 50 columns
    are a fixed, reproducible mix of phasing/zygosity/missingness rather
    than a uniform genotype that wouldn't stress the mask.
    """
    c = i % 6
    if c == 0:
        return "0|0"
    if c == 1:
        return "0|1"
    if c == 2:
        return f"{n_alt}|{n_alt}"
    if c == 3:
        return "1/0"
    if c == 4:
        return ".|."
    return "1|2" if n_alt >= 2 else "0/1"


def _wide_row(n_alt: int) -> str:
    return "\t".join(_wide_gt(i, n_alt) for i in range(_WIDE_N_SAMPLES))


def _write_wide_vcf(d: Path) -> Path:
    """50-sample VCF against the shared 40bp reference (_REF): a SNP, an
    insertion, a multiallelic indel, another SNP, a deletion, and a third
    SNP -- all deterministic (see `_wide_gt`), non-overlapping, and checked
    against 1-based `_REF` positions:
      POS 3  'A'          SNP        A -> G
      POS 7  'C'          insertion  C -> CAT
      POS 12 'GTA' (12-14) multiallelic indel  GTA -> G, GT
      POS 20 'T'          SNP        T -> A
      POS 24 'CTAACC' (24-29) deletion  CTAACC -> C
      POS 32 'T'          SNP        T -> C
    """
    samples = "\t".join(f"S{i}" for i in range(_WIDE_N_SAMPLES))
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n"
        f"chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t{_wide_row(1)}\n"
        f"chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t{_wide_row(1)}\n"
        f"chr1\t12\t.\tGTA\tG,GT\t.\t.\t.\tGT\t{_wide_row(2)}\n"
        f"chr1\t20\t.\tT\tA\t.\t.\t.\tGT\t{_wide_row(1)}\n"
        f"chr1\t24\t.\tCTAACC\tC\t.\t.\t.\tGT\t{_wide_row(1)}\n"
        f"chr1\t32\t.\tT\tC\t.\t.\t.\tGT\t{_wide_row(1)}\n"
    )
    plain = d / "wide.vcf"
    plain.write_text(body)
    gz = d / "wide.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def _build_vcf_wide_store(tmp_path: Path) -> Path:
    ref = _write_ref(tmp_path)
    vcf_gz = _write_wide_vcf(tmp_path)
    out = tmp_path / "out.svar2"
    SparseVar2.from_vcf(out, vcf_gz, ref, threads=1)
    return out


_BUILDERS = {
    "pgen": _build_pgen_store,
    "vcf": _build_vcf_store,
    "vcf_wide": _build_vcf_wide_store,
    "svar1": _build_svar1_store,
    "vcf_list": _build_vcf_list_store,
}


@pytest.mark.parametrize("source", sorted(EXPECTED))
def test_store_is_byte_identical_to_the_pre_mask_reader(source: str, tmp_path: Path):
    out = _BUILDERS[source](tmp_path)
    assert _oracle.store_digest(out) == EXPECTED[source]
