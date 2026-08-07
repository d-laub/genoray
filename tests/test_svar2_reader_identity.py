"""Byte-identity across the presence-mask migration (issue #155).

The reader stopped retaining `Calls::Dense` and started retaining presence
bitsets. That is a representation change with no semantic content, so every
source must still produce exactly the store it produced before. Digests below
were captured at commit fab677f, the last commit before the migration, using
the SAME fixture idiom reused here (imported from the other test modules,
not reinvented): `tests/test_svar2_from_pgen.py`'s (ref, vcf, pgen) trio for
`vcf`/`pgen`, `tests/test_svar2_from_svar1.py`'s `_build_svar1` for `svar1`,
and `tests/test_svar2_from_vcf_list.py`'s `_ss` single-sample-file helper for
`vcf_list`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2

from tests import _oracle
from tests.test_svar2_from_pgen import _REF, _VCF_BODY
from tests.test_svar2_from_svar1 import _build_svar1
from tests.test_svar2_from_vcf import _write_ref as _write_biallelic_ref
from tests.test_svar2_from_vcf_list import _ss

# Filled in from a standalone capture script run at fab677f and at HEAD
# (both gave identical results). One entry per source; do not relax these to
# "digests agree with each other" -- that would pass if every source broke
# identically.
EXPECTED = {
    "pgen": "b3276500a1d417a0",
    "vcf": "b3276500a1d417a0",
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
    # suite's own biallelic fixture rather than inventing a new one.
    ref = _write_biallelic_ref(tmp_path)
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


_BUILDERS = {
    "pgen": _build_pgen_store,
    "vcf": _build_vcf_store,
    "svar1": _build_svar1_store,
    "vcf_list": _build_vcf_list_store,
}


@pytest.mark.parametrize("source", sorted(EXPECTED))
def test_store_is_byte_identical_to_the_pre_mask_reader(source, tmp_path):
    out = _BUILDERS[source](tmp_path)
    assert _oracle.store_digest(out) == EXPECTED[source]
