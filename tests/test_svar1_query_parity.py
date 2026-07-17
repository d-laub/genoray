"""Differential parity: the ungated Rust `svar1_query` core vs genoray's Python/numba
SVAR1 query path.

`search.rs` has always claimed to "mirror the SVAR 1.0 `var_ranges` shape", but until
now nothing tested that claim. These tests pin it.

Two DELIBERATE convention gaps (assert the correspondence; do not "fix" either side):

1. No-overlap sentinel. Python `var_ranges` returns INT32_MAX in both columns; Rust
   returns an in-bounds zero-length range. Rust's is required downstream -- an
   out-of-range offset overflows int64 in seqpro's `Ragged.to_packed` even for an
   empty row (`_svar/_kernels.py:239-243`).
2. `max_v_len`. Python's `(v_ends - v_starts).max()` is exactly 1 larger than
   `overlap_range`'s `>=` contract wants. It is an OVER-estimate, which only widens
   the candidate window -- provably overshoot-safe. Under-estimating would be a bug.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray import SparseVar
from genoray._core import PySvar1Reader
from genoray._types import V_IDX_TYPE
from genoray._var_ranges import _var_end_expr

# chr1: SNP@3, INS@7 (C>CAT), SNP@10, DEL@12 (GTA>G, ILEN -2)  [1-based POS]
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t10\t.\tG\tC\t.\t.\t.\tGT\t1|1\t1|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""

SENTINEL = np.iinfo(V_IDX_TYPE).max


@pytest.fixture(scope="module")
def svar1_store(tmp_path_factory) -> Path:
    from genoray import VCF

    d = tmp_path_factory.mktemp("svar1_query_parity")
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store.svar"
    SparseVar.from_vcf(
        out, VCF(bcf), max_mem="1g", samples=["S0", "S1"], overwrite=True
    )
    return out


def _contig_arrays(sv: SparseVar, contig: str):
    """Contig-local v_starts/v_ends/max_v_len/contig_start, mirroring what
    `_var_ranges.var_ranges` derives internally."""
    idx = sv.index.sort("index").filter(pl.col("CHROM") == contig)
    v_starts = (idx["POS"].to_numpy() - 1).astype(np.uint32)  # 0-based
    v_ends = idx.select(_var_end_expr()).to_series().to_numpy().astype(np.uint32)
    max_v_len = int((v_ends.astype(np.int64) - v_starts.astype(np.int64)).max())
    contig_start = int(idx["index"][0])
    return v_starts, v_ends, max_v_len, contig_start


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 40),  # whole contig
        (0, 5),  # leading
        (3, 20),  # sub-contig, drops the first SNP
        (11, 12),  # inside the DEL's span but after its POS -- the sub-scan case
        (35, 40),  # trailing, no variants
        (20, 21),  # interior, no variants
    ],
)
def test_var_ranges_matches_python(svar1_store: Path, start: int, end: int):
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")

    py = sv.var_ranges("chr1", [start], [end])  # (1, 2), GLOBAL ids
    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, [(start, end)])

    if (py[0] == SENTINEL).all():
        # Convention gap 1: Python signals "no overlap" with a sentinel; Rust with
        # an in-bounds zero-length range.
        assert rs[0, 0] == rs[0, 1], (
            f"rust must be zero-length where python is sentinel: {rs[0]}"
        )
    else:
        np.testing.assert_array_equal(rs[0], py[0].astype(np.int64))


def test_var_ranges_batches_match_python_elementwise(svar1_store: Path):
    """The batched call must agree with Python for every region at once -- a
    per-region loop could hide an ordering bug in the batch path."""
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    starts = [0, 3, 11, 20, 35]
    ends = [40, 20, 12, 21, 40]

    py = sv.var_ranges("chr1", starts, ends)
    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs = r.var_ranges(
        v_starts, v_ends, max_v_len, contig_start, list(zip(starts, ends))
    )

    assert rs.shape == py.shape
    for i in range(len(starts)):
        if (py[i] == SENTINEL).all():
            assert rs[i, 0] == rs[i, 1]
        else:
            np.testing.assert_array_equal(rs[i], py[i].astype(np.int64))


def test_find_ranges_matches_python_find_starts_ends(svar1_store: Path):
    """Stage B vs `_find_starts_ends`. Both are cartesian (r, s, p), so the shapes
    line up directly: Python's (2, r, s, p) -> starts = out[0].ravel()."""
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    starts = [0, 3, 35]
    ends = [40, 20, 40]

    # (2, r, s, p)
    py = sv._find_starts_ends("chr1", starts, ends, samples=None)

    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs_ranges = r.var_ranges(
        v_starts, v_ends, max_v_len, contig_start, list(zip(starts, ends))
    )
    d = r.find_ranges([(int(a), int(b)) for a, b in rs_ranges], None)

    np.testing.assert_array_equal(d["starts"], py[0].ravel())
    np.testing.assert_array_equal(d["stops"], py[1].ravel())
    assert d["n_ranges"] == len(starts)
    assert d["n_samples"] == len(sv.available_samples)
    assert d["ploidy"] == sv.ploidy


def test_find_ranges_sample_subset_matches_python(svar1_store: Path):
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    py = sv._find_starts_ends("chr1", [0], [40], samples=["S1"])

    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs_ranges = r.var_ranges(v_starts, v_ends, max_v_len, contig_start, [(0, 40)])
    d = r.find_ranges([(int(rs_ranges[0, 0]), int(rs_ranges[0, 1]))], [1])

    np.testing.assert_array_equal(d["starts"], py[0].ravel())
    np.testing.assert_array_equal(d["stops"], py[1].ravel())


def test_offsets_are_never_out_of_bounds(svar1_store: Path):
    """Guards convention gap 1 at the point it actually matters: every emitted
    offset must index into `variant_idxs`, including for empty rows. An
    out-of-range value overflows int64 in seqpro's Ragged.to_packed."""
    sv = SparseVar(svar1_store)
    v_starts, v_ends, max_v_len, contig_start = _contig_arrays(sv, "chr1")
    n_entries = len(sv.genos.data)

    r = PySvar1Reader(str(svar1_store), len(sv.available_samples), sv.ploidy)
    rs_ranges = r.var_ranges(
        v_starts, v_ends, max_v_len, contig_start, [(35, 40), (0, 40)]
    )
    d = r.find_ranges([(int(a), int(b)) for a, b in rs_ranges], None)

    assert d["starts"].min() >= 0
    assert d["stops"].max() <= n_entries
    assert (d["stops"] >= d["starts"]).all()
