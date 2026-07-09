from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.typing import ArrayLike, NDArray
from pytest_cases import fixture, parametrize_with_cases

from genoray import Filter
from genoray._vcf import POS_MAX, VCF
from tests import _oracle
from tests.data.fixtures import FIXTURES

_BIALLELIC = FIXTURES["biallelic"]().truth()

tdir = Path(__file__).parent
ddir = tdir / "data"

N_SAMPLES = 2
PLOIDY = 2


@fixture
@pytest.mark.parametrize("with_gvi_index", [True, False])
def vcf(with_gvi_index: bool):
    return VCF(
        ddir / "biallelic.vcf.gz",
        phasing=True,
        dosage_field="DS",
        with_gvi_index=with_gvi_index,
    )


def read_all():
    cse = "chr1", 81261, 81263
    idx = [0, 1]  # genoray returns chr1 records 0 and 1 for this range
    genos = _oracle.genos(_BIALLELIC, idx).astype(np.int8)
    phasing = _oracle.phasing(_BIALLELIC, idx)
    dosages = _oracle.dosages(_BIALLELIC, idx)
    return cse, genos, phasing, dosages


def read_spanning_del():
    # spanning-del read shape is genoray-specific; not a clean oracle case
    cse = "chr1", 81262, 81263
    # (s p v)
    genos = np.array([[[0], [1]], [[1], [1]]], np.int8)
    # (s v)
    phasing = np.array([[1], [1]], np.bool_)
    dosages = np.array([[1.0], [2.0]], np.float32)
    return cse, genos, phasing, dosages


def read_missing_contig():
    cse = "🥸", 81261, 81263
    # (s p v)
    genos_phasing, dosages = VCF.Genos8Dosages.empty(N_SAMPLES, VCF.ploidy + 1, 0)
    genos, phasing = np.array_split(genos_phasing, 2, 1)
    phasing = phasing.squeeze(1).astype(bool)
    return cse, genos, phasing, dosages


def read_none():
    cse = "chr1", 0, 1
    # (s p v)
    genos_phasing, dosages = VCF.Genos8Dosages.empty(N_SAMPLES, VCF.ploidy + 1, 0)
    genos, phasing = np.array_split(genos_phasing, 2, 1)
    phasing = phasing.squeeze(1).astype(bool)
    return cse, genos, phasing, dosages


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_read(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    vcf.phasing = False
    # (s p v)
    g = vcf.read(*cse, VCF.Genos8)
    np.testing.assert_equal(g, genos)

    # (s p v)
    g = vcf.read(*cse, VCF.Genos16)
    np.testing.assert_equal(g, genos)

    # (s p v)
    d = vcf.read(*cse, VCF.Dosages)
    np.testing.assert_equal(d, dosages)

    # (s p v)
    g, d = vcf.read(*cse, VCF.Genos16Dosages)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(d, dosages)

    #! with phasing
    vcf.phasing = True
    gp = vcf.read(*cse)
    # (s p+1 v) -> (s p v), (s v)
    g, p = np.array_split(gp, 2, 1)
    p = p.squeeze(1).astype(bool)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)

    gp, d = vcf.read(*cse, VCF.Genos16Dosages)
    g, p = np.array_split(gp, 2, 1)
    p = p.squeeze(1).astype(bool)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)
    np.testing.assert_equal(d, dosages)


def test_read_with_out_matches_without_out(vcf: VCF):
    """Characterize existing behavior of the (previously untested) ``out=``
    path: filling a caller-provided buffer must match allocating a fresh one."""
    cse = "chr1", 81261, 81263

    expected_g = vcf.read(*cse, VCF.Genos16)
    out_g = VCF.Genos16.empty(
        vcf.n_samples, vcf.ploidy + vcf.phasing, expected_g.shape[-1]
    )
    actual_g = vcf.read(*cse, VCF.Genos16, out=out_g)
    np.testing.assert_array_equal(actual_g, expected_g)

    expected_d = vcf.read(*cse, VCF.Dosages)
    out_d = VCF.Dosages.empty(
        vcf.n_samples, vcf.ploidy + vcf.phasing, expected_d.shape[-1]
    )
    actual_d = vcf.read(*cse, VCF.Dosages, out=out_d)
    np.testing.assert_array_equal(actual_d, expected_d)

    expected_gd = vcf.read(*cse, VCF.Genos16Dosages)
    out_gd = VCF.Genos16Dosages.empty(
        vcf.n_samples, vcf.ploidy + vcf.phasing, expected_gd[0].shape[-1]
    )
    actual_gd = vcf.read(*cse, VCF.Genos16Dosages, out=out_gd)
    np.testing.assert_array_equal(actual_gd[0], expected_gd[0])
    np.testing.assert_array_equal(actual_gd[1], expected_gd[1])


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    vcf.phasing = True
    n_variants = genos.shape[2]
    max_mem = vcf._mem_per_variant(VCF.Genos16Dosages)
    gpd = vcf.chunk(*cse, max_mem, VCF.Genos16Dosages)
    for i, (gp, d) in enumerate(gpd):
        g, p = np.array_split(gp, 2, 1)
        p = p.squeeze(1).astype(bool)
        if n_variants != 0:
            np.testing.assert_equal(g, genos[..., [i]])
            np.testing.assert_equal(p, phasing[..., [i]])
            np.testing.assert_equal(d, dosages[..., [i]])
        else:
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_equal(d, dosages)


def samples_none():
    samples = None
    return samples


def samples_second():
    samples = "sample1"
    return samples


@parametrize_with_cases("samples", cases=".", prefix="samples_")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_set_samples(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8] | None,
    phasing: NDArray[np.bool_] | None,
    dosages: NDArray[np.float32] | None,
    samples: ArrayLike | None,
):
    vcf.set_samples(samples)

    if samples is None:
        samples = vcf.available_samples
        s_idx = slice(None)
    else:
        samples = np.atleast_1d(samples)
        s_idx = vcf._s2i.get(np.asarray(samples))

    assert vcf.current_samples == samples
    assert vcf.n_samples == len(samples)

    vcf.phasing = True
    gpd = vcf.read(*cse, VCF.Genos16Dosages)
    if genos is None or phasing is None or dosages is None:
        assert gpd is None
    else:
        assert gpd is not None
        gp, d = gpd
        g, p = np.array_split(gp, 2, 1)
        p = p.squeeze(1).astype(bool)
        np.testing.assert_equal(g, genos[s_idx])
        np.testing.assert_equal(p, phasing[s_idx])
        np.testing.assert_equal(d, dosages[s_idx])


def test_sample_reorder(vcf: VCF):
    # available_samples = ["sample1", "sample2"]
    # sample1: genos [[0,-1],[1,-1]], dosages [1.0, nan]
    # sample2: genos [[1, 0],[1, 1]], dosages [2.0, 1.0]
    cse = "chr1", 81261, 81263
    vcf.set_samples(["sample2", "sample1"])
    vcf.phasing = True

    gp, d = vcf.read(*cse, VCF.Genos8Dosages)
    g, p = np.array_split(gp, 2, 1)
    p = p.squeeze(1).astype(bool)

    assert vcf.current_samples == ["sample2", "sample1"]
    # row 0 must be sample2, row 1 must be sample1
    np.testing.assert_equal(
        g, np.array([[[1, 0], [1, 1]], [[0, -1], [1, -1]]], np.int8)
    )
    np.testing.assert_equal(p, np.array([[1, 0], [1, 0]], np.bool_))
    np.testing.assert_equal(d, np.array([[2.0, 1.0], [1.0, np.nan]], np.float32))


def test_vcf_mem_per_variant_doubles_when_sorted(vcf: VCF):
    ploidy = vcf.ploidy + vcf.phasing
    # a fresh reader has no active sample sorter -> estimate is NOT doubled
    unsorted = vcf._mem_per_variant(VCF.Genos16)
    assert unsorted == VCF.Genos16.nbytes_per_variant(vcf.n_samples, ploidy)
    # a single-sample subset is trivially "sorted" (a length-1 permutation is
    # always in order), so it does NOT install the ndarray sorter; a genuine
    # reorder of >=2 samples does (mirrors test_sample_reorder above).
    vcf.set_samples(list(reversed(vcf.available_samples)))
    assert vcf._mem_per_variant(VCF.Genos16) == 2 * VCF.Genos16.nbytes_per_variant(
        vcf.n_samples, ploidy
    )


def length_no_ext():
    cse = "chr1", 81264, 81265  # just 81265 in VCF
    # (s p v)
    genos = np.array([[[1], [0]], [[-1], [-1]]], np.int8)
    # (s v)
    phasing = np.array([[1], [0]], np.bool_)
    dosages = np.array([[0.9], [np.nan]], np.float32)
    last_end = 81265
    n_extension = 0
    return cse, genos, phasing, dosages, last_end, n_extension


def length_ext():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # (s p v)
    genos = np.array([[[0, 1], [1, 0]], [[1, -1], [1, -1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 1], [1, 0]], np.bool_)
    dosages = np.array([[1.0, 0.9], [2.0, np.nan]], np.float32)
    last_end = 81265
    n_extension = 1
    return cse, genos, phasing, dosages, last_end, n_extension


def length_none():
    cse = "chr1", 0, 1
    # (s p v)
    genos_phasing, dosages = VCF.Genos8Dosages.empty(N_SAMPLES, VCF.ploidy + 1, 0)
    genos, phasing = np.array_split(genos_phasing, 2, 1)
    phasing = phasing.squeeze(1).astype(bool)
    last_end = 1
    n_extension = 0
    return cse, genos, phasing, dosages, last_end, n_extension


@parametrize_with_cases(
    "cse, genos, phasing, dosages, last_end, n_extension", cases=".", prefix="length_"
)
def test_chunk_with_length(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    last_end: int,
    n_extension: int,
):
    # _chunk_ranges_with_length now requires a loaded .gvi index (it needs
    # variant indices from _var_idxs to build chunk_idxs); the `vcf` fixture is
    # parametrized over with_gvi_index=[True, False], so build+load on demand.
    if vcf._index is None:
        vcf._write_gvi_index()
        vcf._load_index()

    vcf.phasing = True

    # Independently-computed in-range variant indices (not derived from the
    # method under test) so the chunk_idxs assertions below stay non-vacuous.
    contig, start, end_ = cse
    in_range_idxs, _ = vcf._var_idxs(contig, start, end_)
    in_range_idxs = in_range_idxs.astype(np.uint32)

    max_mem = vcf._mem_per_variant(VCF.Genos16Dosages)
    gpd = vcf._chunk_ranges_with_length(*cse, max_mem, VCF.Genos16Dosages)
    all_idxs: list[NDArray[np.uint32]] = []
    for range_ in gpd:
        for chunk, end, chunk_idxs in range_:
            gp, d = chunk
            g, p = np.array_split(gp, 2, 1)
            p = p.squeeze(1).astype(bool)
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_equal(d, dosages)
            assert end == last_end
            # chunk_idxs is now a uint32 variant-index array, one index per
            # variant in this chunk (matching PGEN's contract).
            assert chunk_idxs.dtype == np.uint32
            assert chunk_idxs.shape[0] == gp.shape[-1]
            all_idxs.append(chunk_idxs)

    full = np.concatenate(all_idxs) if all_idxs else np.empty(0, dtype=np.uint32)
    # The in-range prefix must match the independently-computed indices...
    assert np.array_equal(full[: in_range_idxs.size], in_range_idxs)
    # ...and whatever is left over is exactly the extension variants, whose
    # count must match this case's expected `n_extension` (keeps this
    # non-vacuous: a wrong extension count/length fails here).
    assert full.size - in_range_idxs.size == n_extension


def test_chunk_ranges_with_length_yields_chunk_idxs():
    """The 3rd tuple element is now the variant-index array, and the extension
    indices are contiguous after the last in-range index (what gvl assumed)."""
    vcf = VCF(ddir / "biallelic.vcf.gz")
    if vcf._index is None:
        vcf._write_gvi_index()
        vcf._load_index()
    contig = vcf.contigs[0]
    starts = np.array([0], dtype=np.int32)
    ends = np.array([10_000], dtype=np.int32)

    v_idx, offsets = vcf._var_idxs(contig, starts, ends)
    reg_unext = v_idx[offsets[0] : offsets[1]].astype(np.uint32)

    all_idxs = []
    for range_ in vcf._chunk_ranges_with_length(contig, starts, ends, "4g", VCF.Genos8):
        for genos, _end, chunk_idxs in range_:
            assert chunk_idxs.dtype == np.uint32
            assert chunk_idxs.shape[0] == genos.shape[-1]
            all_idxs.append(chunk_idxs)
    full = np.concatenate(all_idxs) if all_idxs else np.empty(0, np.uint32)

    # in-range prefix matches _var_idxs; any extension is contiguous after it
    assert np.array_equal(full[: reg_unext.size], reg_unext)
    if full.size > reg_unext.size:
        ext = full[reg_unext.size :]
        assert np.array_equal(
            ext, np.arange(reg_unext[-1] + 1, reg_unext[-1] + 1 + ext.size)
        )


def test_nbytes_zero_before_index_loaded():
    # _load_index is not called when with_gvi_index=False and no auto-load occurs
    vcf = VCF(ddir / "biallelic.vcf.gz", with_gvi_index=False)
    assert vcf._index is None
    assert vcf.nbytes == 0


def test_nbytes_positive_after_index_loaded():
    vcf = VCF(ddir / "biallelic.vcf.gz")
    if not vcf._valid_index():
        vcf._write_gvi_index()
    vcf._load_index()
    assert vcf._index is not None
    assert vcf.nbytes > 0
    # sanity: at least one byte per row across CHROM/POS/REF/ALT
    assert vcf.nbytes >= vcf._index.height


def test_chunk_with_length_phased_indel_in_extension():
    # Regression: with phasing=True, an indel in the EXTENSION region (past the
    # query) previously crashed _ext_genos_*_with_length with a hap_lens
    # broadcast error (shape (s, ploidy) vs (s, ploidy + phasing)). Region B of
    # the indels fixture (query 1999-2006) extends across the -30 deletion that
    # begins region C, exercising the indel-in-extension path.
    vcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")
    vcf.phasing = True
    gen = vcf._chunk_ranges_with_length("chr1", 1999, 2006, "1g", VCF.Genos16Dosages)
    chunks = []
    for range_ in gen:
        for chunk, end, n_ext in range_:
            gp, d = chunk
            chunks.append(gp)
        break
    genos_phasing = np.concatenate(chunks, axis=-1)
    # phased Genos16 has a phasing-indicator row -> axis 1 == ploidy + 1
    assert genos_phasing.shape[1] == vcf.ploidy + 1
    # and it produced at least the query's variants without crashing
    assert genos_phasing.shape[-1] > 0


def test_filter_object_roundtrip():
    f = Filter(record=lambda v: True, expr=pl.col("POS") > 0)
    vcf = VCF(ddir / "biallelic.vcf.gz", filter=f)
    assert vcf.filter is f
    vcf.filter = None
    assert vcf.filter is None


def test_filter_object_is_frozen():
    f = Filter(record=lambda v: True, expr=pl.lit(True))
    with pytest.raises(Exception):
        f.record = lambda v: False  # frozen dataclass


def test_old_pl_filter_kwarg_removed():
    with pytest.raises(TypeError):
        VCF(ddir / "biallelic.vcf.gz", pl_filter=pl.lit(True))  # kwarg no longer exists


def test_filter_setter_enforces_pair_invariant():
    from genoray import exprs

    vcf = VCF(ddir / "biallelic.vcf.gz")

    def record_fn(v):
        return not any(a.startswith("<") for a in v.ALT)

    pl_expr = ~exprs.is_symbolic
    f = Filter(record=record_fn, expr=pl_expr)

    # Setting a Filter updates both halves together and invalidates the index.
    vcf._index = "sentinel"
    vcf.filter = f
    assert vcf.filter is f
    assert vcf._index is None

    # The getter mirrors the setter, so assigning the getter back round-trips.
    vcf.filter = vcf.filter
    assert vcf.filter is f

    # Assigning None clears the filter; the getter returns None.
    vcf._index = "sentinel"
    vcf.filter = None
    assert vcf.filter is None
    assert vcf._index is None

    # A bare (non-Filter) value is rejected.
    with pytest.raises(TypeError):
        vcf.filter = record_fn


def test_filtered_var_idxs_consistent_with_index():
    """Regression guard for #69 on the VCF backend.

    VCF._var_idxs is private and never gathers _index, but it shares
    var_indices() with PGEN, so it must also return positional indices. Filtered
    reads must return the same genotypes as the matching unfiltered variants.
    """
    from genoray import exprs

    # is_snp as a Filter bundling a cyvcf2 record predicate + pl.Expr (VCF requires both).
    def record_is_snp(rec):
        return len(rec.REF) == 1 and all(len(a) == 1 for a in rec.ALT)

    v_filt = VCF(
        ddir / "biallelic.vcf.gz",
        filter=Filter(record=record_is_snp, expr=exprs.is_snp),
    )
    v_filt._write_gvi_index()
    v_filt._load_index()
    v_full = VCF(ddir / "biallelic.vcf.gz")

    assert v_filt._index.height == 4

    # _var_idxs is positional and in-bounds (pre-#69 it returned physical [4,5]).
    vi, _ = v_filt._var_idxs("chr2", 0, POS_MAX)
    assert int(vi.max()) < v_filt._index.height
    assert np.array_equal(vi, np.array([2, 3], dtype=vi.dtype))
    assert v_filt._index[vi]["POS"].to_list() == [81262, 81265]

    # Filtered chr1 read == the SNP variants (physical [1, 2]) of the full read.
    filt = v_filt.read("chr1", 0, POS_MAX, mode=VCF.Genos16)
    full = v_full.read("chr1", 0, POS_MAX, mode=VCF.Genos16)
    assert filt.shape[-1] == 2
    assert np.array_equal(filt, full[..., [1, 2]])


def test_filter_applied_to_genos_dosages_without_index():
    # A record filter + NO .gvi index forces the out-is-None path of
    # _fill_genos_and_dosages, where the filter was previously dropped.
    from genoray import exprs

    vcf = VCF(
        ddir / "biallelic.vcf.gz",
        phasing=False,
        dosage_field="DS",
        with_gvi_index=False,
        filter=Filter(
            record=lambda rec: len(rec.REF) == 1 and all(len(a) == 1 for a in rec.ALT),
            expr=exprs.is_snp,
        ),
    )
    cse = ("chr1", 81261, 81266)  # covers the GAT>A indel and two SNPs
    genos_only = vcf.read(*cse, VCF.Genos8)  # filters correctly today
    genos, _dosages = vcf.read(*cse, VCF.Genos8Dosages)
    # The genotypes from the combined mode must match the genos-only mode:
    # both must have the SNP filter applied (indel dropped).
    np.testing.assert_array_equal(genos, genos_only)
