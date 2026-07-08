from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from pytest_cases import parametrize_with_cases

from genoray._pgen import PGEN, Genos, POS_MAX, V_IDX_TYPE, _gen_with_length
from genoray._types import POS_TYPE
from tests import _oracle
from tests.data.fixtures import FIXTURES

_BIALLELIC = FIXTURES["biallelic"]().truth()

tdir = Path(__file__).parent
ddir = tdir / "data"

N_SAMPLES = 2


def pgen_no_vzs():
    return PGEN(ddir / "biallelic.pgen")


def pgen_vzs():
    return PGEN(ddir / "biallelic.zst.pgen")


def read_all():
    cse = "chr1", 81261, 81262  # just 81262 in VCF
    # biallelic idx=[0,1]: chr1:81262 GAT>A (0|1,1|1) and chr1:81262 G>A (./.→-1,0/1)
    # idx1 ./. → PGEN keeps as missing (-1); oracle agrees
    genos = _oracle.genos(_BIALLELIC, [0, 1])
    phasing = _oracle.phasing(_BIALLELIC, [0, 1])
    dosages = _oracle.dosages(_BIALLELIC, [0, 1])
    return cse, genos, phasing, dosages


def read_spanning_del():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # biallelic idx=[0]: chr1:81262 GAT>A (0|1,1|1), DS=(1.0,2.0)
    genos = _oracle.genos(_BIALLELIC, [0])
    phasing = _oracle.phasing(_BIALLELIC, [0])
    dosages = _oracle.dosages(_BIALLELIC, [0])
    return cse, genos, phasing, dosages


def read_missing_contig():
    cse = "🥸", 81261, 81263
    # (s p v)
    genos, phasing, dosages = PGEN.GenosPhasingDosages.empty(N_SAMPLES, PGEN.ploidy, 0)
    return cse, genos, phasing, dosages


def read_none():
    cse = "chr1", 0, 1
    # (s p v)
    genos, phasing, dosages = PGEN.GenosPhasingDosages.empty(N_SAMPLES, PGEN.ploidy, 0)
    return cse, genos, phasing, dosages


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_read(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    # (s p v)
    g = pgen.read(*cse)
    np.testing.assert_equal(g, genos)

    d = pgen.read(*cse, PGEN.Dosages)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)

    g, p = pgen.read(*cse, PGEN.GenosPhasing)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)

    g, d = pgen.read(*cse, PGEN.GenosDosages)
    np.testing.assert_equal(g, genos)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)

    g, p, d = pgen.read(*cse, PGEN.GenosPhasingDosages)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    n_variants = genos.shape[2]
    mode = PGEN.GenosPhasingDosages
    gpd = pgen.chunk(*cse, pgen._mem_per_variant(mode), mode)
    for i, (g, p, d) in enumerate(gpd):
        if n_variants != 0:
            np.testing.assert_equal(g, genos[..., [i]])
            np.testing.assert_equal(p, phasing[..., [i]])
            np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)
        else:
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_allclose(d, dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_read_ranges(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    c, s, e = cse
    s = [s, s]
    e = [e, e]

    (g, p, d), o = pgen.read_ranges(c, s, e, PGEN.GenosPhasingDosages)
    np.testing.assert_equal(g[..., o[0] : o[1]], genos)
    np.testing.assert_equal(g[..., o[1] : o[2]], genos)
    np.testing.assert_equal(p[..., o[0] : o[1]], phasing)
    np.testing.assert_equal(p[..., o[1] : o[2]], phasing)
    np.testing.assert_allclose(d[..., o[0] : o[1]], dosages, rtol=1e-5)
    np.testing.assert_allclose(d[..., o[1] : o[2]], dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk_ranges(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    c, s, e = cse
    s = [s, s]
    e = [e, e]

    n_variants = genos.shape[2]
    mode = PGEN.GenosPhasingDosages
    gpdo = pgen.chunk_ranges(c, s, e, max_mem=pgen._mem_per_variant(mode), mode=mode)
    for range_ in gpdo:
        for i, (g, p, d) in enumerate(range_):
            if n_variants != 0:
                np.testing.assert_equal(g, genos[..., [i]])
                np.testing.assert_equal(p, phasing[..., [i]])
                np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)
            else:
                np.testing.assert_equal(g, genos)
                np.testing.assert_equal(p, phasing)
                np.testing.assert_allclose(d, dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
def test_sample_reorder(pgen: PGEN):
    # available_samples = ["sample1", "sample2"]
    # sample1: genos [[0,-1],[1,-1]], dosages [1.0, nan]
    # sample2: genos [[1, 0],[1, 1]], dosages [2.0, 1.0]
    cse = "chr1", 81261, 81262
    pgen.set_samples(["sample2", "sample1"])

    g, p, d = pgen.read(*cse, PGEN.GenosPhasingDosages)

    assert list(pgen.current_samples) == ["sample2", "sample1"]
    # row 0 must be sample2, row 1 must be sample1
    # biallelic idx=[0,1], samples reversed: oracle[1,0] = [sample2, sample1]
    _g = _oracle.genos(_BIALLELIC, [0, 1])[[1, 0]]
    _p = _oracle.phasing(_BIALLELIC, [0, 1])[[1, 0]]
    _d = _oracle.dosages(_BIALLELIC, [0, 1])[[1, 0]]
    np.testing.assert_equal(g, _g)
    np.testing.assert_equal(p, _p)
    np.testing.assert_allclose(d, _d, rtol=1e-5)

    pgen.set_samples(None)  # reset for other tests


def samples_none():
    samples = None
    return samples


def samples_second():
    samples = "sample2"
    return samples


def samples_reverse():
    samples = ["sample2", "sample1"]
    return samples


@pytest.mark.xfail(raises=ValueError, reason="sample3 not in file")
def samples_missing():
    samples = ["sample1", "sample3"]
    return samples


def samples_all():
    samples = ["sample1", "sample2"]
    return samples


@pytest.mark.xfail(raises=ValueError, reason="samples must be unique")
def samples_repeat():
    samples = ["sample1", "sample2", "sample2"]
    return samples


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("samples", cases=".", prefix="samples_")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_set_samples(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    samples: ArrayLike | None,
):
    pgen.set_samples(samples)

    if samples is None:
        samples = np.asarray(pgen.available_samples)
        s_idx = slice(None)
    else:
        samples = np.atleast_1d(samples)
        s_idx = pgen._s2i.get(samples).astype(np.uint32)
        if (s_idx == np.arange(len(pgen.available_samples))).all():
            s_idx = slice(None)

    np.testing.assert_array_equal(pgen.current_samples, samples)
    assert pgen.n_samples == len(samples)
    np.testing.assert_equal(pgen._s_idx, s_idx)

    g, p, d = pgen.read(*cse, PGEN.GenosPhasingDosages)
    np.testing.assert_equal(g, genos[s_idx])
    np.testing.assert_equal(p, phasing[s_idx])
    np.testing.assert_allclose(d, dosages[s_idx], rtol=1e-5)


def length_no_ext():
    cse = "chr1", 81264, 81265  # just 81265 in VCF
    # biallelic idx=[2]: chr1:81265 T>C (1|0,./.)
    genos = _oracle.genos(_BIALLELIC, [2])
    phasing = _oracle.phasing(_BIALLELIC, [2])
    # PGEN DS encoding produces 0.900024 for VCF DS=0.9; oracle has exact 0.9 → literal retained
    dosages = np.array([[0.900024], [np.nan]], np.float32)
    last_end = 81265
    var_idxs = np.array([2], dtype=V_IDX_TYPE)
    return cse, genos, phasing, dosages, last_end, var_idxs


def length_ext():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # biallelic idx=[0,1,2]: chr1:81262 GAT>A, chr1:81262 G>A (./.→-1), chr1:81265 T>C (1|0,./.)
    genos = _oracle.genos(_BIALLELIC, [0, 1, 2])
    phasing = _oracle.phasing(_BIALLELIC, [0, 1, 2])
    # PGEN DS encoding produces 0.900024 for VCF DS=0.9 at idx2; oracle has exact 0.9 → literal retained
    dosages = np.array([[1.0, np.nan, 0.900024], [2.0, 1.0, np.nan]], np.float32)
    last_end = 81265
    var_idxs = np.arange(3, dtype=V_IDX_TYPE)
    return cse, genos, phasing, dosages, last_end, var_idxs


def length_none():
    cse = "chr1", 0, 1
    # (s p v)
    genos, phasing, dosages = PGEN.GenosPhasingDosages.empty(N_SAMPLES, PGEN.ploidy, 0)
    # (s v)
    last_end = 1
    var_idxs = np.array([], dtype=V_IDX_TYPE)
    return cse, genos, phasing, dosages, last_end, var_idxs


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases(
    "cse, genos, phasing, dosages, last_end, var_idxs", cases=".", prefix="length_"
)
def test_chunk_with_length(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    last_end: int,
    var_idxs: np.uint32,
):
    mode = PGEN.GenosPhasingDosages
    max_mem = pgen._mem_per_variant(mode)
    gpd = pgen._chunk_ranges_with_length(*cse, max_mem, mode)
    for range_ in gpd:
        for chunk, end, v_idxs in range_:
            g, p, d = chunk
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_allclose(d, dosages, rtol=1e-5)
            assert end == last_end
            np.testing.assert_equal(v_idxs, var_idxs)


def n_vars_miss_chr():
    contig = "chr3"
    starts = 0
    ends = POS_MAX
    desired = np.array([0], dtype=np.uint32)
    return contig, starts, ends, desired


def n_vars_none():
    contig = "chr1"
    starts = 0
    ends = 1
    desired = np.array([0], dtype=np.uint32)
    return contig, starts, ends, desired


def n_vars_all():
    contig = "chr1"
    starts = 0
    ends = POS_MAX
    desired = np.array([3], dtype=np.uint32)
    return contig, starts, ends, desired


def n_vars_spanning_del():
    contig = "chr1"
    starts = 81262
    ends = 81263
    desired = np.array([1], dtype=np.uint32)
    return contig, starts, ends, desired


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("contig, starts, ends, desired", cases=".", prefix="n_vars_")
def test_n_vars_in_ranges(
    pgen: PGEN,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    desired: NDArray[np.uint32],
):
    n_vars = pgen.n_vars_in_ranges(contig, starts, ends)
    assert n_vars == desired


def var_idxs_miss_chr():
    contig = "chr3"
    starts = 0
    ends = POS_MAX
    desired = (np.array([], dtype=V_IDX_TYPE), np.array([0, 0], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_none():
    contig = "chr1"
    starts = 0
    ends = 1
    desired = (np.array([], dtype=V_IDX_TYPE), np.array([0, 0], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_all():
    contig = "chr1"
    starts = 0
    ends = POS_MAX
    desired = (np.array([0, 1, 2], dtype=V_IDX_TYPE), np.array([0, 3], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_spanning_del():
    contig = "chr1"
    starts = 81262
    ends = 81263
    desired = (np.array([0], dtype=V_IDX_TYPE), np.array([0, 1], dtype=np.uint64))
    return contig, starts, ends, desired


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("contig, starts, ends, desired", cases=".", prefix="var_idxs_")
def test_var_idxs(
    pgen: PGEN,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    desired: tuple[NDArray[V_IDX_TYPE], NDArray[np.uint64]],
):
    var_idxs, offsets = pgen.var_idxs(contig, starts, ends)
    assert np.array_equal(var_idxs, desired[0])
    assert np.array_equal(offsets, desired[1])


def test_pgen_nbytes_positive_after_init():
    # PGEN auto-loads the index in __init__ via _init_index
    pgen = PGEN(ddir / "biallelic.pgen")
    assert pgen._index is not None
    assert pgen.nbytes > 0
    # both the index dataframe and the StartsEndsIlens cache should contribute
    assert pgen.nbytes >= pgen._index.estimated_size()


def _fake_read_factory(dense_genos):
    """dense_genos: (s p V) global int32 array. Returns read(var_idx)->Genos."""

    def read(var_idx):
        return Genos.parse(dense_genos[:, :, var_idx].astype(np.int32))

    return read


def test_gen_with_length_multi_round_extension():
    # 1 sample, ploidy 2 (required by Genos predicate). Variant 0 is a -60
    # deletion (carried on both haplotypes). Variants 1..40 are SNPs 1bp apart
    # starting just past the deletion's reference end. The first extension batch
    # of ~20 variants spans only ~20 bp; with a -60 deletion the haplotype
    # length deficit is still unmet, forcing a 2nd doubling round.
    n = 41
    v_starts = np.empty(n, dtype=POS_TYPE)
    v_starts[0] = 2999  # deletion at 0-based 2999
    v_starts[1:] = np.arange(3060, 3060 + (n - 1), dtype=POS_TYPE)
    ilens = np.zeros(n, dtype=np.int32)
    ilens[0] = -60
    v_ends = v_starts + 1
    v_ends[0] = 2999 + 61  # REF length 61

    # all variants carried on both haplotypes (shape: s=1, p=2, V=n)
    dense = np.ones((1, 2, n), dtype=np.int32)
    read = _fake_read_factory(dense)

    q_start, q_end = 2999, 3060  # len 61
    v_chunks = [np.array([0], dtype=V_IDX_TYPE)]  # query window = just the deletion

    out = list(
        _gen_with_length(
            v_chunks=v_chunks,
            q_start=q_start,
            q_end=q_end,
            read=read,
            v_starts=v_starts,
            v_ends=v_ends,
            ilens=ilens,
            contig_max_idx=n - 1,
        )
    )
    final_genos, _end, final_idx = out[-1]
    assert final_idx[-1] > 20, (
        f"multi-round extension not triggered; reached idx {final_idx[-1]}. "
        f"All chunks: {[(np.asarray(g).shape, int(e), fi.tolist()) for g, e, fi in out]}"
    )


def test_gen_with_length_clamps_at_contig_end():
    # Deletion is the last variant -> extension cannot proceed, must clamp.
    v_starts = np.array([4999], dtype=POS_TYPE)
    v_ends = np.array([4999 + 11], dtype=POS_TYPE)
    ilens = np.array([-10], dtype=np.int32)
    # shape: s=1, p=2, V=1 (ploidy=2 required by Genos predicate)
    dense = np.ones((1, 2, 1), dtype=np.int32)
    read = _fake_read_factory(dense)

    out = list(
        _gen_with_length(
            v_chunks=[np.array([0], dtype=V_IDX_TYPE)],
            q_start=4999,
            q_end=5040,
            read=read,
            v_starts=v_starts,
            v_ends=v_ends,
            ilens=ilens,
            contig_max_idx=0,  # this IS the last variant
        )
    )
    final_genos, _end, final_idx = out[-1]
    np.testing.assert_array_equal(final_idx, np.array([0], dtype=V_IDX_TYPE))


def test_pgen_nbytes_zero_after_free():
    pgen = PGEN(ddir / "biallelic.pgen")
    pgen._free_index()
    assert pgen._index is None
    assert pgen._sei is None
    assert pgen.nbytes == 0


def test_filtered_var_idxs_consistent_with_index():
    """Regression for #69: filtered PGEN var_idxs must index its own _index.

    `is_snp` drops the interior GAT>A indels (physical rows 0 and 3), so for a
    filtered reader physical != positional. var_idxs() must return positional
    indices into the (filtered) _index, and reads must still return the correct
    (physical) genotypes.
    """
    from genoray import exprs

    g_filt = PGEN(ddir / "biallelic.pgen", filter=exprs.is_snp)
    g_full = PGEN(ddir / "biallelic.pgen")

    # Filter drops physical rows 0 and 3 -> 4 rows remain.
    assert g_filt._index.height == 4

    # chr2 query: positional [2, 3] (NOT physical [4, 5]).
    vi, offsets = g_filt.var_idxs("chr2", 0, POS_MAX)
    assert int(vi.max()) < g_filt._index.height  # #69: in-bounds
    assert np.array_equal(vi, np.array([2, 3], dtype=vi.dtype))
    assert np.array_equal(offsets, np.array([0, 2], dtype=offsets.dtype))

    # _index[var_idxs] selects the kept chr2 SNPs (POS 81262, 81265).
    assert g_filt._index[vi]["POS"].to_list() == [81262, 81265]

    # Read correctness: a filtered chr1 read returns exactly the same genotypes
    # as the corresponding (physical) variants from an unfiltered read. chr1's
    # SNPs are physical rows [1, 2] -> within the full chr1 read those are the
    # 2nd and 3rd variants on the variant axis.
    filt = g_filt.read("chr1", 0, POS_MAX, mode=Genos)  # (s, p, 2)
    full = g_full.read("chr1", 0, POS_MAX, mode=Genos)  # (s, p, 3)
    assert filt.shape[-1] == 2
    assert np.array_equal(filt, full[..., [1, 2]])


def test_del_does_not_double_close_shared_reader():
    pgen = PGEN(ddir / "biallelic.pgen")  # no separate dosage path
    assert pgen._dose_pgen is pgen._geno_pgen  # same underlying reader

    class _CloseCounter:
        def __init__(self):
            self.n = 0

        def close(self):
            self.n += 1

    counter = _CloseCounter()
    pgen._geno_pgen = counter
    pgen._dose_pgen = counter
    pgen.__del__()
    assert counter.n == 1  # closed exactly once, not twice
