from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np
from numpy.typing import NDArray
from seqpro.rag import OFFSET_TYPE

from .._types import DOSAGE_TYPE, DTYPE, POS_TYPE, V_IDX_TYPE


@nb.njit(nogil=True, cache=True)
def _nb_af_helper(
    afs: NDArray[np.float32],
    v_idxs: NDArray[np.int32],
    offsets: NDArray[np.int64],
    max_count: int,
):
    for i in range(len(offsets) - 1):
        o_s, o_e = offsets[i], offsets[i + 1]
        v_slice = v_idxs[o_s:o_e]
        afs[v_slice] += 1
    afs /= max_count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_count_kept(
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    out_lengths: NDArray[np.int64],
):
    """Pass 1: count, per output (sample, ploidy) slot, how many source variant
    indices fall in `kept_var_idxs`."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):  # type: ignore[not-iterable]
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            count = 0
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    count += 1
            out_lengths[i * ploidy + p] = count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_count_mac_per_kept(
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    mac_out: NDArray[np.int64],
):
    """Count, per kept variant, the number of non-ref entries across (sample, ploidy)
    in the output. Outer prange is over kept variants so each writes its own slot —
    no atomics needed."""
    n_kept = kept_var_idxs.shape[0]
    n_samples = src_sample_idxs.shape[0]
    for k in nb.prange(n_kept):  # type: ignore[not-iterable]
        v = kept_var_idxs[k]
        count = 0
        for i in range(n_samples):
            s = src_sample_idxs[i]
            for p in range(ploidy):
                src_slot = s * ploidy + p
                lo = src_offsets[src_slot]
                hi = src_offsets[src_slot + 1]
                idx = np.searchsorted(src_data[lo:hi], v)
                if idx < (hi - lo) and src_data[lo + idx] == v:
                    count += 1
        mac_out[k] = count


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_write_var_idxs(
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    new_offsets: NDArray[np.int64],
    out_var_idxs: NDArray[np.int32],
):
    """Pass 2: write remapped variant indices."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):  # type: ignore[not-iterable]
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_var_idxs[wp] = k
                    wp += 1


@nb.njit(parallel=True, nogil=True, cache=True)
def _nb_write_field(
    src_field: NDArray[Any],
    src_data: NDArray[np.int32],
    src_offsets: NDArray[np.int64],
    src_sample_idxs: NDArray[np.int64],
    ploidy: int,
    kept_var_idxs: NDArray[np.int32],
    new_offsets: NDArray[np.int64],
    out_field: NDArray[Any],
):
    """Pass 2 (field variant): writes src_field values at filter-kept positions."""
    n_out = src_sample_idxs.shape[0]
    n_kept = kept_var_idxs.shape[0]
    for i in nb.prange(n_out):  # type: ignore[not-iterable]
        s = src_sample_idxs[i]
        for p in range(ploidy):
            src_slot = s * ploidy + p
            out_slot = i * ploidy + p
            wp = new_offsets[out_slot]
            lo = src_offsets[src_slot]
            hi = src_offsets[src_slot + 1]
            for j in range(lo, hi):
                v = src_data[j]
                k = np.searchsorted(kept_var_idxs, v)
                if k < n_kept and kept_var_idxs[k] == v:
                    out_field[wp] = src_field[j]
                    wp += 1


@nb.njit(parallel=True, nogil=True, cache=True)
def _copy_chunk_helper(
    out_data: NDArray[DTYPE],
    write_offsets: NDArray[OFFSET_TYPE],
    in_data: NDArray[DTYPE],
    in_offsets: NDArray[OFFSET_TYPE],
    variant_offset: int,
    n_samples: int,
    ploidy: int,
):
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in range(ploidy):
            sp = s * ploidy + p

            i_s, i_e = in_offsets[sp], in_offsets[sp + 1]
            length = i_e - i_s

            o_s = write_offsets[sp]

            # Copy and add offset
            for i in range(length):
                out_data[o_s + i] = in_data[i_s + i] + variant_offset  # type: ignore

            write_offsets[sp] += length


@nb.njit(parallel=True, nogil=True, cache=True)
def _copy_chunk_dosages_helper(
    out_data: NDArray[DOSAGE_TYPE],
    write_offsets: NDArray[OFFSET_TYPE],
    in_data: NDArray[DOSAGE_TYPE],
    in_offsets: NDArray[OFFSET_TYPE],
    n_samples: int,
    ploidy: int,
):
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in range(ploidy):
            sp = s * ploidy + p

            i_s, i_e = in_offsets[sp], in_offsets[sp + 1]
            length = i_e - i_s

            o_s = write_offsets[sp]

            out_data[o_s : o_s + length] = in_data[i_s:i_e]

            write_offsets[sp] += length


@nb.njit(parallel=True, nogil=True, cache=True)
def _find_starts_ends(
    genos: NDArray[V_IDX_TYPE],
    geno_offsets: NDArray[OFFSET_TYPE],
    var_ranges: NDArray[V_IDX_TYPE],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
    out_offsets: NDArray[OFFSET_TYPE] | None = None,
):
    """Find the start and end offsets of the sparse genotypes for each range.

    Parameters
    ----------
    genos
        Sparse genotypes
    geno_offsets
        Genotype offsets
    var_ranges
        Shape = (ranges 2) Variant index ranges.
    sample_idxs
        Sample indices
    ploidy
        Ploidy
    out_offsets
        Output array to write to. If None, a new array will be created.

    Returns
    -------
        Shape: (ranges samples ploidy 2). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(var_ranges)
    n_samples = len(sample_idxs)
    if out_offsets is None:
        out_offsets = np.empty((2, n_ranges, n_samples, ploidy), dtype=OFFSET_TYPE)
    sorter = np.argsort(var_ranges[:, 0])
    var_ranges = var_ranges[sorter]

    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in nb.prange(ploidy):  # type: ignore[not-iterable]
            s_idx = sample_idxs[s]
            sp = s_idx * ploidy + p
            o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
            sp_genos = genos[o_s:o_e]
            # add o_s to make indices relative to whole array
            out_offsets[..., s, p] = np.searchsorted(sp_genos, var_ranges).T + o_s

    # Ranges with no overlapping variants already get start == stop from
    # searchsorted above (an in-bounds, zero-length range). Do NOT overwrite
    # them with a sentinel: an out-of-range value (e.g. INT64_MAX) is poison for
    # downstream byte-offset math (seqpro Ragged.to_packed multiplies the offset
    # by the element size and overflows int64), even though the row is empty.

    unsorter = np.argsort(sorter)
    out_offsets[:] = out_offsets[:, unsorter]

    return out_offsets


@nb.njit(nogil=True, cache=True)
def _length_walk_n_keep(
    sp_genos: NDArray[V_IDX_TYPE],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    start_idx: int,
    max_idx: int,
    q_start: POS_TYPE,
    q_end: POS_TYPE,
) -> int:
    """Number of leading variants in ``sp_genos[start_idx:max_idx]`` to include
    so one haplotype reaches ``q_end - q_start`` in length, extending past
    ``q_end`` only as needed. Variants strictly inside ``[q_start, q_end)`` are
    always included; the length budget only gates extension past ``q_end``.
    Returns a count in ``[0, max_idx - start_idx]``."""
    q_len = q_end - q_start
    last_v_end = q_start
    written_len = 0
    for j in range(start_idx, max_idx):
        v_idx = sp_genos[j]
        v_start = v_starts[v_idx]
        ilen = ilens[v_idx]

        maybe_add_one = POS_TYPE(v_start >= q_start)

        if v_start >= q_start:
            past_query = v_start >= q_end
            written_len += v_start - last_v_end
            if past_query and written_len >= q_len:
                return j - start_idx  # exclude this variant
            written_len += max(0, ilen) + maybe_add_one
            if past_query and written_len >= q_len:
                return j - start_idx + 1  # include this variant

        v_end = v_start - min(0, ilen) + maybe_add_one
        last_v_end = max(last_v_end, v_end)  # type: ignore[bad-specialization]

    return max_idx - start_idx


@nb.njit(parallel=True, nogil=True, cache=True)
def _dense2sparse_count(
    genos: NDArray[np.integer],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    q_start: POS_TYPE,
    q_end: POS_TYPE,
    out_lengths: NDArray[np.int64],
) -> None:
    """Pass 1: per (sample, haplotype), count the carried ALT calls to keep.

    Gathers each haplotype's carried (``== 1``) window-local positions in order
    and routes them through :func:`_length_walk_n_keep` (the SAME walk the sparse
    path uses, so the two cannot drift). Writes the kept count to ``out_lengths``.
    """
    n_samples, ploidy, n_var = genos.shape
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        carriers = np.empty(n_var, dtype=V_IDX_TYPE)
        for p in range(ploidy):
            nc = 0
            for v in range(n_var):
                if genos[s, p, v] == 1:
                    carriers[nc] = v
                    nc += 1
            out_lengths[s, p] = _length_walk_n_keep(
                carriers, v_starts, ilens, 0, nc, q_start, q_end
            )


@nb.njit(parallel=True, nogil=True, cache=True)
def _dense2sparse_fill(
    genos: NDArray[np.integer],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: NDArray[DOSAGE_TYPE],
    out_lengths: NDArray[np.int64],
    flat_offsets: NDArray[OFFSET_TYPE],
    out_data: NDArray[V_IDX_TYPE],
    out_dose: NDArray[DOSAGE_TYPE],
    has_dose: bool,
) -> None:
    """Pass 2: emit the first ``out_lengths[s, p]`` carried ALT calls per
    haplotype into the disjoint output range ``[flat_offsets[slot], ...)``."""
    n_samples, ploidy, n_var = genos.shape
    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in range(ploidy):
            slot = s * ploidy + p
            n_keep = out_lengths[s, p]
            w = flat_offsets[slot]
            kept = 0
            for v in range(n_var):
                if kept >= n_keep:
                    break
                if genos[s, p, v] == 1:
                    out_data[w] = var_idxs[v]
                    if has_dose:
                        out_dose[w] = dosages[s, v]
                    w += 1
                    kept += 1


@nb.njit(parallel=False, nogil=True, cache=True)
def _find_starts_ends_with_length(
    genos: NDArray[V_IDX_TYPE],
    geno_offsets: NDArray[OFFSET_TYPE],
    q_starts: NDArray[POS_TYPE],
    q_ends: NDArray[POS_TYPE],
    var_ranges: NDArray[V_IDX_TYPE],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
    contig_max_idx: int,
    out: NDArray[OFFSET_TYPE] | None = None,
):
    """Find the start and end offsets of the sparse genotypes for each range.

    Parameters
    ----------
    genos
        Sparse genotypes
    geno_offsets
        Genotype offsets
    var_ranges
        Shape = (ranges 2) Variant index ranges.

    Notes
    -----
    Correctness requires that ``argsort(q_starts) == argsort(var_ranges[:, 0])``,
    i.e. that the per-range query positions and variant-index ranges are
    co-monotone in input order. This holds whenever ``var_ranges`` is derived
    from ``(q_starts, q_ends)`` (e.g. via ``SparseVar.var_ranges``). The
    function sorts ``var_ranges`` internally but indexes ``q_starts`` /
    ``q_ends`` by the same sorted position, so violating this invariant will
    produce results aligned to the wrong query.

    Returns
    -------
        Shape: (2 ranges samples ploidy). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(q_starts)
    n_samples = len(sample_idxs)
    if out is None:
        out = np.empty((2, n_ranges, n_samples, ploidy), dtype=OFFSET_TYPE)

    sorter = np.argsort(var_ranges[:, 0])
    var_ranges = var_ranges[sorter]

    for s in nb.prange(n_samples):  # type: ignore[not-iterable]
        for p in nb.prange(ploidy):  # type: ignore[not-iterable]
            s_idx = sample_idxs[s]
            sp = s_idx * ploidy + p
            o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
            sp_genos = genos[o_s:o_e]

            max_idx = np.searchsorted(sp_genos, contig_max_idx + 1)
            start_idxs = np.searchsorted(sp_genos, var_ranges[:, 0])

            for r in range(n_ranges):
                start_idx: np.intp = start_idxs[r]

                if var_ranges[r, 0] == var_ranges[r, 1]:
                    # No overlapping variants: emit an in-bounds, zero-length
                    # range (start == stop) rather than an INT64_MAX sentinel,
                    # which would overflow downstream byte-offset math even
                    # though the row is empty.
                    out[:, r, s, p] = start_idx + o_s
                    continue

                # add o_s to make indices relative to whole array
                out[0, r, s, p] = start_idx + o_s
                if start_idx == max_idx:
                    # no variants in this range
                    out[1, r, s, p] = start_idx + o_s
                    continue

                n_keep = _length_walk_n_keep(
                    sp_genos,
                    v_starts,
                    ilens,
                    int(start_idx),
                    int(max_idx),
                    q_starts[r],
                    q_ends[r],
                )
                out[1, r, s, p] = start_idx + o_s + n_keep

    unsorter = np.argsort(sorter)
    out[:] = out[:, unsorter]

    return out
