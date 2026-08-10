"""M6b: raw two-channel batch-query methods for :class:`SparseVar2`."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from genoray import _core

#: Bit width reserved for ``ext`` in a packed max-end key. Mirrors Rust's
#: ``query::MAX_END_SHIFT``; consumers unpack with
#: ``end = (key >> MAX_END_SHIFT) + (key & ((1 << MAX_END_SHIFT) - 1))``.
MAX_END_SHIFT = 21


@dataclass(frozen=True)
class RangesChunk:
    """One hap slice of a chunked ``_find_ranges``.

    Attributes:
        sample_start: Offset of this chunk on the SELECTED sample axis.
        n_samples: Number of selected samples in this chunk.
        vk_snp_range: Shape ``(n_samples, ploidy, n_regions, 2)``, hap-major.
        vk_indel_range: Shape ``(n_samples, ploidy, n_regions, 2)``, hap-major.
        max_end_keys: Shape ``(n_regions,)``. Packed ``(pos << MAX_END_SHIFT) |
            ext`` maxima over this chunk's haps; ``0`` means no variant. Reduce
            across chunks with an elementwise maximum BEFORE unpacking -- the
            ordering rule is position first, end second, so reducing unpacked
            ends would pick the wrong variant.
    """

    sample_start: int
    n_samples: int
    vk_snp_range: "np.ndarray"
    vk_indel_range: "np.ndarray"
    max_end_keys: "np.ndarray"


@dataclass(frozen=True)
class RangesStream:
    """Memory-bounded, chunked form of ``_find_ranges``.

    The ``O(n_regions)`` arrays are computed eagerly; the
    ``O(n_regions * n_samples * ploidy)`` payload arrives via ``chunks``.
    ``n_samples`` is the progress denominator and each ``RangesChunk`` reports
    how many samples it advanced by.
    """

    n_regions: int
    n_samples: int
    ploidy: int
    samples_per_chunk: int
    region_starts: "np.ndarray"
    dense_range: "np.ndarray"
    dense_snp_range: "np.ndarray"
    dense_indel_range: "np.ndarray"
    sample_cols: "np.ndarray"
    dense_max_end_keys: "np.ndarray"
    chunks: "Iterator[RangesChunk]"


class BatchResult(TypedDict):
    """Two-channel batch-query result contract (see py_query_batch.rs)."""

    vk_pos: np.ndarray
    vk_key: np.ndarray
    vk_off: np.ndarray
    dense_pos: np.ndarray
    dense_key: np.ndarray
    dense_range: np.ndarray
    dense_present: np.ndarray
    dense_present_off: np.ndarray
    lut_bytes: np.ndarray
    lut_off: np.ndarray
    n_regions: int
    n_samples: int
    ploidy: int


class RangesBundle(TypedDict):
    """Compact search-only bundle replayed by ``_gather_ranges`` (see py_query_ranges.rs)."""

    dense_range: np.ndarray
    region_starts: np.ndarray
    sample_cols: np.ndarray
    vk_snp_range: np.ndarray
    vk_indel_range: np.ndarray
    dense_snp_range: np.ndarray
    dense_indel_range: np.ndarray
    n_regions: int
    n_samples: int
    ploidy: int


class BatchResultSplit(TypedDict):
    """Split-dense batch-query result contract (see py_query_ranges.rs).

    As :class:`BatchResult`, except the single unified ``dense_*`` channel is
    replaced by the per-class ``dense_snp_*`` / ``dense_indel_*`` pair, because
    the read-bound path never builds the contig-wide dense union. Consumers
    merge ``var_key``, ``dense_snp`` and ``dense_indel`` by position downstream.
    ``n_samples`` is always 1 and the hap index is ``q * ploidy + p``.
    """

    vk_pos: np.ndarray
    vk_key: np.ndarray
    vk_off: np.ndarray
    dense_snp_pos: np.ndarray
    dense_snp_key: np.ndarray
    dense_snp_range: np.ndarray
    dense_snp_present: np.ndarray
    dense_snp_present_off: np.ndarray
    dense_indel_pos: np.ndarray
    dense_indel_key: np.ndarray
    dense_indel_range: np.ndarray
    dense_indel_present: np.ndarray
    dense_indel_present_off: np.ndarray
    lut_bytes: np.ndarray
    lut_off: np.ndarray
    n_regions: int
    n_samples: int
    ploidy: int


class HapRanges(TypedDict):
    """Flat per-``(region, sample)`` search result replayed by ``_gather_haps_readbound``.

    One row ``q`` per query *pair*, so an arbitrary (non-rectangular) set of
    ``(region, sample)`` cells costs exactly those cells -- unlike
    :class:`RangesBundle`, whose ``sample_cols`` axis makes it inherently a
    region-by-sample rectangle. Dtypes match :class:`RangesBundle`'s so a
    bundle's arrays can be sliced straight in.

    Attributes:
        region_starts: Shape ``(n_q,)`` int32. Each query's ``q_start``.
        orig_samples: Shape ``(n_q,)`` int64. Each query's sample index in the
            store's FULL cohort (not a subset slot).
        vk_snp_range: Shape ``(n_q * ploidy, 2)`` int64, row ``q * ploidy + p``.
        vk_indel_range: Shape ``(n_q * ploidy, 2)`` int64, row ``q * ploidy + p``.
        dense_snp_range: Shape ``(n_q, 2)`` int32. Dense is cohort-shared, so
            this is per-query, not per-hap.
        dense_indel_range: Shape ``(n_q, 2)`` int32.
        ploidy: Must equal the contig's ploidy.
    """

    region_starts: np.ndarray
    orig_samples: np.ndarray
    vk_snp_range: np.ndarray
    vk_indel_range: np.ndarray
    dense_snp_range: np.ndarray
    dense_indel_range: np.ndarray
    ploidy: int


@dataclass(frozen=True)
class HapRangesRect:
    """Search output for the read-bound gather.

    The region-by-sample rectangle of var_key ranges plus the per-region,
    cohort-shared dense class windows.

    Produced by ``_find_haps_ranges``; :meth:`select` folds it down to the flat
    :class:`HapRanges` the gather replays. Splitting search (rectangle) from
    gather (flat) is deliberate: the var_key search is column-outer, so
    searching a sample's whole region list at once is what amortizes its index
    build, while the gather should touch only the cells asked for.

    Attributes:
        region_starts: Shape ``(n_regions,)`` int32.
        sample_cols: Shape ``(n_samples,)`` int64 -- selected samples' indices in
            the store's full cohort.
        vk_snp_range: Shape ``(n_samples, ploidy, n_regions, 2)`` int64.
        vk_indel_range: Shape ``(n_samples, ploidy, n_regions, 2)`` int64.
        dense_snp_range: Shape ``(n_regions, 2)`` int32.
        dense_indel_range: Shape ``(n_regions, 2)`` int32.
    """

    n_regions: int
    n_samples: int
    ploidy: int
    region_starts: "np.ndarray"
    sample_cols: "np.ndarray"
    vk_snp_range: "np.ndarray"
    vk_indel_range: "np.ndarray"
    dense_snp_range: "np.ndarray"
    dense_indel_range: "np.ndarray"

    def select(self, regions: "ArrayLike", samples: "ArrayLike") -> HapRanges:
        """Fold this rectangle down to the flat cells the parallel index arrays name.

        Args:
            regions: Row indices into ``region_starts``.
            samples: Slots on THIS rectangle's ``sample_cols`` axis (not
                full-cohort sample indices).

        Raises:
            ValueError: If the two index arrays have different lengths.
        """
        r = np.asarray(regions, dtype=np.intp).ravel()
        s = np.asarray(samples, dtype=np.intp).ravel()
        if r.shape != s.shape:
            raise ValueError(
                f"regions and samples must be parallel, got {r.shape} and {s.shape}"
            )
        # Advanced indexing on axes 0 and 2 with a slice between them puts the
        # broadcast (n_q) axis FIRST, giving (n_q, ploidy, 2) -- i.e. exactly
        # HapRanges' row order q*ploidy + p once flattened.
        shape = (r.size * self.ploidy, 2)
        return HapRanges(
            region_starts=np.ascontiguousarray(self.region_starts[r]),
            orig_samples=np.ascontiguousarray(self.sample_cols[s]),
            vk_snp_range=self.vk_snp_range[s, :, r, :].reshape(shape),
            vk_indel_range=self.vk_indel_range[s, :, r, :].reshape(shape),
            dense_snp_range=np.ascontiguousarray(self.dense_snp_range[r]),
            dense_indel_range=np.ascontiguousarray(self.dense_indel_range[r]),
            ploidy=self.ploidy,
        )


class _BatchQueryMixin:
    """Raw ``BatchResult`` → numpy query methods."""

    # Provided by the concrete SparseVar2 host class (see SparseVar2.__init__);
    # declared here so the mixin's use of them type-checks in isolation.
    _readers: dict[str, Any]
    available_samples: list[str]
    ploidy: int

    def _reader(self, contig: str) -> "_core.PyContigReader":  # host-provided
        ...

    def _overlap_batch(
        self, contig: str, regions: Iterable[tuple[int, int]]
    ) -> BatchResult:
        """Batched two-channel query for one ``contig``.

        ``regions`` is an iterable of half-open ``(q_start, q_end)`` pairs. Returns
        the frozen ``BatchResult`` → numpy contract as a dict of arrays (see the M6b
        plan). Cross-contig batching is the caller's job (query each contig).
        """
        reg = [(int(s), int(e)) for s, e in regions]
        return self._reader(contig).overlap_batch(reg)

    @staticmethod
    def _regions(starts: "ArrayLike", ends: "ArrayLike") -> list[tuple[int, int]]:
        s = np.atleast_1d(np.asarray(starts))
        e = np.atleast_1d(np.asarray(ends))
        return [(int(a), int(b)) for a, b in zip(s, e)]

    def _sample_idxs(self, samples: "ArrayLike | None") -> list[int] | None:
        if samples is None:
            return None
        idxs = []
        for s in np.atleast_1d(np.asarray(samples)).tolist():
            if s not in self.available_samples:
                raise ValueError(f"Sample {s!r} not found in the dataset.")
            idxs.append(self.available_samples.index(s))
        return idxs

    def read_ranges(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
    ) -> BatchResult:
        """Fused search+gather query for one ``contig``.

        ``starts``/``ends`` are parallel 1D arrays of half-open ``(start, end)``
        region bounds (mirrors ``SparseVar.read_ranges``'s ``starts``/``ends``
        signature rather than ``_overlap_batch``'s ``regions`` iterable). When
        ``samples=None`` the result is byte-identical to ``_overlap_batch`` over
        the same regions; the returned dict has the exact same contract
        (``vk_pos``/``vk_key``/``vk_off``, ``dense_*``, ``lut_*``, plus
        ``n_regions``/``n_samples``/``ploidy``). ``samples``, if given, is a
        list of sample names selecting (and reordering) a subset by name.
        """
        reg = self._regions(starts, ends)
        return self._reader(contig).read_ranges(reg, self._sample_idxs(samples))

    def _find_ranges(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
        out: Mapping[str, "np.ndarray"] | None = None,
    ) -> RangesBundle:
        """Search-only step: returns a compact ranges bundle to be replayed by ``_gather_ranges``, doing no per-element gather.

        ``starts``/``ends`` and ``samples`` behave as in ``read_ranges``.

        If ``out`` is given, it must be a dict of preallocated arrays keyed by
        the bundle's field names (e.g. ``dense_range``, ``region_starts``,
        ``sample_cols``, ``vk_snp_range``, ``vk_indel_range``); each is
        overwritten in place with the freshly computed values and the same
        buffer is returned in the result dict, so repeated calls can reuse
        caller-owned memory instead of allocating a new bundle each time.
        """
        reg = self._regions(starts, ends)
        d = self._reader(contig).find_ranges(reg, self._sample_idxs(samples))
        if out is not None:
            for k, buf in out.items():
                src = np.asarray(d[k])
                dst = np.asarray(buf)
                if dst.shape != src.shape:
                    raise ValueError(
                        f"out[{k!r}] has shape {dst.shape}, expected {src.shape}"
                    )
                if dst.dtype != src.dtype:
                    raise ValueError(
                        f"out[{k!r}] has dtype {dst.dtype}, expected {src.dtype}"
                    )
                dst[...] = src
                d[k] = buf
        return d

    def _gather_ranges(
        self,
        contig: str,
        ranges: dict[str, Any],
        samples: "ArrayLike | None" = None,
    ) -> BatchResult:
        """Tree-free gather step: replay a ``_find_ranges`` bundle into the same dict contract as ``_overlap_batch``/``read_ranges``, with no further search-tree work.

        ``samples`` is accepted only for call-signature symmetry with
        ``read_ranges``/``_find_ranges``: the sample subset is already fixed by
        ``ranges`` (it was baked in when the bundle was produced), so passing a
        ``samples`` value that disagrees with the bundle's ``sample_cols`` is a
        ``ValueError``; passing ``None`` (or a value that matches) is a no-op.
        """
        if samples is not None:
            want = self._sample_idxs(samples)
            have = np.asarray(ranges["sample_cols"]).tolist()
            if want != have:
                raise ValueError(
                    "samples does not match the bundle's fixed subset "
                    f"(got {want!r}, bundle has {have!r})"
                )
        return self._reader(contig).gather_ranges(ranges)

    def _find_haps_ranges(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
    ) -> HapRangesRect:
        """Search-only step for the READ-BOUND gather.

        Returns var_key ranges for the region-by-sample rectangle plus the
        per-region dense class windows.

        ``starts``/``ends`` and ``samples`` behave as in :meth:`read_ranges`.
        Unlike :meth:`_find_ranges` this builds no contig-wide dense union, so
        the result can only be replayed by :meth:`_gather_haps_readbound` (via
        :meth:`HapRangesRect.select`), never by :meth:`_gather_ranges`.
        """
        reg = self._regions(starts, ends)
        sample_idxs = self._sample_idxs(samples)
        reader = self._reader(contig)
        ploidy = self.ploidy
        n_regions = len(reg)
        n_samples = (
            len(sample_idxs) if sample_idxs is not None else len(self.available_samples)
        )

        vk = reader.find_ranges_chunk(reg, sample_idxs, 0, n_samples * ploidy)
        dense = reader.find_dense_class_ranges(reg)
        shape = (n_samples, ploidy, n_regions, 2)
        cols = (
            np.asarray(sample_idxs, np.int64)
            if sample_idxs is not None
            else np.arange(n_samples, dtype=np.int64)
        )
        return HapRangesRect(
            n_regions=n_regions,
            n_samples=n_samples,
            ploidy=ploidy,
            region_starts=np.asarray([s for s, _ in reg], np.int32),
            sample_cols=cols,
            vk_snp_range=np.asarray(vk["vk_snp_range"]).reshape(shape),
            vk_indel_range=np.asarray(vk["vk_indel_range"]).reshape(shape),
            dense_snp_range=np.asarray(dense["dense_snp_range"]),
            dense_indel_range=np.asarray(dense["dense_indel_range"]),
        )

    def _gather_haps_readbound(
        self, contig: str, hap_ranges: Mapping[str, Any]
    ) -> BatchResultSplit:
        """Tree-free read-bound gather.

        Replays a flat :class:`HapRanges` into a :class:`BatchResultSplit`,
        touching only the ``(region, sample)`` cells it names.

        The exact-cell counterpart of :meth:`_gather_ranges`, whose
        :class:`RangesBundle` is inherently a region-by-sample rectangle: an
        arbitrary pair set has to be covered either by over-gathering the
        cross-product or by one bundle per sample. This path does neither, and
        (unlike ``_gather_ranges``) never rebuilds the contig-wide dense union.
        """
        return self._reader(contig).gather_haps_readbound(dict(hap_ranges))

    def _find_ranges_chunked(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
        *,
        max_mem: int | None = None,
    ) -> RangesStream:
        """Chunked, memory-bounded ``_find_ranges``.

        ``starts``/``ends`` and ``samples`` behave as in :meth:`read_ranges`.

        The var_key payload is ``n_regions * n_samples * ploidy * 2`` int64
        pairs per channel, which is tens of GiB at cohort scale. This splits it
        along the SAMPLE axis -- not the region axis -- because the search is
        column-outer: chunking regions would re-sweep the whole packed store per
        chunk, while chunking samples keeps a single sweep.

        Args:
            contig: Contig name.
            starts: 0-based start positions of the query regions.
            ends: 0-based, exclusive end positions of the query regions.
            samples: Sample names selecting (and reordering) a subset.
            max_mem: Approximate byte budget for one chunk's payload. ``None``
                yields a single chunk covering every sample.

        Returns:
            A :class:`RangesStream` whose ``chunks`` generator yields
            :class:`RangesChunk` in ascending ``sample_start`` order.

        Raises:
            ValueError: If ``max_mem`` cannot fit a single sample's payload, or
                if the contig's largest deletion overflows the max-end key
                packing width.
        """
        reg = self._regions(starts, ends)
        sample_idxs = self._sample_idxs(samples)
        reader = self._reader(contig)
        header = reader.find_ranges_header(reg, sample_idxs)

        n_regions = int(header["n_regions"])
        n_samples = int(header["n_samples"])
        ploidy = int(header["ploidy"])

        # `bytes_per_sample` is the real per-sample payload: both channels
        # (snp + indel) x 2 endpoints x int64 x ploidy x regions. The extra
        # `2 *` in the division below -- not this factor -- is the safety
        # margin, covering the transient the binding holds while handing the
        # freshly filled arrays back.
        bytes_per_sample = n_regions * ploidy * 2 * 8 * 2
        if max_mem is None:
            per = max(n_samples, 1)
        else:
            per = (
                int(max_mem) // (2 * bytes_per_sample)
                if bytes_per_sample
                else n_samples
            )
            if per < 1:
                raise ValueError(
                    f"max_mem ({int(max_mem)} bytes) is too small for even one "
                    f"sample of {n_regions} regions at ploidy {ploidy}: needs at "
                    f"least {2 * bytes_per_sample} bytes."
                )
            per = min(per, max(n_samples, 1))

        def _gen() -> "Iterator[RangesChunk]":
            for s0 in range(0, n_samples, per):
                s1 = min(s0 + per, n_samples)
                d = reader.find_ranges_chunk(reg, sample_idxs, s0 * ploidy, s1 * ploidy)
                cs = s1 - s0
                shape = (cs, ploidy, n_regions, 2)
                yield RangesChunk(
                    sample_start=s0,
                    n_samples=cs,
                    vk_snp_range=np.asarray(d["vk_snp_range"]).reshape(shape),
                    vk_indel_range=np.asarray(d["vk_indel_range"]).reshape(shape),
                    max_end_keys=np.asarray(d["max_end_keys"], np.int64),
                )

        return RangesStream(
            n_regions=n_regions,
            n_samples=n_samples,
            ploidy=ploidy,
            samples_per_chunk=per,
            region_starts=np.asarray(header["region_starts"]),
            dense_range=np.asarray(header["dense_range"]),
            dense_snp_range=np.asarray(header["dense_snp_range"]),
            dense_indel_range=np.asarray(header["dense_indel_range"]),
            sample_cols=np.asarray(header["sample_cols"]),
            dense_max_end_keys=np.asarray(header["dense_max_end_keys"], np.int64),
            chunks=_gen(),
        )
