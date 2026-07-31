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


class _BatchQueryMixin:
    """Raw ``BatchResult`` → numpy query methods."""

    # Provided by the concrete SparseVar2 host class (see SparseVar2.__init__);
    # declared here so the mixin's use of them type-checks in isolation.
    _readers: dict[str, Any]
    available_samples: list[str]

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
