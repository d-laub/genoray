"""M6b: raw two-channel batch-query methods for :class:`SparseVar2`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class _BatchQueryMixin:
    """Raw ``BatchResult`` → numpy query methods."""

    # Provided by the concrete SparseVar2 host class (see SparseVar2.__init__);
    # declared here so the mixin's use of it type-checks in isolation.
    _readers: dict[str, Any]

    def overlap_batch(
        self, contig: str, regions: Iterable[tuple[int, int]]
    ) -> dict[str, "np.ndarray"]:
        """Batched two-channel query for one ``contig``.

        ``regions`` is an iterable of half-open ``(q_start, q_end)`` pairs. Returns
        the frozen ``BatchResult`` → numpy contract as a dict of arrays (see the M6b
        plan). Cross-contig batching is the caller's job (query each contig).
        """
        reg = [(int(s), int(e)) for s, e in regions]
        return self._readers[contig].overlap_batch(reg)

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
            if s not in self.samples:
                raise ValueError(f"Sample {s!r} not found in the dataset.")
            idxs.append(self.samples.index(s))
        return idxs

    def read_ranges(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
    ) -> dict[str, "np.ndarray"]:
        """Fused search+gather query for one ``contig``.

        ``starts``/``ends`` are parallel 1D arrays of half-open ``(start, end)``
        region bounds (mirrors ``SparseVar.read_ranges``'s ``starts``/``ends``
        signature rather than ``overlap_batch``'s ``regions`` iterable). When
        ``samples=None`` the result is byte-identical to ``overlap_batch`` over
        the same regions; the returned dict has the exact same contract
        (``vk_pos``/``vk_key``/``vk_off``, ``dense_*``, ``lut_*``, plus
        ``n_regions``/``n_samples``/``ploidy``). ``samples``, if given, is a
        list of sample names selecting (and reordering) a subset by name.
        """
        reg = self._regions(starts, ends)
        return self._readers[contig].read_ranges(reg, self._sample_idxs(samples))

    def find_ranges(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
        out: dict[str, "np.ndarray"] | None = None,
    ) -> dict[str, Any]:
        """Search-only step: returns a compact ranges bundle to be replayed by
        ``gather_ranges``, doing no per-element gather. ``starts``/``ends`` and
        ``samples`` behave as in ``read_ranges``.

        If ``out`` is given, it must be a dict of preallocated arrays keyed by
        the bundle's field names (e.g. ``dense_range``, ``region_starts``,
        ``sample_cols``, ``vk_snp_range``, ``vk_indel_range``); each is
        overwritten in place with the freshly computed values and the same
        buffer is returned in the result dict, so repeated calls can reuse
        caller-owned memory instead of allocating a new bundle each time.
        """
        reg = self._regions(starts, ends)
        d = self._readers[contig].find_ranges(reg, self._sample_idxs(samples))
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

    def gather_ranges(
        self,
        contig: str,
        ranges: dict[str, Any],
        samples: "ArrayLike | None" = None,
    ) -> dict[str, "np.ndarray"]:
        """Tree-free gather step: replay a ``find_ranges`` bundle into the same
        dict contract as ``overlap_batch``/``read_ranges``, with no further
        search-tree work.

        ``samples`` is accepted only for call-signature symmetry with
        ``read_ranges``/``find_ranges``: the sample subset is already fixed by
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
        return self._readers[contig].gather_ranges(ranges)
