"""M6b: raw two-channel batch-query methods for :class:`SparseVar2`."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class _BatchQueryMixin:
    """Raw ``BatchResult`` → numpy query methods."""

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
