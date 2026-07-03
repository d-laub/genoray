"""M6c: decoded ``seqpro.rag.Ragged`` + region-count methods for ``SparseVar2``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from seqpro.rag import Ragged


class _DecodeMixin:
    """Decoded-record and decode-free-count query methods."""

    def decode(self, contig: str, regions: Iterable[tuple[int, int]]) -> "Ragged":
        """Materialize overlapping variants for ``contig`` into a record ``Ragged``.

        Fields ``pos`` (i32), ``ilen`` (i32), ``allele`` (opaque-string ALT bytes),
        one shared variant-axis offsets object, shape ``(R, S, P, None)`` — the same
        layout as gvl's ``RaggedVariants``. Pure-deletion ALT is empty.
        """
        from seqpro.rag import Ragged

        reg = [(int(s), int(e)) for s, e in regions]
        d = self._readers[contig].decode_batch(reg)
        shape = (d["n_regions"], d["n_samples"], d["ploidy"], None)
        off = d["off"]
        pos = Ragged.from_offsets(d["pos"], shape, off)
        ilen = Ragged.from_offsets(d["ilen"], shape, off)
        allele = Ragged.from_offsets(
            d["allele"].view("S1"), shape, off, str_offsets=d["str_off"]
        )
        # If a consumer hits an error reading `.lengths` on a (2, N) offsets
        # layout, call `.to_packed()` first — a known seqpro slicing quirk.
        return Ragged.from_fields({"pos": pos, "ilen": ilen, "allele": allele})

    def region_counts(
        self, contig: str, regions: Iterable[tuple[int, int]]
    ) -> "np.ndarray":
        """Decode-free per-``(region, sample, ploid)`` variant count, shape
        ``(R, S, P)``. The simplified ``SparseVar.var_ranges`` replacement."""
        reg = [(int(s), int(e)) for s, e in regions]
        flat = self._readers[contig].region_counts(reg)
        return flat.reshape(len(reg), self.n_samples, self.ploidy)
