"""M6c: decoded ``seqpro.rag.Ragged`` + region-count methods for ``SparseVar2``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from seqpro.rag import Ragged

    from genoray import _core


class _DecodeMixin:
    """Decoded-record and decode-free-count query methods."""

    # Provided by the concrete SparseVar2 host class (see SparseVar2.__init__);
    # declared here so the mixin's use of it type-checks in isolation. (n_samples
    # and ploidy are @property on the host, so they can't be redeclared as plain
    # attributes here without a bad-override; they're accessed via an ignore below.)
    _readers: dict[str, Any]
    _fields: list[Any]
    path: Any

    def _reader(self, contig: str) -> "_core.PyContigReader":  # host-provided
        ...

    def decode(self, contig: str, regions: Iterable[tuple[int, int]]) -> "Ragged":
        """Materialize overlapping variants for ``contig`` into a record ``Ragged``.

        Fields ``pos`` (i32), ``ilen`` (i32), ``allele`` (opaque-string ALT bytes),
        plus one entry per selected INFO/FORMAT field (see
        :meth:`SparseVar2.with_fields`) — every one sharing a single variant-axis
        offsets object, shape ``(R, S, P, None)``, the same layout as gvl's
        ``RaggedVariants``. Pure-deletion ALT is empty.

        Field values come back in the dtype they are STORED as (SVAR2
        losslessly auto-narrows integers), and VCF-missing entries carry the
        store's ``default`` or its reserved sentinel (``NaN`` for floats,
        ``iinfo.min``/``iinfo.max`` for ints) — neither is translated.
        """
        from seqpro.rag import Ragged

        from genoray._svar2_fields import _META_DTYPE

        reg = [(int(s), int(e)) for s, e in regions]
        reader = self._reader(contig)
        if not self._fields:
            d = reader.decode_batch(reg)
        else:
            d = reader.decode_batch_fields(
                reg,
                [(f.category, f.name, _META_DTYPE[f.dtype]) for f in self._fields],
                str(self.path),
                contig,
            )

        shape = (d["n_regions"], d["n_samples"], d["ploidy"], None)
        off = d["off"]
        pos = Ragged.from_offsets(d["pos"], shape, off)
        ilen = Ragged.from_offsets(d["ilen"], shape, off)
        allele = Ragged.from_offsets(
            d["allele"].view("S1"), shape, off, str_offsets=d["str_off"]
        )
        # If a consumer hits an error reading `.lengths` on a (2, N) offsets
        # layout, call `.to_packed()` first — a known seqpro slicing quirk.
        rec: dict[str, Ragged] = {"pos": pos, "ilen": ilen, "allele": allele}
        for f in self._fields:
            # Rust hands back raw little-endian bytes + an itemsize; the dtype is
            # applied here (same trick as `allele`), so Rust does no dtype
            # dispatch. `.view` is zero-copy.
            raw: np.ndarray = d[f"field_{f.category}/{f.name}"]
            itemsize = d[f"field_itemsize_{f.category}/{f.name}"]
            if itemsize != f.dtype.itemsize:
                # A width/dtype disagreement would silently reinterpret the
                # bytes into wrong values — fail loudly instead.
                raise ValueError(
                    f"field {f.key!r}: store wrote {itemsize}-byte elements but "
                    f"meta.json declares {f.dtype} ({f.dtype.itemsize} bytes)"
                )
            vals = raw.view(f.dtype)
            rec[f.key] = Ragged.from_offsets(vals, shape, off)
        return Ragged.from_fields(rec)

    def region_counts(
        self, contig: str, regions: Iterable[tuple[int, int]]
    ) -> "np.ndarray":
        """Decode-free per-``(region, sample, ploid)`` variant count, shape ``(R, S, P)``.

        The simplified ``SparseVar.var_ranges`` replacement.
        """
        reg = [(int(s), int(e)) for s, e in regions]
        flat = self._reader(contig).region_counts(reg)
        # n_samples/ploidy are @property on the SparseVar2 host (see note above).
        return flat.reshape(len(reg), self.n_samples, self.ploidy)  # type: ignore[missing-attribute]
