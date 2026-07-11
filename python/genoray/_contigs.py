from __future__ import annotations

from collections.abc import Iterable
from typing import overload

import numpy as np
from hirola import HashTable
from numpy.typing import ArrayLike, NDArray

_MITO_ALIASES = ("M", "MT", "chrM", "chrMT")


class ContigNormalizer:
    """Normalizes contig name(s) to match alternative naming schemes. For example, "chr1" to "1" or "1" to "chr1".
    Mitochondrial aliases {M, MT, chrM, chrMT} are treated as mutually equivalent and
    resolve to whichever spelling the reference actually contains.
    """

    contigs: list[str]
    contig_map: dict[str, str]

    def __init__(self, contigs: Iterable[str]):
        self.contigs = list(contigs)
        mito = next((c for c in self.contigs if c in _MITO_ALIASES), None)
        mito_map = {a: mito for a in _MITO_ALIASES} if mito is not None else {}
        self.contig_map = (
            {f"{c[3:]}": c for c in contigs if c.startswith("chr")}
            | {f"chr{c}": c for c in contigs if not c.startswith("chr")}
            | {c: c for c in contigs}
            | mito_map
        )
        self.remapper = {k: self.contigs.index(c) for k, c in self.contig_map.items()}
        keys = np.array(list(self.remapper.keys()))
        self._c2dup = HashTable(
            max=len(self.contig_map) * 2,  # type: ignore
            dtype=keys.dtype,
        )
        self._c2dup.add(keys)
        self.dup2i = np.array(list(self.remapper.values()))

    @overload
    def norm(self, contigs: str) -> str | None: ...
    @overload
    def norm(self, contigs: list[str]) -> list[str | None]: ...
    def norm(self, contigs: str | list[str]) -> str | None | list[str | None]:
        """Normalize contig name(s) to match the naming scheme of the contig normalizer.

        Parameters
        ----------
        contigs
            Contig name(s) to normalize.
        """
        if isinstance(contigs, str):
            return self.contig_map.get(contigs, None)
        else:
            return [self.contig_map.get(c, None) for c in contigs]

    def c_idxs(self, contigs: ArrayLike) -> NDArray[np.integer]:
        """Map contig names to their indices in the contig normalizer, automatically mapping unnormalized contigs.

        Parameters
        ----------
        contigs
            Contig name(s) to map.
        """
        dup_idx = self._c2dup.get(contigs)
        return self.dup2i[dup_idx]
