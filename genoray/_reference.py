from __future__ import annotations

from pathlib import Path

import numpy as np
import pysam
from numpy.typing import NDArray

from ._utils import ContigNormalizer

_PAD = ord("N")


class Reference:
    """A reference genome backed by an indexed FASTA, read on demand via pysam.

    One contig is held in memory at a time and sliced for flanking-base lookups.
    Queries accept ``chr``-prefixed or unprefixed contig names interchangeably.

    Do not instantiate directly; use :meth:`Reference.from_path`.
    """

    def __init__(self, path: Path, contigs: list[str]):
        self.path = path
        self._fasta = pysam.FastaFile(str(path))
        # pysam reports the FASTA's own contig names; build a normalizer from them.
        self._c_norm = ContigNormalizer(list(self._fasta.references))
        self.contigs = contigs
        self._cur_contig: str | None = None
        self._cur_seq: NDArray[np.uint8] | None = None

    def __del__(self):
        if hasattr(self, "_fasta"):
            self._fasta.close()

    @classmethod
    def from_path(
        cls, fasta: str | Path, contigs: list[str] | None = None
    ) -> "Reference":
        path = Path(fasta)
        if not path.exists():
            raise FileNotFoundError(f"FASTA {path} does not exist.")
        fai = path.with_suffix(path.suffix + ".fai")
        if not fai.exists():
            pysam.faidx(str(path))
        f = pysam.FastaFile(str(path))
        all_contigs = list(f.references)
        f.close()
        return cls(path, contigs if contigs is not None else all_contigs)

    def _load_contig(self, contig: str) -> NDArray[np.uint8]:
        norm = self._c_norm.norm(contig)
        if norm is None:
            raise ValueError(f"Contig {contig!r} not found in reference {self.path}.")
        if norm != self._cur_contig:
            seq = self._fasta.fetch(norm)  # whole contig as str
            self._cur_seq = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            self._cur_contig = norm
        assert self._cur_seq is not None
        return self._cur_seq

    def fetch(self, contig: str, start: int, end: int) -> NDArray[np.uint8]:
        """Return reference bytes for 0-based half-open ``[start, end)``.

        Positions outside the contig are padded with ``N``. Returns a uint8
        array; ``bytes(...)`` gives the ASCII sequence.
        """
        seq = self._load_contig(contig)
        n = len(seq)
        out = np.full(end - start, _PAD, dtype=np.uint8)
        src_s = max(start, 0)
        src_e = min(end, n)
        if src_e > src_s:
            out[src_s - start : src_e - start] = seq[src_s:src_e]
        return out
