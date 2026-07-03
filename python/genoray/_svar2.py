from __future__ import annotations

import json
from pathlib import Path

from genoray import _core
from genoray._svar2_batch import _BatchQueryMixin
from genoray._svar2_decode import _DecodeMixin


class SparseVar2(_BatchQueryMixin, _DecodeMixin):
    """Reader for a finished SVAR2 store (M6a skeleton).

    Loads the top-level ``meta.json`` and opens one native
    :class:`genoray._core.PyContigReader` per contig. Query methods land in M6b
    (raw two-channel result) and M6c (decoded ``seqpro.rag.Ragged``).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        self.format_version: int = meta["format_version"]
        self.samples: list[str] = list(meta["samples"])
        self.contigs: list[str] = list(meta["contigs"])
        self.ploidy: int = meta["ploidy"]
        self._readers: dict[str, _core.PyContigReader] = {
            contig: _core.PyContigReader(
                str(self.path), contig, len(self.samples), self.ploidy
            )
            for contig in self.contigs
        }

    @property
    def n_samples(self) -> int:
        return len(self.samples)
