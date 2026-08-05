"""Sample-name selection, deliberately free of heavy imports.

`_normalize_samples` is pure Python, but it used to live in
`genoray._svar._regions`, whose module scope imports seqpro (and through it
numba and llvmlite) and hirola -- and importing that module also executes
`genoray._svar`'s package ``__init__``, i.e. the whole SVAR1 stack.
`SparseVar2.from_vcf` imports it on every call, so a ~2.2s import sat on the
critical path of every conversion, constant across cohort width and file size.

Keep this module dependency-free: only the standard library, and only names
needed to select sample columns. `genoray._svar._regions` re-exports
`_normalize_samples` so existing importers keep working.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path


def _normalize_samples(
    samples: "str | Sequence[str] | PathLike",
    available: Sequence[str],
) -> list[str]:
    """Normalize `samples` to a list of valid sample names, preserving caller order and deduping by first occurrence.

    Raises ValueError on unknown samples.
    """
    if isinstance(samples, str):
        candidates: list[str] = [samples]
    elif isinstance(samples, PathLike):
        candidates = Path(samples).read_text().splitlines()
        candidates = [s for s in candidates if s.strip()]
    else:
        candidates = list(samples)

    avail_set = set(available)
    missing = [s for s in candidates if s not in avail_set]
    if missing:
        raise ValueError(f"Samples not found in dataset: {missing}")

    seen: set[str] = set()
    deduped: list[str] = []
    for s in candidates:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped
