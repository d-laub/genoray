from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeGuard

DTYPE = TypeVar("DTYPE", bound=np.generic)


class ContigNormalizer:
    contig_map: dict[str, str]

    def __init__(self, contigs: Iterable[str]):
        self.contig_map = (
            {f"{c[3:]}": c for c in contigs if c.startswith("chr")}
            | {f"chr{c}": c for c in contigs if not c.startswith("chr")}
            | {c: c for c in contigs}
        )

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


def is_dtype(obj: Any, dtype: type[DTYPE]) -> TypeGuard[NDArray[DTYPE]]:
    """Check if the object is a NumPy array with the given dtype.

    Parameters
    ----------
    obj
        Object to check.
    dtype
        Dtype to check against.

    Returns
    -------
    bool
        True if the object is an array with the given dtype, False otherwise.
    """
    return isinstance(obj, np.ndarray) and obj.dtype.type == dtype


_MEM_PARSER = re.compile(r"(?i)(\d+)(.*)")
_MEM_COEF = dict(zip(["", "k", "m", "g", "t", "p", "e"], 2 ** (np.arange(8) * 10)))
_MEM_COEF |= {f"{unit}ib": mem for unit, mem in _MEM_COEF.items() if unit != ""}
_MEM_COEF |= dict(
    zip(["kb", "mb", "gb", "tb", "pb", "eb"], 10 ** (3 * np.arange(1, 8)))
)


def parse_memory(memory: int | str) -> int:
    if isinstance(memory, int):
        return memory

    n = _MEM_PARSER.match(memory)
    if n is None:
        raise ValueError(f"Couldn't parse maximum memory '{memory}'")
    n, unit = n.groups()
    unit = unit.strip()
    mem_i = int(n)
    coef = _MEM_COEF.get(unit.lower(), None)

    if coef is None:
        raise ValueError(f"Unrecognized memory unit '{unit}'.")

    return mem_i * coef.item()


def format_memory(memory: int):
    """Format an integer as a human-readable memory size string."""
    if memory < 1024:
        return f"{memory} B"

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
    exponent = min(int(math.log2(memory) // 10), len(units) - 1)
    value = memory / (1 << (10 * exponent))
    return f"{value:.2f} {units[exponent]}"


def hap_ilens(
    genotypes: NDArray[np.integer], ilens: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Get the indel lengths of haplotypes from genotypes i.e. the difference in their lengths compared
    to the reference sequence. Assumes phased genotypes.

    Parameters
    ----------
    genotypes
        Genotypes array. Shape: (samples, ploidy, variants).
    ilens
        Lengths of the segments. Shape: (variants).

    Returns
    -------
    hap_lengths
        Lengths of the haplotypes. Shape: (samples, ploidy).
    """
    ilens = np.broadcast_to(ilens, genotypes.shape)  # zero-copy, read only
    return ilens.sum(-1, dtype=np.int32, where=genotypes == 1)


_VCF_EXT = re.compile(r"\.[vb]cf(\.gz)?$")
_PGEN_EXT = re.compile(r"\.(pgen|pvar|psam)$")


def variant_file_type(path: str | Path):
    path = Path(path)
    if _VCF_EXT.match(path.name) is not None:
        return "vcf"
    elif _PGEN_EXT.match(path.name) is not None or (
        path.with_suffix(".pgen").exists()
        and path.with_suffix(".pvar").exists()
        and path.with_suffix(".psam").exists()
    ):
        return "pgen"
