from __future__ import annotations

import re
from typing import Iterable, TypeVar, overload

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
        """Normalize the contig name to match the naming scheme of `contigs`.

        Parameters
        ----------
        contigs
            Contig name(s) to normalize.
        """
        if isinstance(contigs, str):
            return self.contig_map.get(contigs, None)
        else:
            return [self.contig_map.get(c, None) for c in contigs]


def is_dtype(array: NDArray, dtype: type[DTYPE]) -> TypeGuard[NDArray[DTYPE]]:
    """Check if the array has the given dtype.

    Parameters
    ----------
    array
        Array to check.
    dtype
        Dtype to check against.

    Returns
    -------
    bool
        True if the array has the given dtype, False otherwise.
    """
    return array.dtype.type == dtype


_MEM_PARSER = re.compile(r"([0-9]+?)(.*)")


def parse_memory(memory: int | str) -> int:
    if isinstance(memory, int):
        return memory

    n = _MEM_PARSER.match(memory)
    if n is None:
        raise ValueError(f"Couldn't parse maximum memory '{memory}'")
    n, unit = n.groups()
    mem_i = int(n)
    if unit in {"T", "TB"}:
        mem_i *= 2**40
    elif unit in {"G", "GB"}:
        mem_i *= 2**30
    elif unit in {"M", "MB"}:
        mem_i *= 2**20
    elif unit in {"K", "KB"}:
        mem_i *= 2**10
    elif unit == "":
        pass
    else:
        raise ValueError(f"Unknown memory unit '{unit}'. Use T, G, M, K or nothing.")

    return mem_i
