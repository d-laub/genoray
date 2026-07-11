from __future__ import annotations

import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import DTypeLike

from ._types import DTYPE as DTYPE

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


_VCF_EXT = re.compile(r"\.[vb]cf(\.gz)?$")
_PGEN_EXT = re.compile(r"\.(pgen|pvar|psam)$")


def variant_file_type(path: str | Path) -> Literal["vcf", "pgen"] | None:
    path = Path(path)
    if _VCF_EXT.search(path.name) is not None:
        return "vcf"
    elif _PGEN_EXT.search(path.name) is not None or (
        path.with_suffix(".pgen").exists()
        and path.with_suffix(".pvar").exists()
        and path.with_suffix(".psam").exists()
    ):
        return "pgen"
    return None


_NP_TO_PL: dict[type[np.generic], type[pl.DataType]] = {
    np.float16: pl.Float16,
    np.float32: pl.Float32,
    np.float64: pl.Float64,
    np.int8: pl.Int8,
    np.int16: pl.Int16,
    np.int32: pl.Int32,
    np.int64: pl.Int64,
    np.uint8: pl.UInt8,
    np.uint16: pl.UInt16,
    np.uint32: pl.UInt32,
    np.uint64: pl.UInt64,
    np.datetime64: pl.Datetime,
    np.timedelta64: pl.Duration,
    np.str_: pl.Utf8,
    np.bytes_: pl.Binary,
    np.bool_: pl.Boolean,
    np.object_: pl.Object,
}


def np_to_pl_dtype(dtype: DTypeLike) -> type[pl.DataType]:
    key = np.dtype(dtype).type
    try:
        return _NP_TO_PL[key]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_threads(threads: int | None) -> int:
    """Resolve the effective number of threads.

    - If `threads` is given, return it as-is.
    - Else prefer `os.sched_getaffinity(0)` (Linux), else `os.cpu_count()`, else 1.
    """
    if threads is not None:
        return threads
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


@contextmanager
def numba_threads(n: int):
    """Temporarily set the numba thread count, restoring the previous value on exit."""
    import numba

    prev = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield
    finally:
        numba.set_num_threads(prev)
