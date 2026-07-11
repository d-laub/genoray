from __future__ import annotations

import math
import os
import re
from contextlib import contextmanager
from pathlib import Path

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


def variant_file_type(path: str | Path):
    path = Path(path)
    if _VCF_EXT.search(path.name) is not None:
        return "vcf"
    elif _PGEN_EXT.search(path.name) is not None or (
        path.with_suffix(".pgen").exists()
        and path.with_suffix(".pvar").exists()
        and path.with_suffix(".psam").exists()
    ):
        return "pgen"


def np_to_pl_dtype(dtype: DTypeLike) -> type[pl.DataType]:
    dtype = np.dtype(dtype)

    if dtype == np.float16:
        return pl.Float16
    elif dtype == np.float32:
        return pl.Float32
    elif dtype == np.float64:
        return pl.Float64

    elif dtype == np.int8:
        return pl.Int8
    elif dtype == np.int16:
        return pl.Int16
    elif dtype == np.int32:
        return pl.Int32
    elif dtype == np.int64:
        return pl.Int64

    elif dtype == np.uint8:
        return pl.UInt8
    elif dtype == np.uint16:
        return pl.UInt16
    elif dtype == np.uint32:
        return pl.UInt32
    elif dtype == np.uint64:
        return pl.UInt64

    elif dtype == np.datetime64:
        return pl.Datetime
    elif dtype == np.timedelta64:
        return pl.Duration

    elif dtype == np.str_:
        return pl.Utf8
    elif dtype == np.bytes_:
        return pl.Binary

    elif dtype == np.bool_:
        return pl.Boolean
    elif dtype == np.object_:
        return pl.Object

    else:
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
