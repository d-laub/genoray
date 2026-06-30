from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeGuard, TypeVar, overload

import numpy as np
import polars as pl
from hirola import HashTable
from numpy.typing import ArrayLike, DTypeLike, NDArray

DTYPE = TypeVar("DTYPE", bound=np.generic)

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
    # (s p v)
    ilens = np.broadcast_to(ilens, genotypes.shape)  # zero-copy, read only
    # (s p v) -> (s p)
    return ilens.sum(-1, dtype=np.int32, where=genotypes == 1)


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


def _unique_sibling(dest: Path, suffix: str) -> Path:
    """Return a not-yet-existing sibling path of *dest* with *suffix* injected.

    Used to move an existing output dir aside before an atomic swap. The name is
    hidden (leading dot) and keyed by PID; a counter disambiguates collisions.
    """
    base = dest.with_name(f".{dest.name}{suffix}.{os.getpid()}")
    candidate = base
    i = 0
    while candidate.exists():
        i += 1
        candidate = dest.with_name(f"{base.name}.{i}")
    return candidate


@contextmanager
def atomic_write_path(dest: Path) -> Iterator[Path]:
    """Write a single file atomically: yield a sibling temp path to write to, then
    ``os.replace`` it onto *dest* on clean exit. On any exception the temp file is
    removed and *dest* is left untouched.

    The temp file is created in ``dest.parent`` so the final rename stays on one
    filesystem (a true atomic rename, no cross-device copy).
    """
    dest = Path(dest)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    else:
        os.replace(tmp, dest)


@contextmanager
def atomic_write_dir(dest: Path) -> Iterator[Path]:
    """Write a directory atomically: yield a sibling staging dir to populate, then
    swap it into place on clean exit.

    Swap is backup-then-swap: if *dest* already exists it is first moved aside to a
    fresh sibling name (so the staging dir is renamed onto a *non-existent* path,
    which is portable on POSIX and Windows), then the moved-aside dir is removed.
    If the final rename fails, the moved-aside dir is rolled back. The staging dir
    is always cleaned up; on an exception in the body *dest* is left untouched.

    The staging dir is created in ``dest.parent`` (same filesystem) so the swap is
    a true atomic rename and intermediate writes never cross devices.
    """
    dest = Path(dest)
    staging = Path(
        tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    )
    backup: Path | None = None
    try:
        yield staging
        if dest.exists():
            backup = _unique_sibling(dest, ".old")
            os.replace(dest, backup)
        os.replace(staging, dest)
    except BaseException:
        # If we moved dest aside but failed before/at the swap, roll it back.
        if backup is not None and backup.exists() and not dest.exists():
            os.replace(backup, dest)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
