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


# Planning to 100% of a limit means the first prediction error is an OOM kill.
# The RAM law it feeds predicts peak RSS with R^2=0.9040 and a 3% hold-out
# error; this headroom covers that residual plus everything the law does not
# model -- the interpreter, glibc arena fragmentation, the merge tail.
MEM_BUDGET_FRACTION = 0.8

# Module-level so tests can point them at fixtures.
#
# `/proc/self/cgroup` tells us which cgroup THIS process actually lives in;
# the roots below are where that relative path gets resolved. Under Slurm (or
# any cgroup manager) the process sits several levels below the root, e.g.
# `/slurm/uid_1111/job_13336789` -- reading `memory.max`/`memory.limit_in_bytes`
# straight off the root only sees the right number when this process happens
# to run un-namespaced at the top of the hierarchy.
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
_CGROUP_V1_ROOT = Path("/sys/fs/cgroup/memory")
# Last-resort fixed paths: the ROOT cgroup's own limit. Used only when this
# process's own cgroup can't be resolved from `_PROC_SELF_CGROUP` (missing
# /proc, unreadable, or no matching hierarchy line) -- e.g. no cgroups at
# all, or a namespaced container where "root" already IS this process's
# cgroup. Kept as fixed module-level paths (not derived) so a host with no
# `/proc/self/cgroup` still gets a best-effort answer instead of none.
_CGROUP_V2 = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_MEMINFO = Path("/proc/meminfo")

# cgroup v1 writes a near-INT64_MAX sentinel rather than a real limit when the
# group is uncapped. Anything at or above this is "no limit", not a budget.
_CGROUP_V1_UNLIMITED = 1 << 62


def _read_int(path: Path) -> int | None:
    """Parse a single integer from `path`, or None if absent/unparseable."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None  # cgroup v2 writes the literal "max" when uncapped.


def _meminfo_total(path: Path) -> int | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _own_cgroup_path(cgroup_file: Path, *, v2: bool) -> str | None:
    """This process's path within one cgroup hierarchy.

    Read from `cgroup_file` (normally `/proc/self/cgroup`); None if
    unreadable or not found. Each line is
    `<hierarchy-id>:<controller-list>:<path>`. v2's unified
    hierarchy always has `hierarchy-id == 0` and an EMPTY controller list
    (`0::<path>`); v1's memory controller has a comma-separated controller
    list that may name `memory` alongside others (e.g. `11:memory:<path>` or
    `9:cpu,memory:<path>`).
    """
    try:
        text = cgroup_file.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _hier_id, controllers, path = parts
        if v2:
            if controllers == "":
                return path
        elif "memory" in controllers.split(","):
            return path
    return None


def _cgroup_v2_limit(cgroup_file: Path, root: Path) -> int | None:
    """This process's effective cgroup v2 memory limit.

    The MINIMUM of `memory.max` at its own cgroup and every ancestor up to
    `root` (an ancestor can be more restrictive than the leaf; v2's unified
    hierarchy has no separate "which controller won" question the way v1's
    nesting does, so a plain min is correct here).
    """
    rel = _own_cgroup_path(cgroup_file, v2=True)
    if rel is None:
        return None
    parts = [p for p in rel.split("/") if p]
    limits = []
    for depth in range(len(parts), -1, -1):
        limit = _read_int(root.joinpath(*parts[:depth]) / "memory.max")
        if limit is not None:
            limits.append(limit)
    return min(limits) if limits else None


def _cgroup_v1_limit(cgroup_file: Path, root: Path) -> int | None:
    """This process's cgroup v1 `memory.limit_in_bytes`.

    Rejects the uncapped sentinel; returns None if unresolvable.
    """
    rel = _own_cgroup_path(cgroup_file, v2=False)
    if rel is None:
        return None
    parts = [p for p in rel.split("/") if p]
    limit = _read_int(root.joinpath(*parts) / "memory.limit_in_bytes")
    if limit is not None and limit < _CGROUP_V1_UNLIMITED:
        return limit
    return None


def detect_memory_budget(fraction: float = MEM_BUDGET_FRACTION) -> int:
    """Bytes of memory the conversion planner may plan against.

    Prefers the cgroup limit over `/proc/meminfo`. Under Slurm the two differ:
    `/proc/meminfo` reports the node, the cgroup reports the job. Planning
    against the node hands the planner a budget it does not have, on exactly
    the allocations where the planner matters most.

    Resolves THIS PROCESS's own cgroup from `/proc/self/cgroup` first (v2,
    walking ancestors for the tightest limit; then v1), falling back to the
    fixed root-cgroup paths only when self-discovery finds nothing --
    reading the root directly is unlimited on any host without a cgroup
    namespace (e.g. a bare Slurm compute node, where the real job limit
    lives several levels down at `/slurm/uid_<uid>/job_<id>/...`).
    """
    limit = _cgroup_v2_limit(_PROC_SELF_CGROUP, _CGROUP_V2_ROOT)
    if limit is None:
        limit = _read_int(_CGROUP_V2)
    if limit is None:
        limit = _cgroup_v1_limit(_PROC_SELF_CGROUP, _CGROUP_V1_ROOT)
    if limit is None:
        v1 = _read_int(_CGROUP_V1)
        limit = v1 if v1 is not None and v1 < _CGROUP_V1_UNLIMITED else None
    if limit is None:
        limit = _meminfo_total(_MEMINFO)
    if limit is None:
        raise RuntimeError(
            "could not detect a memory budget (no cgroup limit and no "
            "/proc/meminfo); pass max_mem explicitly"
        )
    return int(limit * fraction)
