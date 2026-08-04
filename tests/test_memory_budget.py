"""detect_memory_budget must resolve THIS PROCESS's own cgroup limit, not
just any path on the host.

`/sys/fs/cgroup/memory.max` (v2) and `/sys/fs/cgroup/memory/memory.limit_in_bytes`
(v1) are the ROOT cgroup's own files -- unlimited on any host without a
cgroup namespace, e.g. a bare Slurm compute node, where the real per-job
limit lives several levels down at `/slurm/uid_<uid>/job_<id>/...`. The fix
resolves this process's actual cgroup path from `/proc/self/cgroup` first
(walking ancestors for v2, since an ancestor can be more restrictive than the
leaf), and only falls back to the fixed root paths -- then `/proc/meminfo` --
when self-discovery finds nothing.

Under Slurm the cgroup and `/proc/meminfo` differ: `/proc/meminfo` reports
the node, the cgroup reports the job. Planning against the node OOMs the job.
"""

from pathlib import Path

import pytest

from genoray._utils import MEM_BUDGET_FRACTION, detect_memory_budget


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _block_self_discovery(monkeypatch, tmp_path: Path) -> None:
    """Point every self-discovery input at a path that can never resolve, so
    only the legacy fixed roots / meminfo (whatever the test itself sets up
    afterward) are reachable."""
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(
        "genoray._utils._PROC_SELF_CGROUP", missing / "proc_self_cgroup"
    )
    monkeypatch.setattr("genoray._utils._CGROUP_V2_ROOT", missing / "cgroup_v2_root")
    monkeypatch.setattr("genoray._utils._CGROUP_V1_ROOT", missing / "cgroup_v1_root")


def _block_legacy_and_meminfo(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr("genoray._utils._CGROUP_V2", missing / "cgroup_v2")
    monkeypatch.setattr("genoray._utils._CGROUP_V1", missing / "cgroup_v1")
    monkeypatch.setattr("genoray._utils._MEMINFO", missing / "meminfo")


def test_v2_self_discovered_leaf_limit(tmp_path, monkeypatch):
    """The common case: this process's own (leaf) cgroup carries the limit."""
    v2_root = tmp_path / "sys-fs-cgroup"
    _write(v2_root / "slurm" / "job" / "memory.max", "8589934592\n")  # 8 GiB
    cgroup_file = _write(tmp_path / "proc_self_cgroup", "0::/slurm/job\n")

    monkeypatch.setattr("genoray._utils._PROC_SELF_CGROUP", cgroup_file)
    monkeypatch.setattr("genoray._utils._CGROUP_V2_ROOT", v2_root)
    _block_legacy_and_meminfo(monkeypatch, tmp_path)

    assert detect_memory_budget() == int(8589934592 * MEM_BUDGET_FRACTION)


def test_v2_ancestor_more_restrictive_than_leaf_wins(tmp_path, monkeypatch):
    """An ancestor cgroup can cap tighter than the leaf; the effective limit
    is the minimum across the whole chain, not just the leaf's own file."""
    v2_root = tmp_path / "sys-fs-cgroup"
    _write(v2_root / "memory.max", "4294967296\n")  # 4 GiB at the ROOT -- tighter
    _write(v2_root / "slurm" / "memory.max", "max\n")
    _write(v2_root / "slurm" / "job" / "memory.max", "8589934592\n")  # 8 GiB leaf
    cgroup_file = _write(tmp_path / "proc_self_cgroup", "0::/slurm/job\n")

    monkeypatch.setattr("genoray._utils._PROC_SELF_CGROUP", cgroup_file)
    monkeypatch.setattr("genoray._utils._CGROUP_V2_ROOT", v2_root)
    _block_legacy_and_meminfo(monkeypatch, tmp_path)

    assert detect_memory_budget() == int(4294967296 * MEM_BUDGET_FRACTION)


def test_v1_self_discovered_real_path(tmp_path, monkeypatch):
    """Mirrors this exact cluster: cgroup v1, job cgroup several levels below
    the mount root (`/slurm/uid_<uid>/job_<id>`)."""
    v1_root = tmp_path / "sys-fs-cgroup-memory"
    _write(
        v1_root / "slurm" / "uid_1111" / "job_13336789" / "memory.limit_in_bytes",
        "68719476736\n",  # 64 GiB
    )
    cgroup_file = _write(
        tmp_path / "proc_self_cgroup", "11:memory:/slurm/uid_1111/job_13336789\n"
    )

    monkeypatch.setattr("genoray._utils._PROC_SELF_CGROUP", cgroup_file)
    monkeypatch.setattr("genoray._utils._CGROUP_V1_ROOT", v1_root)
    _block_legacy_and_meminfo(monkeypatch, tmp_path)

    assert detect_memory_budget() == int(68719476736 * MEM_BUDGET_FRACTION)


def test_v1_root_sentinel_falls_through_to_meminfo(tmp_path, monkeypatch):
    """cgroup v1 writes a huge sentinel (PAGE_COUNTER_MAX) when uncapped; it
    is not a real limit and must not be believed, even when self-discovered."""
    v1_root = tmp_path / "sys-fs-cgroup-memory"
    _write(v1_root / "memory.limit_in_bytes", "9223372036854771712\n")
    cgroup_file = _write(tmp_path / "proc_self_cgroup", "11:memory:/\n")
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       1048576 kB\n")  # 1 GiB

    monkeypatch.setattr("genoray._utils._PROC_SELF_CGROUP", cgroup_file)
    monkeypatch.setattr("genoray._utils._CGROUP_V1_ROOT", v1_root)
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    monkeypatch.setattr(
        "genoray._utils._CGROUP_V2", tmp_path / "does-not-exist" / "cgroup_v2"
    )
    monkeypatch.setattr(
        "genoray._utils._CGROUP_V1", tmp_path / "does-not-exist" / "cgroup_v1"
    )

    assert detect_memory_budget() == int(1024**3 * MEM_BUDGET_FRACTION)


def test_no_self_discovery_falls_back_to_legacy_root_paths(tmp_path, monkeypatch):
    """When `/proc/self/cgroup` itself can't be read (no /proc, e.g. a
    container without it mounted), the fixed root-cgroup paths are still
    tried before giving up -- they only see the right number when this
    process happens to sit at the root of the hierarchy, but that's still
    better than nothing. This is the same scenario the pre-fix tests pinned
    (a fixed leaf path standing in for "the" cgroup limit)."""
    _block_self_discovery(monkeypatch, tmp_path)
    v2 = _write(tmp_path / "memory.max", "2147483648\n")  # 2 GiB
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr(
        "genoray._utils._CGROUP_V1", tmp_path / "does-not-exist" / "cgroup_v1"
    )
    monkeypatch.setattr(
        "genoray._utils._MEMINFO", tmp_path / "does-not-exist" / "meminfo"
    )

    assert detect_memory_budget() == int(2147483648 * MEM_BUDGET_FRACTION)


def test_all_sources_absent_raises(tmp_path, monkeypatch):
    _block_self_discovery(monkeypatch, tmp_path)
    _block_legacy_and_meminfo(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="could not detect a memory budget"):
        detect_memory_budget()


def test_fraction_is_applied_and_never_the_whole_limit(tmp_path, monkeypatch):
    v2_root = tmp_path / "sys-fs-cgroup"
    _write(v2_root / "memory.max", "1000000000\n")
    cgroup_file = _write(tmp_path / "proc_self_cgroup", "0::/\n")

    monkeypatch.setattr("genoray._utils._PROC_SELF_CGROUP", cgroup_file)
    monkeypatch.setattr("genoray._utils._CGROUP_V2_ROOT", v2_root)
    _block_legacy_and_meminfo(monkeypatch, tmp_path)

    got = detect_memory_budget()
    assert got == 800_000_000
    assert got < 1_000_000_000


def test_zero_byte_cgroup_limit_is_a_real_limit_not_absent(tmp_path, monkeypatch):
    """A cgroup file containing the literal digit "0" is a valid (if useless)
    limit, not "not found". `_read_int` returns the int 0, which is falsy in
    Python but must still be distinguished from None -- a `if limit:` bug
    instead of `if limit is None:` would silently skip straight past it to
    the next tier."""
    v2_root = tmp_path / "sys-fs-cgroup"
    _write(v2_root / "job" / "memory.max", "0\n")
    cgroup_file = _write(tmp_path / "proc_self_cgroup", "0::/job\n")

    monkeypatch.setattr("genoray._utils._PROC_SELF_CGROUP", cgroup_file)
    monkeypatch.setattr("genoray._utils._CGROUP_V2_ROOT", v2_root)
    _block_legacy_and_meminfo(monkeypatch, tmp_path)

    assert detect_memory_budget() == 0
