"""detect_memory_budget must prefer the cgroup limit over /proc/meminfo.

Under Slurm those differ: /proc/meminfo reports the node, the cgroup reports
the job. Planning against the node OOMs the job.
"""

import pytest

from genoray._utils import MEM_BUDGET_FRACTION, detect_memory_budget


def _write(path, text):
    path.write_text(text)
    return path


def test_prefers_cgroup_v2_over_meminfo(tmp_path, monkeypatch):
    v2 = _write(tmp_path / "memory.max", "8589934592\n")  # 8 GiB
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       263192360 kB\n")
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr("genoray._utils._CGROUP_V1", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    assert detect_memory_budget() == int(8589934592 * MEM_BUDGET_FRACTION)


def test_cgroup_v2_max_falls_through_to_v1(tmp_path, monkeypatch):
    # cgroup v2 writes the literal "max" when the group is uncapped.
    v2 = _write(tmp_path / "memory.max", "max\n")
    v1 = _write(tmp_path / "limit_in_bytes", "4294967296\n")  # 4 GiB
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       263192360 kB\n")
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr("genoray._utils._CGROUP_V1", v1)
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    assert detect_memory_budget() == int(4294967296 * MEM_BUDGET_FRACTION)


def test_cgroup_v1_sentinel_falls_through_to_meminfo(tmp_path, monkeypatch):
    # cgroup v1 writes a huge sentinel (PAGE_COUNTER_MAX) when uncapped; it is
    # not a real limit and must not be believed.
    v1 = _write(tmp_path / "limit_in_bytes", "9223372036854771712\n")
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       1048576 kB\n")  # 1 GiB
    monkeypatch.setattr("genoray._utils._CGROUP_V2", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._CGROUP_V1", v1)
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    assert detect_memory_budget() == int(1024**3 * MEM_BUDGET_FRACTION)


def test_all_sources_absent_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("genoray._utils._CGROUP_V2", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._CGROUP_V1", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._MEMINFO", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="could not detect a memory budget"):
        detect_memory_budget()


def test_fraction_is_applied_and_never_the_whole_limit(tmp_path, monkeypatch):
    v2 = _write(tmp_path / "memory.max", "1000000000\n")
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr("genoray._utils._CGROUP_V1", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._MEMINFO", tmp_path / "absent")
    got = detect_memory_budget()
    assert got == 800_000_000
    assert got < 1_000_000_000
