import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "svar2_region_parallel_bench",
    Path(__file__).resolve().parents[2] / "scripts" / "svar2_region_parallel_bench.py",
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_compute_scaling_adds_speedup_and_efficiency():
    rows = [
        {"backend": "vcf", "chunk_size": 25000, "threads": 1, "wall_s": 100.0},
        {"backend": "vcf", "chunk_size": 25000, "threads": 4, "wall_s": 40.0},
    ]
    out = {(r["threads"]): r for r in bench.compute_scaling(rows)}
    assert out[1]["speedup"] == 1.0
    assert out[1]["efficiency"] == 1.0
    assert out[4]["speedup"] == 2.5
    assert abs(out[4]["efficiency"] - 0.625) < 1e-9


def _make_store(root: Path, files: dict[str, bytes]) -> Path:
    for relpath, data in files.items():
        p = root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def test_oracle_hash_identical_layout_hashes_equal(tmp_path):
    files = {"meta.json": b'{"a":1}', "sub/data.bin": b"\x00\x01\x02"}
    a = _make_store(tmp_path / "a", files)
    b = _make_store(tmp_path / "b", files)
    assert bench.oracle_hash(a) == bench.oracle_hash(b)


def test_oracle_hash_differing_bytes_hash_differ(tmp_path):
    a = _make_store(tmp_path / "a", {"meta.json": b'{"a":1}', "d.bin": b"\x00\x01"})
    b = _make_store(tmp_path / "b", {"meta.json": b'{"a":1}', "d.bin": b"\x00\x02"})
    assert bench.oracle_hash(a) != bench.oracle_hash(b)


def test_oracle_hash_keys_on_relative_path(tmp_path):
    # Same content nested under different parent dirs must hash equal, since the
    # hash keys on the path RELATIVE to the store root, not the absolute path.
    files = {"meta.json": b"x", "sub/data.bin": b"y"}
    a = _make_store(tmp_path / "deep/nested/a", files)
    b = _make_store(tmp_path / "b", files)
    assert bench.oracle_hash(a) == bench.oracle_hash(b)
