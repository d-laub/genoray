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
