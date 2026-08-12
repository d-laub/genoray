from scripts.bench_svar2.fit_ram import gate_report
from scripts.bench_svar2.model import RamRow, fit_ram_law


def test_gate_report_flags_an_underpredicted_point():
    # A law is an UPPER BOUND: the gate is over-prediction at EVERY point,
    # evaluated the way plan_sharded evaluates it. One point below the line
    # fails the whole law, however good the fit looks elsewhere.
    rows = [
        RamRow(
            workers=1,
            pending=0,
            chunk_bytes=10_000_000,
            samples=4_000,
            peak_rss_mb=1_000.0,
            concurrent_chroms=1,
        ),
        RamRow(
            workers=1,
            pending=0,
            chunk_bytes=10_000_000,
            samples=4_000,
            peak_rss_mb=1_000_000.0,
            concurrent_chroms=1,
        ),
    ]
    law = fit_ram_law(rows[:1], margin=1.0)
    report = gate_report(law, rows)
    assert not report["passes"]
    assert report["n_under"] == 1


def test_gate_report_passes_a_true_envelope():
    rows = [
        RamRow(
            workers=1,
            pending=0,
            chunk_bytes=10_000_000,
            samples=4_000,
            peak_rss_mb=1_000.0,
            concurrent_chroms=1,
        ),
        RamRow(
            workers=1,
            pending=0,
            chunk_bytes=20_000_000,
            samples=32_000,
            peak_rss_mb=2_000.0,
            concurrent_chroms=4,
        ),
    ]
    law = fit_ram_law(rows, margin=1.25)
    report = gate_report(law, rows)
    assert report["passes"], report
    assert report["n_under"] == 0
    assert report["worst_ratio"] >= report["min_ratio"] >= 1.25 - 1e-9
