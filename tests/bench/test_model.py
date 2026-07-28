import math

import pytest

from scripts.bench_svar2.model import (
    decide,
    extrapolate,
    fit_cost_law,
    fit_ram_law,
    fit_v_law,
    knee_from_probe,
)

pytestmark = pytest.mark.bench


def test_v_law_recovers_planted_line():
    variants = [25_000, 50_000, 100_000, 200_000]
    walls = [1.0 + 1e-4 * v for v in variants]
    law = fit_v_law(list(zip(variants, walls)))
    assert law.r2 > 0.999
    assert math.isclose(law.slope_s_per_variant, 1e-4, rel_tol=1e-6)
    assert math.isclose(law.intercept_s, 1.0, abs_tol=1e-6)


def test_v_law_reports_low_r2_on_nonlinear_data():
    """If V-linearity fails, every downstream extrapolation is invalid and the
    harness must be able to say so rather than report a number."""
    variants = [25_000, 50_000, 100_000, 200_000]
    walls = [1e-8 * v**2.0 for v in variants]
    assert fit_v_law(list(zip(variants, walls))).r2 < 0.98


def test_cost_law_recovers_planted_exponent():
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    costs = [3.5 * s**0.8 for s in samples]
    law = fit_cost_law("read", samples, costs)
    assert math.isclose(law.beta, 0.8, rel_tol=1e-6)
    assert law.beta_ci95[0] < 0.8 < law.beta_ci95[1]


def test_knee_is_ratio_of_read_to_exec_cost():
    # 360% shard CPU against 60% exec CPU is a 6:1 cost ratio.
    assert knee_from_probe((360.0, 360.0), (60.0, 60.0)) == 6


def test_knee_floors_at_one():
    assert knee_from_probe((10.0,), (100.0,)) == 1


def test_ram_law_recovers_planted_slope():
    rows = [
        # (workers, pending_highwater, chunk_bytes, peak_rss_mb)
        (w, 0, 25_000_000, 100.0 + 3.0 * w * 25_000_000 / 1e6)
        for w in (1, 3, 5, 7, 11)
    ]
    law = fit_ram_law(rows)
    assert math.isclose(law.kappa, 3.0, rel_tol=1e-6)
    assert math.isclose(law.base_mb, 100.0, abs_tol=1e-6)


def test_decide_picks_h1_when_knee_is_flat():
    knees = {250: 5, 1_000: 5, 4_000: 5, 16_000: 6, 500_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, rows=[(5, 0, 1, 1.0)])
    assert v.hypothesis == "H1"


def test_decide_picks_h2_when_knee_trends():
    knees = {250: 3, 1_000: 5, 4_000: 7, 16_000: 11, 64_000: 17}
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    read = fit_cost_law("read", samples, [3.0 * s**0.9 for s in samples])
    exec_ = fit_cost_law("exec", samples, [3.0 * s**0.5 for s in samples])
    v = decide(knees, read, exec_, rows=[(5, 0, 1, 1.0)])
    assert v.hypothesis == "H2"


def test_decide_picks_h3_when_pending_backlog_is_material():
    """Pending >= workers/2 means bytes, not worker count, set peak RSS."""
    knees = {250: 5, 1_000: 5, 4_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, rows=[(8, 6, 1, 1.0)])
    assert v.hypothesis == "H3"


def test_decide_returns_none_when_nothing_is_supported():
    """Ambiguous data must not silently default to a hypothesis."""
    knees = {250: 3, 1_000: 9, 4_000: 2}
    samples = [250, 1_000, 4_000]
    read = fit_cost_law("read", samples, [1.0, 5.0, 0.5])
    exec_ = fit_cost_law("exec", samples, [1.0, 0.4, 3.0])
    v = decide(knees, read, exec_, rows=[(8, 0, 1, 1.0)])
    assert v.hypothesis == "none"


def test_extrapolate_flags_the_current_default_as_over_budget():
    """chunk_size=25_000 at 500k samples is ~3.1 GB of packed grid per chunk."""
    v_law = fit_v_law([(25_000, 1.0 + 1e-4 * 25_000), (200_000, 1.0 + 1e-4 * 200_000)])
    samples = [250, 1_000, 4_000]
    read = fit_cost_law("read", samples, [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", samples, [1.0, 1.0, 1.0])
    ram = fit_ram_law([(w, 0, 25_000_000, 100.0 + 3.0 * w * 25.0) for w in (1, 3, 5)])
    out = extrapolate(
        v_law,
        read,
        exec_,
        ram,
        samples=500_000,
        variants=1_000_000_000,
        chunk_size=25_000,
        workers=1,
        format_fields=0,
    )
    assert out["chunk_bytes"] > 3e9
    assert out["predicted_peak_rss_mb"] > 9_000
