import json
import math
from dataclasses import asdict

import pytest

from scripts.bench_svar2.model import (
    decide,
    extrapolate,
    fit_cost_law,
    fit_ram_law,
    fit_v_law,
    knee_from_probe,
    main,
)
from scripts.bench_svar2.records import (
    CorpusManifest,
    CostLaw,
    ProbeRecord,
    RamLaw,
    SweepPoint,
    VLaw,
    append_ndjson,
    to_json,
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
        v_law_samples=25_000,
    )
    assert out["chunk_bytes"] > 3e9
    assert out["predicted_peak_rss_mb"] > 9_000


# --- I2: extrapolate must project the pending term it was fitted with ------


def test_extrapolate_includes_the_pending_term():
    """`fit_ram_law` regresses on `(workers + pending) * chunk_bytes`
    (model.py:fit_ram_law). Before this fix `extrapolate` dropped `pending`
    entirely, so a measured reorder backlog (H3's whole subject) had zero
    effect on the projected peak RSS."""
    ram = RamLaw(base_mb=100.0, kappa=3.0, r2=1.0, n_points=5)
    v_law = VLaw(
        slope_s_per_variant=0.0,
        intercept_s=0.0,
        r2=1.0,
        n_points=2,
        max_extrapolation_factor=1.0,
    )
    flat = CostLaw(name="x", alpha=1.0, beta=0.0, beta_ci95=(0.0, 0.0), n_points=3)
    # samples=1000, format_fields=0 -> grid=(1000*2)//8=250, chunk_size=1000
    # -> chunk_bytes = 250_000.
    kwargs = dict(
        v_law=v_law,
        read_law=flat,
        exec_law=flat,
        ram_law=ram,
        samples=1_000,
        variants=0,
        chunk_size=1_000,
        workers=2,
        format_fields=0,
        v_law_samples=1_000,
    )
    no_pending = extrapolate(**kwargs, pending=0)
    with_pending = extrapolate(**kwargs, pending=6)
    assert math.isclose(no_pending["predicted_peak_rss_mb"], 101.5)
    assert math.isclose(with_pending["predicted_peak_rss_mb"], 106.0)
    assert with_pending["predicted_peak_rss_mb"] > no_pending["predicted_peak_rss_mb"]


# --- I3: extrapolation_factor must be variants / (max V actually fitted) ---


def test_extrapolate_extrapolation_factor_is_variants_over_max_fitted_v():
    """The old formula was `variants / n_points` -- a variant count divided by
    a POINT COUNT. The correct quantity is variants / (largest V the V-law
    was fitted against), recovered from `VLaw.max_extrapolation_factor`
    (records.py, frozen) since `VLaw` carries no raw max-V field."""
    variants = [25_000, 50_000, 100_000, 200_000]
    walls = [1.0 + 1e-4 * v for v in variants]
    v_law = fit_v_law(list(zip(variants, walls)))
    flat = CostLaw(name="x", alpha=1.0, beta=0.0, beta_ci95=(0.0, 0.0), n_points=3)
    ram = RamLaw(base_mb=0.0, kappa=0.0, r2=1.0, n_points=2)

    out_1e9 = extrapolate(
        v_law,
        flat,
        flat,
        ram,
        samples=1,
        variants=1_000_000_000,
        chunk_size=1,
        workers=0,
        format_fields=0,
        v_law_samples=25_000,
    )
    # fit_v_law hardcodes max_extrapolation_factor against 1e9 variants, so at
    # exactly 1e9 the factor must equal max_extrapolation_factor itself:
    # 1e9 / 200_000 = 5000.
    assert math.isclose(out_1e9["extrapolation_factor"], 5_000.0)

    out_2e9 = extrapolate(
        v_law,
        flat,
        flat,
        ram,
        samples=1,
        variants=2_000_000_000,
        chunk_size=1,
        workers=0,
        format_fields=0,
        v_law_samples=25_000,
    )
    assert math.isclose(out_2e9["extrapolation_factor"], 10_000.0)

    # The old bug: variants / n_points = 1e9 / 4 = 2.5e8, wildly different
    # in both value and meaning from the correct ~5000.
    assert not math.isclose(out_1e9["extrapolation_factor"], 2.5e8, rel_tol=0.5)


# --- I4: predicted_wall_s must scale the per-variant term with cohort size -


def test_extrapolate_scales_wall_by_the_cohort_size_ratio():
    """`slope_s_per_variant` is fitted at ONE small S (the V-ladder's fixed
    cohort). Applying it unscaled at a different target S silently assumes
    per-variant parse cost doesn't depend on cohort size, which the design
    spec explicitly says is false (2000x more genotype text per record at
    S=500,000 vs S=250). The fix scales the per-variant term by
    `(samples / v_law_samples) ** read_law.beta`; the intercept (fill/drain
    overhead) is left unscaled -- nothing here fits how IT moves with S."""
    v_law = VLaw(
        slope_s_per_variant=2.0,
        intercept_s=100.0,
        r2=1.0,
        n_points=4,
        max_extrapolation_factor=1.0,
    )
    read = CostLaw(name="read", alpha=1.0, beta=0.5, beta_ci95=(0.4, 0.6), n_points=5)
    exec_ = CostLaw(name="exec", alpha=1.0, beta=0.5, beta_ci95=(0.4, 0.6), n_points=5)
    ram = RamLaw(base_mb=0.0, kappa=0.0, r2=1.0, n_points=2)

    out = extrapolate(
        v_law,
        read,
        exec_,
        ram,
        samples=1_000,
        variants=50,
        chunk_size=10,
        workers=1,
        format_fields=0,
        v_law_samples=250,
    )
    # cohort_scale = (1000/250)**0.5 = 2.0
    # predicted_wall = 100 + 2.0 * 50 * 2.0 = 300.0 -- intercept untouched,
    # only the slope*variants term is scaled.
    assert math.isclose(out["cohort_scale"], 2.0)
    assert math.isclose(out["predicted_wall_s"], 300.0)

    same_s = extrapolate(
        v_law,
        read,
        exec_,
        ram,
        samples=250,
        variants=50,
        chunk_size=10,
        workers=1,
        format_fields=0,
        v_law_samples=250,
    )
    assert math.isclose(same_s["cohort_scale"], 1.0)
    assert math.isclose(same_s["predicted_wall_s"], 100.0 + 2.0 * 50)


# --- minor: degenerate fits must not report zero-width (maximally
# confident) confidence intervals --------------------------------------


def test_cost_law_degenerate_two_points_has_unbounded_ci():
    """n<=2 always fits exactly (zero residual), so there is no information
    to bound the true slope from. Before this fix `_linfit` reported
    stderr=0.0 in this case, collapsing `beta_ci95` to a single point --
    which made `decide` report H2 on a difference of any size, including
    pure noise. The fix reports unbounded uncertainty instead."""
    law = fit_cost_law("read", [250, 1_000], [3.5 * 250**0.8, 3.5 * 1_000**0.8])
    lo, hi = law.beta_ci95
    assert lo == -math.inf
    assert hi == math.inf


# --- C1: the H3 gate must not fire on the pending-gauge's structural floor -


def test_decide_returns_h1_on_planted_h1_data():
    """Regression for C1 -- the reviewer's exact repro.

    The gauge used to sample AFTER inserting the just-read chunk, so
    `pending_highwater` floored at 1 on every sharded row even with zero
    reordering. At w=1 that made `pending/workers == 1.0 >= 0.5`
    unconditionally, so the H3 gate fired on the first row of every sweep
    regardless of the H1/H2 evidence: planted H1 data, got H3 anyway.

    The gauge now samples BEFORE the insert, so a no-backlog run reports 0 and
    the planted hypothesis has to survive on its own evidence.
    """
    knees = {250: 5, 1_000: 5, 4_000: 5, 16_000: 6, 500_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    rows = [(w, 0, 1, 1.0) for w in (1, 3, 5, 7)]  # no backlog observed
    v = decide(knees, read, exec_, rows=rows)
    assert v.hypothesis == "H1"


def test_decide_returns_h2_on_planted_h2_data():
    """Same repro as above, planted with H2 data instead."""
    knees = {250: 3, 1_000: 5, 4_000: 7, 16_000: 11, 64_000: 17}
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    read = fit_cost_law("read", samples, [3.0 * s**0.9 for s in samples])
    exec_ = fit_cost_law("exec", samples, [3.0 * s**0.5 for s in samples])
    rows = [(w, 0, 1, 1.0) for w in (1, 3, 5, 7)]
    v = decide(knees, read, exec_, rows=rows)
    assert v.hypothesis == "H2"


def test_decide_h3_fires_on_a_genuine_backlog():
    """H3 must still be reachable. With the corrected 0-based gauge, a
    `pending_highwater` of 1 at w=1 is a real one-chunk backlog, not an
    artifact, so it is legitimate H3 evidence -- which is precisely why the
    same input had to be read as H1/H2 back when 1 was the floor."""
    knees = {250: 5, 1_000: 5, 4_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, rows=[(1, 1, 1, 1.0)])
    assert v.hypothesis == "H3"


# --- I1: the missing results.ndjson + manifests -> laws + verdict entry ----
# point ------------------------------------------------------------------


def _manifest(name: str, **overrides) -> CorpusManifest:
    base = dict(
        path=name,
        samples=250,
        variants=25_000,
        contigs=("chr22",),
        format_fields=(),
        ploidy=2,
        cells=250 * 25_000,
        compressed_bytes=1,
        seed=1,
        generator_version=1,
    )
    base.update(overrides)
    return CorpusManifest(**base)


def _point(
    corpus: str, workers: int, chunk_size: int = 1_000, **overrides
) -> SweepPoint:
    base = dict(
        corpus=corpus,
        reader_workers=workers,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=4,
        chunk_size=chunk_size,
        threads=4,
        reps=1,
    )
    base.update(overrides)
    return SweepPoint(**base)


def _record(point_id: str, **overrides) -> ProbeRecord:
    base = dict(
        point_id=point_id,
        ok=True,
        wall_s=10.0,
        phase1_s=8.0,
        cpu_s=10.0,
        maxrss_mb=1_000.0,
        digest="d",
        dense_cap=1,
        dense_occupancy=(),
        cpu_shard_pct=(),
        cpu_exec_pct=(),
        pending_highwater=0,
        pending_bytes_highwater=0,
        shard_unit_secs=(),
    )
    base.update(overrides)
    return ProbeRecord(**base)


def test_main_end_to_end(tmp_path, capsys, monkeypatch):
    manifests_dir, plans_dir, results_dir = (
        tmp_path / "manifests",
        tmp_path / "plans",
        tmp_path / "results",
    )
    for d in (manifests_dir, plans_dir, results_dir):
        d.mkdir()

    # scale sweep: S in {250, 1000, 4000}, one w=1 run per S, planted power
    # laws (same shape as test_decide_picks_h2_when_knee_trends) so the
    # driver's fitted verdict is deterministic. pending_highwater=0 at every
    # point: the corrected gauge reports 0 for a run with no reorder backlog,
    # so the planted H2 evidence decides the verdict rather than the H3 gate.
    scale_points = []
    for s in (250, 1_000, 4_000):
        (manifests_dir / f"s{s}.manifest.json").write_text(
            to_json(_manifest(f"s{s}.manifest.json", samples=s))
        )
        pt = _point(f"s{s}.manifest.json", workers=1)
        scale_points.append(pt)
        append_ndjson(
            results_dir / "scale.ndjson",
            _record(
                pt.point_id,
                cpu_shard_pct=(3.0 * s**0.9,),
                cpu_exec_pct=(3.0 * s**0.5,),
                pending_highwater=0,
                maxrss_mb=500.0 + s,
            ),
        )
    (plans_dir / "scale.json").write_text(json.dumps([asdict(p) for p in scale_points]))

    # vlinear ladder: V in {25_000, 50_000} at one fixed small S.
    v_points = []
    for v in (25_000, 50_000):
        name = f"vlin_v{v}.manifest.json"
        (manifests_dir / name).write_text(
            to_json(_manifest(name, samples=250, variants=v))
        )
        pt = _point(name, workers=1, chunk_size=64)
        v_points.append(pt)
        append_ndjson(
            results_dir / "vlinear.ndjson",
            _record(pt.point_id, phase1_s=1.0 + 1e-4 * v, maxrss_mb=300.0),
        )
    (plans_dir / "vlinear.json").write_text(json.dumps([asdict(p) for p in v_points]))

    # holdout: actuals deliberately absurd so the >25% error gate must fire.
    (manifests_dir / "holdout.manifest.json").write_text(
        to_json(
            _manifest(
                "holdout.manifest.json",
                samples=100_000,
                variants=28_000,
                format_fields=("DP", "GQ", "AD"),
            )
        )
    )
    hpt = _point("holdout.manifest.json", workers=1, chunk_size=2_000)
    (plans_dir / "holdout.json").write_text(json.dumps([asdict(hpt)]))
    append_ndjson(
        results_dir / "holdout.ndjson",
        _record(hpt.point_id, wall_s=1e-6, maxrss_mb=1e-6, pending_highwater=0),
    )

    # contig.json/ndjson intentionally absent -- must degrade gracefully.

    monkeypatch.setattr(
        "sys.argv",
        [
            "model.py",
            "--results",
            str(results_dir),
            "--manifests",
            str(manifests_dir),
            "--plans",
            str(plans_dir),
        ],
    )
    main()

    out = capsys.readouterr().out
    assert "VERDICT: H2" in out
    assert "contig: 0 usable record(s)" in out
    # contig.ndjson/plan.json are both absent; reported clearly, not crashed.
    assert "no results file at" in out
    assert "EXTRAPOLATION" in out
    assert "HOLD-OUT" in out
    assert "MODEL FAILURE" in out
