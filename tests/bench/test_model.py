import json
import math
from dataclasses import asdict

import pytest

from scripts.bench_svar2.model import (
    _LoadedSweep,
    _ram_rows,
    _resident_chunk_size,
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


# `rows` are (workers, pending_highwater, chunk_bytes, peak_rss_mb). Wherever a
# test needs a row that is NOT byte-material, it uses a tiny `chunk_bytes`
# against a realistic peak RSS -- which is also what the small-S rows of the
# real sweep look like (a few MB of chunk against a ~450 MB process). The
# `pending` values are the STRUCTURALLY REALISTIC ones for their worker count:
# `ReorderBuffer` keeps the w-1 non-head units' chunks buffered, so a w=3 run
# really does sit at pending~5 and a w=7 run higher still. Planting pending=0 at
# w>1, as these tests used to, is a state the harness cannot produce.
_KAPPA3 = RamLaw(base_mb=100.0, kappa=3.0, r2=1.0, n_points=5)


def _immaterial_rows() -> list[tuple[int, int, int, float]]:
    """One row per swept worker count, each with a realistic structural
    backlog but a small chunk: 1 MB is the small-S end of the real sweep
    (S=250 at chunk_size=25_000 is 1.55 MB of packed grid) against a ~450 MB
    process. Backlog share peaks at 3 * 21 * 1 MB / 450 MB = 14% at w=11,
    under the 25% gate -- and 3 * 5 * 1 / 450 = 3.3% at w=3, where the
    old count-based gate read pending=5 as 1.67x workers and fired."""
    return [(w, 2 * (w - 1) + 1, 1_000_000, 450.0) for w in (1, 3, 5, 7, 11)]


def test_decide_picks_h1_when_knee_is_flat():
    knees = {250: 5, 1_000: 5, 4_000: 5, 16_000: 6, 500_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, _immaterial_rows(), _KAPPA3)
    assert v.hypothesis == "H1"


def test_decide_picks_h2_when_knee_trends():
    knees = {250: 3, 1_000: 5, 4_000: 7, 16_000: 11, 64_000: 17}
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    read = fit_cost_law("read", samples, [3.0 * s**0.9 for s in samples])
    exec_ = fit_cost_law("exec", samples, [3.0 * s**0.5 for s in samples])
    v = decide(knees, read, exec_, _immaterial_rows(), _KAPPA3)
    assert v.hypothesis == "H2"


def test_decide_picks_h3_when_the_backlog_is_byte_material():
    """H3(a) is a BYTE share of peak RSS, not a chunk count.

    Planted at the regime the harness exists to probe: S=500_000 with
    `from_vcf`'s hardcoded chunk_size=25_000 is ~3.1 GB per chunk, so even the
    minimum structural one-chunk backlog at w=2 is many times the whole
    process footprint.
    """
    knees = {250: 5, 1_000: 5, 4_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    rows = [(2, 1, 3_125_000_000, 9_800.0)]
    v = decide(knees, read, exec_, rows, _KAPPA3)
    assert v.hypothesis == "H3"
    assert v.evidence["max_backlog_rss_share"] > 0.25


def test_decide_returns_none_when_nothing_is_supported():
    """Ambiguous data must not silently default to a hypothesis."""
    knees = {250: 3, 1_000: 9, 4_000: 2}
    samples = [250, 1_000, 4_000]
    read = fit_cost_law("read", samples, [1.0, 5.0, 0.5])
    exec_ = fit_cost_law("exec", samples, [1.0, 0.4, 3.0])
    v = decide(knees, read, exec_, _immaterial_rows(), _KAPPA3)
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
        cohort_beta=0.0,
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
        cohort_beta=0.0,
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
        cohort_beta=0.0,
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
        cohort_beta=0.0,
    )
    assert math.isclose(out_2e9["extrapolation_factor"], 10_000.0)

    # The old bug: variants / n_points = 1e9 / 4 = 2.5e8, wildly different
    # in both value and meaning from the correct ~5000.
    assert not math.isclose(out_1e9["extrapolation_factor"], 2.5e8, rel_tol=0.5)


# --- I4: predicted_phase1_s must scale the per-variant term with cohort size -


def test_extrapolate_scales_wall_by_the_cohort_size_ratio():
    """`slope_s_per_variant` is fitted at ONE small S (the V-ladder's fixed
    cohort). Applying it unscaled at a different target S silently assumes
    per-variant parse cost doesn't depend on cohort size, which the design
    spec explicitly says is false (2000x more genotype text per record at
    S=500,000 vs S=250). The per-variant term is scaled by
    `(samples / v_law_samples) ** cohort_beta`; the intercept (fill/drain
    overhead) is left unscaled -- nothing here fits how IT moves with S."""
    v_law = VLaw(
        slope_s_per_variant=2.0,
        intercept_s=100.0,
        r2=1.0,
        n_points=4,
        max_extrapolation_factor=1.0,
    )
    flat = CostLaw(name="x", alpha=1.0, beta=0.0, beta_ci95=(0.0, 0.0), n_points=5)
    ram = RamLaw(base_mb=0.0, kappa=0.0, r2=1.0, n_points=2)

    out = extrapolate(
        v_law,
        flat,
        flat,
        ram,
        samples=1_000,
        variants=50,
        chunk_size=10,
        workers=1,
        format_fields=0,
        v_law_samples=250,
        cohort_beta=0.5,
    )
    # cohort_scale = (1000/250)**0.5 = 2.0
    # predicted_wall = 100 + 2.0 * 50 * 2.0 = 300.0 -- intercept untouched,
    # only the slope*variants term is scaled.
    assert math.isclose(out["cohort_scale"], 2.0)
    assert math.isclose(out["predicted_phase1_s"], 300.0)

    same_s = extrapolate(
        v_law,
        flat,
        flat,
        ram,
        samples=250,
        variants=50,
        chunk_size=10,
        workers=1,
        format_fields=0,
        v_law_samples=250,
        cohort_beta=0.5,
    )
    assert math.isclose(same_s["cohort_scale"], 1.0)
    assert math.isclose(same_s["predicted_phase1_s"], 100.0 + 2.0 * 50)


def test_cohort_scale_does_not_come_from_the_utilization_cost_law():
    """I3: `read_law.beta` is fitted on `cpu_shard_pct`, a UTILIZATION
    percentage capped at 100% per thread. At w=1 the reader-bound bottleneck
    pegs near 100% at every S, so `beta_read ~ 0` and
    `(500_000/250) ** beta_read = 1` -- the correction silently evaporated at
    exactly the scale it exists for. `cohort_beta` comes from an absolute
    per-variant cost instead, so a flat read law must NOT flatten it."""
    v_law = VLaw(
        slope_s_per_variant=2.0,
        intercept_s=0.0,
        r2=1.0,
        n_points=4,
        max_extrapolation_factor=1.0,
    )
    # Exactly the degenerate read law a saturated w=1 sweep produces.
    saturated = CostLaw(
        name="read", alpha=100.0, beta=0.0, beta_ci95=(0.0, 0.0), n_points=7
    )
    ram = RamLaw(base_mb=0.0, kappa=0.0, r2=1.0, n_points=2)
    out = extrapolate(
        v_law,
        saturated,
        saturated,
        ram,
        samples=500_000,
        variants=1_000,
        chunk_size=10,
        workers=1,
        format_fields=0,
        v_law_samples=250,
        cohort_beta=0.9,
    )
    # (500_000/250) ** 0.9 = 2000 ** 0.9, ~910x -- not the 1.0x the saturated
    # read law would have produced.
    assert math.isclose(out["cohort_scale"], 2_000**0.9)
    assert out["cohort_scale"] > 100.0


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


# --- C1: the H3 gate must not fire on backlog the architecture forces -------


def _structural_rows() -> list[tuple[int, int, int, float]]:
    """What a real sharded run at w in {3, 7} actually produces.

    `ReorderBuffer::push` (src/shard_exec.rs) releases a chunk on arrival only
    when its ordinal is the head, so the w-1 units ahead of the head hold
    everything they produce until the head unit's `Done`. A real 12-unit,
    w=3, overshard=4 probe log in this repo sustains `pending=5` for the whole
    run; w=7 over the same overshard holds proportionally more. These are the
    numbers the old `pending >= w/2` gate had to survive and did not:
    5/3 = 1.67 and 13/7 = 1.86 both clear 0.5, so it fired on every row of
    every planned sweep. `chunk_bytes` is 1 MB, the small-S end of the sweep.
    """
    return [(3, 5, 1_000_000, 450.0), (7, 13, 1_000_000, 450.0)]


def test_decide_returns_h1_on_planted_h1_data():
    """Regression for C1 -- the reviewer's exact repro, re-planted.

    Planted H1 data must come back H1 even though every row carries the
    structural backlog a w>1 run cannot avoid. The original version of this
    test planted `pending=0` at w in {1,3,5,7}, which passed only because it
    is a state the harness can never actually produce.
    """
    knees = {250: 5, 1_000: 5, 4_000: 5, 16_000: 6, 500_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, _structural_rows(), _KAPPA3)
    assert v.hypothesis == "H1"
    # The count-based gate this replaced would have fired on these same rows.
    assert v.evidence["max_pending_fraction"] >= 0.5


def test_decide_returns_h2_on_planted_h2_data():
    """Same repro as above, planted with H2 data instead."""
    knees = {250: 3, 1_000: 5, 4_000: 7, 16_000: 11, 64_000: 17}
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    read = fit_cost_law("read", samples, [3.0 * s**0.9 for s in samples])
    exec_ = fit_cost_law("exec", samples, [3.0 * s**0.5 for s in samples])
    v = decide(knees, read, exec_, _structural_rows(), _KAPPA3)
    assert v.hypothesis == "H2"
    assert v.evidence["max_pending_fraction"] >= 0.5


def test_decide_h3_fires_when_the_same_backlog_is_byte_material():
    """H3 must stay REACHABLE: the identical structural backlog fires H3 once
    the chunks are big enough for it to matter.

    Same rows as `_structural_rows`, same `pending`, same worker counts --
    only `chunk_bytes` changes, from the 1 MB of a small-S corpus to the
    ~350 MB a resident chunk reaches at S=64,000. That is exactly the
    distinction H3 is about, and the one a count-based gate cannot see.
    """
    knees = {250: 5, 1_000: 5, 4_000: 5, 16_000: 6, 500_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    rows = [(3, 5, 350_000_000, 6_000.0), (7, 13, 350_000_000, 6_000.0)]
    v = decide(knees, read, exec_, rows, _KAPPA3)
    assert v.hypothesis == "H3"


def test_decide_cannot_evaluate_h3a_without_a_ram_law():
    """Fewer than two RAM rows means no fitted kappa, so the byte share is
    unknown -- reported as None, not silently as zero."""
    knees = {250: 5, 1_000: 5, 4_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, _structural_rows(), None)
    assert v.evidence["max_backlog_rss_share"] is None
    assert v.hypothesis == "H1"


def test_every_verdict_carries_the_full_evidence():
    """An H3 verdict used to ship an evidence dict with no knee spread and no
    beta CI, because both were computed after the H3 return paths -- a human
    reading it could not tell whether H1 also held."""
    knees = {250: 5, 1_000: 5, 4_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    byte_material = [(3, 5, 350_000_000, 6_000.0)]
    keys = {"knee_spread", "beta_diff_ci95", "max_backlog_rss_share"}

    h3a = decide(knees, read, exec_, byte_material, _KAPPA3)
    assert h3a.hypothesis == "H3"
    assert keys <= set(h3a.evidence)

    h3b = decide(
        knees,
        read,
        exec_,
        _structural_rows(),
        _KAPPA3,
        contig_counterfactual=(10.0, 20.0),
    )
    assert h3b.hypothesis == "H3"
    assert keys <= set(h3b.evidence)


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


# --- C2: the RAM law's regressor is the TOUCHED chunk, not the nominal one --


def _sweep_of(rows: list[tuple[CorpusManifest, SweepPoint, ProbeRecord]]):
    sweep = _LoadedSweep()
    for m, pt, r in rows:
        sweep.records.append(r)
        sweep.point_of[r.point_id] = pt
        sweep.manifest_of[r.point_id] = m
    return sweep


def test_resident_chunk_size_is_bounded_by_the_corpus():
    assert _resident_chunk_size(25_000, 2_800) == 2_800  # S=500_000 sweep point
    assert _resident_chunk_size(25_000, 1_000_000_000) == 25_000  # biobank target


def test_ram_rows_do_not_charge_untouched_chunk_tail_to_rss():
    """`BitGrid3::zeros` is calloc; untouched pages never become resident (a
    3 GB zeroed allocation adds 0 MB to ru_maxrss on this node). The sweep
    holds S*V fixed, so its large-S corpora are far smaller than
    `chunk_size=25_000` and a nominal chunk_bytes is mostly fiction.

    Plant the TRUE law `rss = 100 + 3 * (w + p) * touched_bytes` across the
    sweep's own S ladder and require the fit to recover kappa=3. Feeding the
    nominal chunk_bytes instead puts two enormous-leverage points (1,562 and
    3,125 vs <=400 for every other row) into the OLS and drags kappa to ~0.23
    -- a ~13x under-estimate, which is what made the harness report
    `from_vcf`'s hardcoded chunk_size as SAFE at biobank scale.
    """
    cells = 1_400_000_000
    rows = []
    for s in (250, 64_000, 250_000, 500_000):
        v = cells // s
        m = _manifest(f"s{s}.manifest.json", samples=s, variants=v, cells=cells)
        pt = _point(f"s{s}.manifest.json", workers=1, chunk_size=25_000)
        touched = m.chunk_bytes * min(25_000, v)
        rows.append(
            (m, pt, _record(pt.point_id, maxrss_mb=100.0 + 3.0 * touched / 1e6))
        )

    ram_rows = _ram_rows(_sweep_of(rows))
    assert math.isclose(fit_ram_law(ram_rows).kappa, 3.0, rel_tol=1e-9)

    nominal = [
        (
            pt.reader_workers,
            r.pending_highwater,
            m.chunk_bytes * pt.chunk_size,
            r.maxrss_mb,
        )
        for (m, pt, r) in rows
    ]
    assert fit_ram_law(nominal).kappa < 0.5


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


# --- Minor 11: H1 needs enough cohort sizes to be a claim about the range ---


def _flat_cost_laws():
    """Cost laws with no S-trend, so H2's CI includes zero and the verdict
    falls through to the H1 test rather than being decided earlier."""
    samples = [250, 1_000, 4_000]
    return (
        fit_cost_law("read", samples, [1.0, 1.0, 1.0]),
        fit_cost_law("exec", samples, [1.0, 1.0, 1.0]),
    )


@pytest.mark.parametrize("knees", [{250: 5}, {250: 5, 1_000: 5}])
def test_decide_refuses_h1_from_too_few_cohort_sizes(knees):
    """H1 claims w* is FLAT ACROSS THE SAMPLE RANGE. One cohort size has
    spread 0 by definition and two can only show a difference, never a trend,
    so neither witnesses that claim -- yet an unguarded `spread <= tolerance`
    reads both as a confident "a static cap suffices, no autotuner needed".

    This is the shape a partly-failed sweep takes: if most w=1 rows yield no
    usable cpu ticks, `_cost_points` drops them and `knees` shrinks. The
    verdict must degrade to `none` and say the flatness is unevaluable, not
    grow more confident as evidence disappears.
    """
    read, exec_ = _flat_cost_laws()
    v = decide(knees, read, exec_, _immaterial_rows(), _KAPPA3)
    assert v.hypothesis == "none"
    assert v.evidence["knee_points"] == len(knees)
    assert v.evidence["knee_spread"] == 0
    assert "not evaluable" in v.rationale


def test_decide_allows_h1_once_the_minimum_cohort_sizes_are_present():
    """Guard boundary: the same flat spread that is unevaluable at two cohort
    sizes IS H1 at three. Without this the guard could pass its own test by
    simply never returning H1."""
    read, exec_ = _flat_cost_laws()
    v = decide({250: 5, 1_000: 5, 4_000: 5}, read, exec_, _immaterial_rows(), _KAPPA3)
    assert v.hypothesis == "H1"
    assert v.evidence["knee_points"] == 3


def test_decide_reports_knee_support_on_every_verdict():
    """`knee_spread=0` from one cohort size and from seven are the same
    number carrying opposite evidence; `knee_points` is what tells them
    apart, so it must ride along on H2 and H3 too, not just H1."""
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    read = fit_cost_law("read", samples, [3.0 * s**0.9 for s in samples])
    exec_ = fit_cost_law("exec", samples, [3.0 * s**0.5 for s in samples])
    h2 = decide(
        {250: 3, 1_000: 5, 4_000: 7, 16_000: 11, 64_000: 17},
        read,
        exec_,
        _immaterial_rows(),
        _KAPPA3,
    )
    assert h2.hypothesis == "H2"
    assert h2.evidence["knee_points"] == 5

    flat_read, flat_exec = _flat_cost_laws()
    h3 = decide(
        {250: 5, 1_000: 5, 4_000: 5},
        flat_read,
        flat_exec,
        [(3, 5, 350_000_000, 6_000.0)],
        _KAPPA3,
    )
    assert h3.hypothesis == "H3"
    assert h3.evidence["knee_points"] == 3


# --- Minor 12: the hold-out gate must score like-for-like ------------------


def _driver_dirs(tmp_path, holdout_record):
    """Minimal scale + vlinear + holdout fixture for `main()`."""
    manifests_dir, plans_dir, results_dir = (
        tmp_path / "manifests",
        tmp_path / "plans",
        tmp_path / "results",
    )
    for d in (manifests_dir, plans_dir, results_dir):
        d.mkdir()

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

    (manifests_dir / "holdout.manifest.json").write_text(
        to_json(_manifest("holdout.manifest.json", samples=1_000, variants=28_000))
    )
    hpt = _point("holdout.manifest.json", workers=1, chunk_size=2_000)
    (plans_dir / "holdout.json").write_text(json.dumps([asdict(hpt)]))
    append_ndjson(results_dir / "holdout.ndjson", holdout_record(hpt.point_id))
    return manifests_dir, plans_dir, results_dir


def _run_main(tmp_path, capsys, monkeypatch, holdout_record) -> str:
    manifests_dir, plans_dir, results_dir = _driver_dirs(tmp_path, holdout_record)
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
    return capsys.readouterr().out


def test_holdout_scores_the_phase1_prediction_against_measured_phase1(
    tmp_path, capsys, monkeypatch
):
    """The V-law is fitted `phase1_s ~ a + b*V`, so the projection is a
    PHASE-1 time. `wall_s` additionally carries the rayon merge tail and
    process startup and is therefore always larger, which made every hold-out
    error one-sidedly inflated -- fed into a 25% gate whose documented
    meaning is "this invalidates the model".

    Plant a record whose two time fields are far apart and assert the scored
    actual is the phase-1 one. 8.0 vs 999.0 is unambiguous: no rounding of
    the wall figure could print as the phase-1 figure.
    """
    out = _run_main(
        tmp_path,
        capsys,
        monkeypatch,
        lambda pid: _record(pid, wall_s=999.0, phase1_s=8.0, maxrss_mb=1_000.0),
    )
    holdout_line = next(ln for ln in out.splitlines() if ln.startswith("HOLD-OUT"))
    assert "phase1 pred=" in holdout_line
    assert "actual=8.0s" in holdout_line
    assert "999" not in holdout_line


def test_holdout_skips_the_time_gate_rather_than_falling_back_to_wall(
    tmp_path, capsys, monkeypatch
):
    """A trace with no per-contig span leaves `phase1_s` at 0. There is then
    nothing commensurable to score the phase-1 projection against, and the
    old fallback -- compare to `wall_s` anyway -- is precisely the
    incommensurable comparison this fix removes. Say it was skipped instead,
    and do not let it raise MODEL FAILURE on its own.
    """
    out = _run_main(
        tmp_path,
        capsys,
        monkeypatch,
        # RSS planted on the fitted line so the rss half of the gate is quiet
        # and any MODEL FAILURE would have to come from the time half.
        lambda pid: _record(pid, wall_s=999.0, phase1_s=0.0, maxrss_mb=1_000.0),
    )
    holdout_line = next(ln for ln in out.splitlines() if ln.startswith("HOLD-OUT"))
    assert "NOT scored against wall_s" in holdout_line
    assert "999" not in holdout_line


def test_extrapolate_does_not_expose_a_wall_named_key():
    """Naming is the fix: `predicted_wall_s` invited exactly one wrong
    comparison and got it. Nothing may reintroduce the name."""
    law_v = VLaw(
        intercept_s=1.0,
        slope_s_per_variant=2.0,
        r2=1.0,
        n_points=4,
        max_extrapolation_factor=1.0,
    )
    read, exec_ = _flat_cost_laws()
    out = extrapolate(
        law_v,
        read,
        exec_,
        _KAPPA3,
        samples=250,
        variants=50,
        chunk_size=64,
        workers=1,
        format_fields=0,
        v_law_samples=250,
        cohort_beta=0.0,
    )
    assert "predicted_phase1_s" in out
    assert "predicted_wall_s" not in out
